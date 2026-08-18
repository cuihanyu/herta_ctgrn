"""Cell-neighborhood context samplers for local HERTA training and inference."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from herta.data.neighborhood import build_wnn_neighbors
from herta.data.sampling.base import BaseSubgraphSampler, EdgeType, SamplerConfig, SubgraphBatch


def _as_matrix(value: object, name: str, n_cells: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != n_cells or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [n_cells, n_features].")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.asarray(array, dtype=np.float32)


def _embedding_neighbors(
    embedding: np.ndarray,
    n_neighbors: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit = normalize(embedding, norm="l2", axis=1).astype(np.float32, copy=False)
    model = NearestNeighbors(
        n_neighbors=min(n_neighbors + 1, len(unit)), metric="cosine"
    ).fit(unit)
    distances, indices = model.kneighbors(unit)
    final_ids = np.empty((len(unit), n_neighbors), dtype=np.int64)
    final_similarity = np.empty((len(unit), n_neighbors), dtype=np.float32)
    for row in range(len(unit)):
        keep = indices[row] != row
        ids = indices[row][keep][:n_neighbors]
        similarity = (1.0 - distances[row][keep])[:n_neighbors]
        if len(ids) != n_neighbors:
            raise RuntimeError("Unable to construct the requested non-self cell neighborhood.")
        final_ids[row] = ids
        final_similarity[row] = similarity
    scaled = final_similarity / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)
    weights /= weights.sum(axis=1, keepdims=True)
    return final_ids, final_similarity, weights.astype(np.float32)


class CellKNNSampler(BaseSubgraphSampler):
    """Construct a local state graph from seed cells and nearest cells.

    Neighborhoods can be derived either from a learned ``z_cell`` embedding or
    from paired RNA PCA and ATAC LSI representations.  The sampler only emits
    observed state relations.  A small set of isolated feature nodes is added
    when needed so link reconstruction always has eligible negative targets;
    these nodes introduce no message-passing edges.
    """

    def __init__(
        self,
        data,
        config: SamplerConfig | None = None,
        *,
        z_cell: object | None = None,
        rna_pca: object | None = None,
        atac_lsi: object | None = None,
        seed_cell_ids: object | None = None,
    ) -> None:
        super().__init__(data, config)
        if "cell" not in self.data.node_types:
            raise ValueError("CellKNNSampler requires a cell node type.")
        n_cells = int(self.data["cell"].num_nodes)
        if self.config.n_neighbors >= n_cells:
            raise ValueError("n_neighbors must be smaller than the number of cells.")
        has_z = z_cell is not None
        has_modalities = rna_pca is not None or atac_lsi is not None
        if has_z == has_modalities:
            raise ValueError("Provide exactly one of z_cell or paired RNA PCA + ATAC LSI.")
        if has_z:
            embedding = _as_matrix(z_cell, "z_cell", n_cells)
            ids, similarity, weights = _embedding_neighbors(
                embedding, self.config.n_neighbors, self.config.neighbor_temperature
            )
            source = "z_cell"
        else:
            if rna_pca is None or atac_lsi is None:
                raise ValueError("Both rna_pca and atac_lsi are required for multimodal WNN.")
            rna = _as_matrix(rna_pca, "rna_pca", n_cells)
            atac = _as_matrix(atac_lsi, "atac_lsi", n_cells)
            result = build_wnn_neighbors(
                rna,
                atac,
                n_neighbors=self.config.n_neighbors,
                candidate_neighbors=min(self.config.candidate_neighbors, n_cells - 1),
                neighbor_temperature=self.config.neighbor_temperature,
            )
            ids = result.neighbor_indices
            similarity = result.neighbor_similarities
            weights = result.neighbor_weights
            source = "rna_pca_atac_lsi_wnn"
        self.neighbor_indices = torch.as_tensor(ids, dtype=torch.long)
        self.neighbor_similarities = torch.as_tensor(similarity, dtype=torch.float32)
        self.neighbor_weights = torch.as_tensor(weights, dtype=torch.float32)
        self.neighborhood_source = source
        self._default_seed_cell_ids = self._validate_seed_ids(seed_cell_ids, allow_none=True)
        self._validate_state_relations()
        self._source_edge_index = self._build_source_edge_index()
        self._aligned_reverse_types = self._validate_aligned_reverse_relations()

    def _validate_aligned_reverse_relations(self) -> dict[EdgeType, EdgeType]:
        """Require HERTA state reverse edges to share forward edge IDs."""

        available = set(self.data.edge_types)
        aligned: dict[EdgeType, EdgeType] = {}
        for edge_type in self.config.balanced_edge_types:
            reverse = self._reverse_edge_type(edge_type, available)
            if reverse is None:
                continue
            forward_index = self.data[edge_type].edge_index
            reverse_index = self.data[reverse].edge_index
            if not torch.equal(reverse_index, forward_index.flip(0)):
                raise ValueError(
                    f"Reverse relation {reverse} must align one-to-one with {edge_type}."
                )
            aligned[edge_type] = reverse
        return aligned

    def _build_source_edge_index(
        self,
    ) -> dict[EdgeType, tuple[torch.Tensor, torch.Tensor]]:
        """Cache source-sorted edge IDs and row pointers for local extraction."""

        n_cells = int(self.data["cell"].num_nodes)
        index: dict[EdgeType, tuple[torch.Tensor, torch.Tensor]] = {}
        for edge_type in self.config.balanced_edge_types:
            eligible = self._eligible_edge_ids(edge_type)
            sources = self.data[edge_type].edge_index[0, eligible]
            order = torch.argsort(sources, stable=True)
            sorted_edge_ids = eligible[order]
            counts = torch.bincount(sources[order], minlength=n_cells)
            row_ptr = torch.zeros(n_cells + 1, dtype=torch.long)
            row_ptr[1:] = torch.cumsum(counts, dim=0)
            index[edge_type] = (sorted_edge_ids, row_ptr)
        return index

    def _validate_state_relations(self) -> None:
        missing = set(self.config.balanced_edge_types).difference(self.data.edge_types)
        if missing:
            raise ValueError(f"State relations are absent from the graph: {sorted(missing)}")
        invalid = [edge_type for edge_type in self.config.balanced_edge_types if edge_type[0] != "cell"]
        if invalid:
            raise ValueError(f"CellKNNSampler relations must use cell sources: {invalid}")

    def _validate_seed_ids(self, values: object | None, *, allow_none: bool) -> torch.Tensor | None:
        if values is None:
            return None if allow_none else torch.arange(int(self.data["cell"].num_nodes))
        ids = torch.as_tensor(values, dtype=torch.long).flatten()
        if ids.numel() == 0:
            raise ValueError("seed_cell_ids must not be empty.")
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError("seed_cell_ids contains duplicates.")
        n_cells = int(self.data["cell"].num_nodes)
        if bool((ids < 0).any()) or bool((ids >= n_cells).any()):
            raise IndexError("seed_cell_ids contains an out-of-range cell ID.")
        return ids

    def _resolve_seeds(self, seed_cell_ids: object | None) -> torch.Tensor:
        resolved = self._validate_seed_ids(seed_cell_ids, allow_none=True)
        if resolved is not None:
            return resolved
        if self._default_seed_cell_ids is not None:
            return self._default_seed_cell_ids.clone()
        return torch.arange(int(self.data["cell"].num_nodes), dtype=torch.long)

    def _context_members(self, seeds: torch.Tensor) -> torch.Tensor:
        return torch.cat([seeds[:, None], self.neighbor_indices[seeds]], dim=1)

    def _state_edge_ids(self, context_cells: torch.Tensor) -> dict[EdgeType, torch.Tensor]:
        edge_ids: dict[EdgeType, torch.Tensor] = {
            edge_type: torch.empty(0, dtype=torch.long) for edge_type in self.data.edge_types
        }
        for edge_type in self.config.balanced_edge_types:
            sorted_edge_ids, row_ptr = self._source_edge_index[edge_type]
            selected_parts = [
                sorted_edge_ids[row_ptr[cell_id] : row_ptr[cell_id + 1]]
                for cell_id in context_cells.tolist()
            ]
            selected = (
                torch.cat(selected_parts)
                if selected_parts
                else torch.empty(0, dtype=torch.long)
            )
            edge_ids[edge_type] = selected
        if self.config.include_reverse_edges:
            for edge_type in self.config.balanced_edge_types:
                reverse = self._aligned_reverse_types.get(edge_type)
                if reverse is not None:
                    edge_ids[reverse] = edge_ids[edge_type].clone()
        return edge_ids

    def _node_ids_from_context(
        self,
        context_cells: torch.Tensor,
        edge_ids: Mapping[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        parts: dict[str, list[torch.Tensor]] = {
            node_type: [] for node_type in self.data.node_types
        }
        parts["cell"].append(context_cells)
        for edge_type, selected in edge_ids.items():
            if selected.numel() == 0:
                continue
            edges = self.data[edge_type].edge_index[:, selected]
            parts[edge_type[0]].append(edges[0])
            parts[edge_type[2]].append(edges[1])
        node_ids = {
            node_type: (
                torch.unique(torch.cat(values), sorted=True)
                if values
                else torch.empty(0, dtype=torch.long)
            )
            for node_type, values in parts.items()
        }
        return self._add_negative_feature_reservoir(context_cells, node_ids)

    def _add_negative_feature_reservoir(
        self,
        context_cells: torch.Tensor,
        node_ids: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Add isolated feature candidates without inventing observed edges."""

        added: dict[str, list[torch.Tensor]] = {
            node_type: [] for node_type in self.data.node_types
        }
        for edge_type in self.config.balanced_edge_types:
            target_type = edge_type[2]
            num_targets = int(self.data[target_type].num_nodes)
            edge_index = self.data[edge_type].edge_index
            for cell_id in context_cells.tolist():
                positive = edge_index[1, edge_index[0] == cell_id]
                if positive.numel() >= num_targets:
                    raise ValueError(
                        f"Cell {cell_id} has no eligible negative {target_type} target."
                    )
                positive_set = set(map(int, positive.tolist()))
                start = (self.config.seed + 104729 * int(cell_id)) % num_targets
                candidates: list[int] = []
                for offset in range(num_targets):
                    candidate = (start + offset) % num_targets
                    if candidate not in positive_set:
                        candidates.append(candidate)
                        if len(candidates) == self.config.negative_ratio:
                            break
                added[target_type].append(torch.tensor(candidates, dtype=torch.long))
        for node_type, values in added.items():
            if values:
                node_ids[node_type] = torch.unique(
                    torch.cat([node_ids[node_type], *values]), sorted=True
                )
        return node_ids

    def _sample_context(self, seed_cell_ids: object | None) -> tuple[SubgraphBatch, torch.Tensor]:
        seeds = self._resolve_seeds(seed_cell_ids)
        members = self._context_members(seeds)
        context_cells = torch.unique(members.flatten(), sorted=True)
        edge_ids = self._state_edge_ids(context_cells)
        node_ids = self._node_ids_from_context(context_cells, edge_ids)
        batch = self._build_batch(
            node_ids,
            edge_ids,
            {
                "sampler": type(self).__name__,
                "seed": self.config.seed,
                "neighborhood_source": self.neighborhood_source,
                "n_neighbors": self.config.n_neighbors,
                "seed_cell_ids": seeds.clone(),
                "context_member_ids": members.clone(),
                "neighbor_similarities": self.neighbor_similarities[seeds].clone(),
                "neighbor_weights": self.neighbor_weights[seeds].clone(),
                "context_cell_count": int(context_cells.numel()),
                "role": "local_context_only",
            },
        )
        return batch, members

    def sample(self, seed_cell_ids: object | None = None) -> SubgraphBatch:
        batch, _ = self._sample_context(seed_cell_ids)
        return batch


class MetacellContextSampler(CellKNNSampler):
    """Aggregate RNA/ATAC activity over each seed cell and nearest cells.

    The output remains a regular HERTA subgraph.  Aggregated activities are
    attached to ``SubgraphBatch.context_features`` and do not create metacell
    graph nodes or cell-specific model parameters.
    """

    def __init__(
        self,
        data,
        config: SamplerConfig | None = None,
        *,
        rna_activity: object,
        atac_activity: object,
        z_cell: object | None = None,
        rna_pca: object | None = None,
        atac_lsi: object | None = None,
        seed_cell_ids: object | None = None,
    ) -> None:
        super().__init__(
            data,
            config,
            z_cell=z_cell,
            rna_pca=rna_pca,
            atac_lsi=atac_lsi,
            seed_cell_ids=seed_cell_ids,
        )
        n_cells = int(self.data["cell"].num_nodes)
        self.rna_activity = torch.as_tensor(
            _as_matrix(rna_activity, "rna_activity", n_cells), dtype=torch.float32
        )
        self.atac_activity = torch.as_tensor(
            _as_matrix(atac_activity, "atac_activity", n_cells), dtype=torch.float32
        )

    def sample(self, seed_cell_ids: object | None = None) -> SubgraphBatch:
        batch, members = self._sample_context(seed_cell_ids)
        rna_mean = self.rna_activity[members].mean(dim=1)
        atac_mean = self.atac_activity[members].mean(dim=1)
        batch.context_features = {
            "rna_activity": rna_mean,
            "atac_activity": atac_mean,
        }
        diagnostics = dict(batch.diagnostics)
        diagnostics.update(
            {
                "aggregation": "arithmetic_mean",
                "members_per_metacell": int(members.shape[1]),
                "creates_metacell_nodes": False,
                "trains_cell_specific_models": False,
                "role": "inference_context_only",
            }
        )
        batch.diagnostics = diagnostics
        return batch
