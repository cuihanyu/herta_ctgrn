"""HERTA cell-context-conditioned regulatory path sampler."""

from __future__ import annotations

import time
from types import MappingProxyType

import numpy as np
import torch
from scipy import sparse

from herta.data.sampling.base import EdgeType, SamplerConfig, SubgraphBatch
from herta.data.sampling.baseline import EdgeTypeBalancedSampler
from herta.data.sampling.context import CellKNNSampler, _as_matrix
from herta.data.sampling.path import (
    RegulatoryPathSampler,
    complete_paths,
    relation_edge_weight,
)


class HERTAContextPathSampler(CellKNNSampler):
    """Sample cell state and complete TF->peak->gene paths in one local graph."""

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
        cell_types: object | None = None,
        all_positive_edges: dict[EdgeType, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(
            data,
            config,
            z_cell=z_cell,
            rna_pca=rna_pca,
            atac_lsi=atac_lsi,
            seed_cell_ids=seed_cell_ids,
        )
        missing = set(self.config.path_edge_types).difference(self.data.edge_types)
        if missing:
            raise ValueError(f"Regulatory relations are absent from the graph: {sorted(missing)}")
        n_cells = int(self.data["cell"].num_nodes)
        self.rna_activity = self._activity_matrix(rna_activity, "rna_activity", n_cells)
        self.atac_activity = self._activity_matrix(atac_activity, "atac_activity", n_cells)
        if self.rna_activity.shape[1] != int(self.data["gene"].num_nodes):
            raise ValueError("rna_activity columns must align with gene nodes.")
        if self.atac_activity.shape[1] != int(self.data["peak"].num_nodes):
            raise ValueError("atac_activity columns must align with peak nodes.")
        if cell_types is None:
            self.cell_types = None
        else:
            labels = np.asarray(cell_types).reshape(-1)
            if len(labels) != n_cells:
                raise ValueError("cell_types must align with cell nodes.")
            self.cell_types = labels
        self.all_positive_edges = all_positive_edges

    @staticmethod
    def _activity_matrix(value: object, name: str, n_cells: int):
        if sparse.issparse(value):
            matrix = sparse.csr_matrix(value, dtype=np.float32)
            if matrix.shape[0] != n_cells or matrix.shape[1] < 1:
                raise ValueError(f"{name} must have shape [n_cells, n_features].")
            if not np.isfinite(matrix.data).all():
                raise ValueError(f"{name} must contain only finite values.")
            return matrix
        return torch.as_tensor(_as_matrix(value, name, n_cells), dtype=torch.float32)

    @staticmethod
    def _mean_rows(matrix, row_ids: torch.Tensor) -> torch.Tensor:
        ids = row_ids.detach().cpu().numpy()
        if sparse.issparse(matrix):
            return torch.as_tensor(np.asarray(matrix[ids].mean(axis=0)).ravel(), dtype=torch.float32)
        return matrix[row_ids].mean(dim=0)

    def _context_activity(self, members: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.use_metacell_context:
            rows = members.flatten()
            return self._mean_rows(self.rna_activity, rows), self._mean_rows(self.atac_activity, rows)
        seeds = members[:, 0]
        return self._mean_rows(self.rna_activity, seeds), self._mean_rows(self.atac_activity, seeds)

    @staticmethod
    def _top_active(values: torch.Tensor, count: int) -> torch.Tensor:
        count = min(count, values.numel())
        return torch.argsort(values, descending=True, stable=True)[:count]

    def _rank_context_paths(
        self,
        active_genes: torch.Tensor,
        active_peaks: torch.Tensor,
        gene_activity: torch.Tensor,
        peak_activity: torch.Tensor,
    ) -> tuple[list[tuple[int, int, int, int, int]], int]:
        tp_type, pg_type = self.config.path_edge_types
        all_paths = complete_paths(
            self.data,
            tp_type,
            pg_type,
            self._eligible_edge_ids(tp_type),
            self._eligible_edge_ids(pg_type),
        )
        gene_set = set(active_genes.tolist())
        peak_set = set(active_peaks.tolist())
        candidates = [
            path for path in all_paths if path[1] in peak_set and path[2] in gene_set
        ]
        candidates.sort(
            key=lambda path: (
                -float(gene_activity[path[0]] + 1e-8)
                * float(peak_activity[path[1]] + 1e-8)
                * float(gene_activity[path[2]] + 1e-8)
                * relation_edge_weight(self.data, tp_type, path[3])
                * relation_edge_weight(self.data, pg_type, path[4]),
                path,
            )
        )
        return candidates[: self.config.path_budget], len(all_paths)

    def _balanced_state_edges(
        self,
        context_cells: torch.Tensor,
        active_genes: torch.Tensor,
        active_peaks: torch.Tensor,
        target_budget: int,
    ) -> dict[EdgeType, torch.Tensor]:
        active_by_type = {"gene": active_genes, "peak": active_peaks}
        available: dict[EdgeType, torch.Tensor] = {}
        for edge_type in self.config.balanced_edge_types:
            eligible = self._eligible_edge_ids(edge_type)
            edges = self.data[edge_type].edge_index[:, eligible]
            mask = torch.isin(edges[0], context_cells) & torch.isin(
                edges[1], active_by_type[edge_type[2]]
            )
            available[edge_type] = eligible[mask]
        budget = min([target_budget, *(int(ids.numel()) for ids in available.values())])
        if self.config.edges_per_type is not None:
            budget = min(budget, self.config.edges_per_type)
        if budget < 1:
            raise ValueError("Context has no balanced cell-gene/cell-peak state edges.")
        generator = self._generator()
        return {
            edge_type: self._sample_relation(edge_type, ids, budget, generator)
            for edge_type, ids in available.items()
        }

    # Reuse the source-covering relation sampler from the Stage 1 baseline.
    _sample_relation = EdgeTypeBalancedSampler._sample_relation

    def sample(self, seed_cell_ids: object | None = None) -> SubgraphBatch:
        started = time.perf_counter()
        if seed_cell_ids is None and self._default_seed_cell_ids is None:
            n_cells = int(self.data["cell"].num_nodes)
            seeds = torch.randperm(n_cells, generator=self._generator())[
                : min(self.config.seed_cell_count, n_cells)
            ]
        else:
            seeds = self._resolve_seeds(seed_cell_ids)
        members = self._context_members(seeds)
        context_cells = torch.unique(members.flatten(), sorted=True)
        gene_activity, peak_activity = self._context_activity(members)
        active_genes = self._top_active(gene_activity, self.config.active_genes)
        active_peaks = self._top_active(peak_activity, self.config.active_peaks)
        paths, full_path_count = self._rank_context_paths(
            active_genes, active_peaks, gene_activity, peak_activity
        )
        if not paths:
            raise ValueError("No complete regulatory path connects the active context features.")

        helper = RegulatoryPathSampler(
            self.data,
            self.config,
            all_positive_edges=self.all_positive_edges,
        )
        regulatory_edge_ids = helper._edge_ids_for_paths(paths)
        tp_type, pg_type = self.config.path_edge_types
        path_genes = torch.as_tensor(
            sorted({path[0] for path in paths} | {path[2] for path in paths}),
            dtype=torch.long,
        )
        path_peaks = torch.as_tensor(
            sorted({path[1] for path in paths}), dtype=torch.long
        )
        state_genes = torch.unique(torch.cat([active_genes, path_genes]), sorted=True)
        state_peaks = torch.unique(torch.cat([active_peaks, path_peaks]), sorted=True)
        state_edge_ids = self._balanced_state_edges(
            context_cells,
            state_genes,
            state_peaks,
            max(regulatory_edge_ids[tp_type].numel(), regulatory_edge_ids[pg_type].numel()),
        )
        edge_ids = {
            edge_type: torch.empty(0, dtype=torch.long) for edge_type in self.data.edge_types
        }
        edge_ids.update(regulatory_edge_ids)
        edge_ids.update(state_edge_ids)
        if self.config.include_reverse_edges:
            available_types = set(self.data.edge_types)
            for edge_type in self.config.balanced_edge_types:
                reverse = self._reverse_edge_type(edge_type, available_types)
                if reverse is not None:
                    edge_ids[reverse] = self._matching_reverse_ids(
                        edge_type, edge_ids[edge_type], reverse
                    )

        negatives = helper._negative_edges(edge_ids, self._generator())
        node_parts = {node_type: [] for node_type in self.data.node_types}
        node_parts["cell"].append(context_cells)
        for edge_type, selected in edge_ids.items():
            if selected.numel():
                edges = self.data[edge_type].edge_index[:, selected]
                node_parts[edge_type[0]].append(edges[0])
                node_parts[edge_type[2]].append(edges[1])
        for edge_type, edges in negatives.items():
            node_parts[edge_type[0]].append(edges[0])
            node_parts[edge_type[2]].append(edges[1])
        node_ids = {
            node_type: (
                torch.unique(torch.cat(values), sorted=True)
                if values
                else torch.empty(0, dtype=torch.long)
            )
            for node_type, values in node_parts.items()
        }

        selected_path_count = len(
            complete_paths(
                self.data,
                tp_type,
                pg_type,
                edge_ids[tp_type],
                edge_ids[pg_type],
            )
        )
        forward_types = (*self.config.balanced_edge_types, tp_type, pg_type)
        forward_counts = [int(edge_ids[edge_type].numel()) for edge_type in forward_types]
        edge_balance = min(forward_counts) / max(forward_counts)
        type_coverage = None
        if self.cell_types is not None:
            type_coverage = len(set(self.cell_types[context_cells.numpy()].tolist())) / len(
                set(self.cell_types.tolist())
            )
        feature_bytes = sum(
            int(node_ids[node_type].numel())
            * int(getattr(self.data[node_type], "x", torch.empty((0, 0))).shape[1])
            * 4
            for node_type in self.data.node_types
        )
        message_edge_count = sum(int(ids.numel()) for ids in edge_ids.values())
        estimated_memory = (feature_bytes + message_edge_count * 20) / (1024**2)
        allowed_splits = set(self.config.allowed_split_codes)
        held_out_overlap = sum(
            int(
                sum(
                    int(code) not in allowed_splits
                    for code in self.data[edge_type].split_code[selected].tolist()
                )
            )
            for edge_type, selected in edge_ids.items()
            if selected.numel() and hasattr(self.data[edge_type], "split_code")
        )
        negative_collisions = 0
        for edge_type, sampled_negatives in negatives.items():
            known_positives = set(
                map(tuple, helper.all_positive_edges[edge_type].t().tolist())
            )
            negative_collisions += sum(
                tuple(edge) in known_positives for edge in sampled_negatives.t().tolist()
            )
        diagnostics = {
            "sampler": type(self).__name__,
            "seed_cell_ids": seeds.clone(),
            "seed_cell_count": int(seeds.numel()),
            "context_member_ids": members.clone(),
            "context_cell_count": int(context_cells.numel()),
            "context_mode": "metacell_mean" if self.config.use_metacell_context else "seed_mean",
            "active_gene_ids": active_genes.clone(),
            "active_peak_ids": active_peaks.clone(),
            "active_gene_count": int(active_genes.numel()),
            "active_peak_count": int(active_peaks.numel()),
            "state_gene_candidate_count": int(state_genes.numel()),
            "state_peak_candidate_count": int(state_peaks.numel()),
            "edge_counts": {edge_type: int(ids.numel()) for edge_type, ids in edge_ids.items()},
            "edge_type_balance": edge_balance,
            "selected_complete_path_count": selected_path_count,
            "full_complete_path_count": full_path_count,
            "path_retention_ratio": selected_path_count / full_path_count,
            "cell_type_coverage": type_coverage,
            "negative_edge_counts": {
                edge_type: int(edges.shape[1]) for edge_type, edges in negatives.items()
            },
            "negative_positive_collisions": negative_collisions,
            "held_out_message_overlap": held_out_overlap,
            "estimated_memory_mib": estimated_memory,
            "sampling_time_ms": (time.perf_counter() - started) * 1000,
            "path_completion_enforced": True,
            "role": "stage2_context_path",
        }
        batch = self._build_batch(node_ids, edge_ids, diagnostics)
        helper._attach_negatives(batch, negatives)
        batch.context_features = MappingProxyType(
            {
                "rna_activity": gene_activity[None, :],
                "atac_activity": peak_activity[None, :],
            }
        )
        return batch
