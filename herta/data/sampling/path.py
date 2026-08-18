"""Train-only regulatory-path sampling utilities."""

from __future__ import annotations

from collections import defaultdict
from types import MappingProxyType
from typing import Mapping

import torch

from herta.data.negative_sampling import sample_negative_targets
from herta.data.sampling.base import BaseSubgraphSampler, EdgeType, SamplerConfig, SubgraphBatch


def complete_paths(
    data,
    tp_type: EdgeType,
    pg_type: EdgeType,
    tp_ids: torch.Tensor,
    pg_ids: torch.Tensor,
) -> list[tuple[int, int, int, int, int]]:
    """Return ``(tf, peak, gene, tp_eid, pg_eid)`` complete paths."""

    children: dict[int, list[tuple[int, int]]] = defaultdict(list)
    pg_edges = data[pg_type].edge_index[:, pg_ids]
    for edge_id, (peak, gene) in zip(pg_ids.tolist(), pg_edges.t().tolist()):
        children[int(peak)].append((int(gene), int(edge_id)))
    paths: list[tuple[int, int, int, int, int]] = []
    tp_edges = data[tp_type].edge_index[:, tp_ids]
    for tp_id, (tf, peak) in zip(tp_ids.tolist(), tp_edges.t().tolist()):
        paths.extend(
            (int(tf), int(peak), gene, int(tp_id), pg_id)
            for gene, pg_id in children.get(int(peak), [])
        )
    return paths


def relation_edge_weight(data, edge_type: EdgeType, edge_id: int) -> float:
    weights = getattr(data[edge_type], "edge_weight", None)
    return float(weights[edge_id]) if weights is not None else 1.0


class RegulatoryPathSampler(BaseSubgraphSampler):
    """Sample complete train-only TF->peak->gene paths without cell context."""

    def __init__(
        self,
        data,
        config: SamplerConfig | None = None,
        *,
        all_positive_edges: Mapping[EdgeType, torch.Tensor] | None = None,
    ) -> None:
        super().__init__(data, config)
        missing = set(self.config.path_edge_types).difference(self.data.edge_types)
        if missing:
            raise ValueError(f"Regulatory relations are absent from the graph: {sorted(missing)}")
        supplied = dict(all_positive_edges or {})
        self.all_positive_edges = {
            edge_type: torch.as_tensor(
                supplied.get(edge_type, self.data[edge_type].edge_index), dtype=torch.long
            )
            for edge_type in self.config.path_edge_types
        }
        for edge_type, edges in self.all_positive_edges.items():
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError(f"all_positive_edges for {edge_type} must have shape [2, n].")

    def _select_paths(self) -> tuple[list[tuple[int, int, int, int, int]], int]:
        tp_type, pg_type = self.config.path_edge_types
        all_paths = complete_paths(
            self.data,
            tp_type,
            pg_type,
            self._eligible_edge_ids(tp_type),
            self._eligible_edge_ids(pg_type),
        )
        ranked = sorted(
            all_paths,
            key=lambda path: (
                -relation_edge_weight(self.data, tp_type, path[3])
                * relation_edge_weight(self.data, pg_type, path[4]),
                path,
            ),
        )
        return ranked[: self.config.path_budget], len(all_paths)

    def _edge_ids_for_paths(
        self, paths: list[tuple[int, int, int, int, int]]
    ) -> dict[EdgeType, torch.Tensor]:
        tp_type, pg_type = self.config.path_edge_types
        edge_ids = {
            edge_type: torch.empty(0, dtype=torch.long) for edge_type in self.data.edge_types
        }
        edge_ids[tp_type] = torch.as_tensor(sorted({path[3] for path in paths}), dtype=torch.long)
        edge_ids[pg_type] = torch.as_tensor(sorted({path[4] for path in paths}), dtype=torch.long)
        if self.config.include_reverse_edges:
            available = set(self.data.edge_types)
            for edge_type in (tp_type, pg_type):
                reverse = self._reverse_edge_type(edge_type, available)
                if reverse is not None:
                    edge_ids[reverse] = self._matching_reverse_ids(
                        edge_type, edge_ids[edge_type], reverse
                    )
        return edge_ids

    def _negative_edges(
        self,
        edge_ids: Mapping[EdgeType, torch.Tensor],
        generator: torch.Generator,
    ) -> dict[EdgeType, torch.Tensor]:
        negatives: dict[EdgeType, torch.Tensor] = {}
        for edge_type in self.config.path_edge_types:
            selected = self.data[edge_type].edge_index[:, edge_ids[edge_type]]
            if selected.shape[1] == 0:
                negatives[edge_type] = torch.empty((2, 0), dtype=torch.long)
                continue
            targets = sample_negative_targets(
                selected,
                int(self.data[edge_type[2]].num_nodes),
                self.config.negative_ratio,
                generator,
                all_positive_edge_index=self.all_positive_edges[edge_type],
            )
            sources = selected[0].repeat_interleave(self.config.negative_ratio)
            negatives[edge_type] = torch.stack([sources, targets.reshape(-1)])
        return negatives

    def _node_ids(
        self,
        edge_ids: Mapping[EdgeType, torch.Tensor],
        negatives: Mapping[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        parts = {node_type: [] for node_type in self.data.node_types}
        for edge_type, selected in edge_ids.items():
            if selected.numel():
                edges = self.data[edge_type].edge_index[:, selected]
                parts[edge_type[0]].append(edges[0])
                parts[edge_type[2]].append(edges[1])
        for edge_type, edges in negatives.items():
            if edges.numel():
                parts[edge_type[0]].append(edges[0])
                parts[edge_type[2]].append(edges[1])
        return {
            node_type: (
                torch.unique(torch.cat(values), sorted=True)
                if values
                else torch.empty(0, dtype=torch.long)
            )
            for node_type, values in parts.items()
        }

    @staticmethod
    def _attach_negatives(
        batch: SubgraphBatch,
        negatives: Mapping[EdgeType, torch.Tensor],
    ) -> None:
        local: dict[EdgeType, torch.Tensor] = {}
        global_copy: dict[EdgeType, torch.Tensor] = {}
        for edge_type, edges in negatives.items():
            global_copy[edge_type] = edges.clone()
            local[edge_type] = torch.stack(
                [
                    batch.global_to_local[edge_type[0]][edges[0]],
                    batch.global_to_local[edge_type[2]][edges[1]],
                ]
            )
        batch.global_negative_edge_index = MappingProxyType(global_copy)
        batch.negative_edge_index = MappingProxyType(local)

    def sample(self) -> SubgraphBatch:
        paths, full_path_count = self._select_paths()
        if not paths:
            raise ValueError("No split-eligible complete regulatory paths are available.")
        edge_ids = self._edge_ids_for_paths(paths)
        negatives = self._negative_edges(edge_ids, self._generator())
        node_ids = self._node_ids(edge_ids, negatives)
        selected_path_count = len(
            complete_paths(
                self.data,
                *self.config.path_edge_types,
                edge_ids[self.config.path_edge_types[0]],
                edge_ids[self.config.path_edge_types[1]],
            )
        )
        counts = {edge_type: int(ids.numel()) for edge_type, ids in edge_ids.items()}
        tp_count = counts[self.config.path_edge_types[0]]
        pg_count = counts[self.config.path_edge_types[1]]
        batch = self._build_batch(
            node_ids,
            edge_ids,
            {
                "sampler": type(self).__name__,
                "path_budget": self.config.path_budget,
                "selected_complete_path_count": selected_path_count,
                "full_complete_path_count": full_path_count,
                "path_retention_ratio": selected_path_count / full_path_count,
                "edge_counts": counts,
                "edge_type_balance": min(tp_count, pg_count) / max(tp_count, pg_count),
                "negative_edge_counts": {
                    key: int(value.shape[1]) for key, value in negatives.items()
                },
                "role": "regulatory_path_only",
            },
        )
        self._attach_negatives(batch, negatives)
        return batch
