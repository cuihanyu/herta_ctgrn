"""Frequency-based random walk with restart for HERTA heterogeneous graphs."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping

import torch

from herta.data.sampling.base import BaseSubgraphSampler, EdgeType, SamplerConfig, SubgraphBatch


TypedNode = tuple[str, int]
AdjacencyEntry = tuple[str, int, EdgeType, int]


class FRWRSampler(BaseSubgraphSampler):
    """Sample typed local context by visit frequency under random restart.

    Walks follow the directed, split-eligible relations present in the input
    graph.  The sampler records complete visit-count tensors for every node
    type and retains the most visited nodes independently per type.
    """

    def __init__(
        self,
        data,
        config: SamplerConfig | None = None,
        *,
        seed_node_ids: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(data, config)
        self._default_seed_node_ids = self._validate_seeds(seed_node_ids, allow_none=True)
        self._eligible = {
            edge_type: self._eligible_edge_ids(edge_type)
            for edge_type in self.data.edge_types
        }
        self._adjacency, self._out_degree = self._build_adjacency()
        if self.config.path_completion:
            missing = set(self.config.path_edge_types).difference(self.data.edge_types)
            if missing:
                raise ValueError(f"Path-completion relations are absent from the graph: {sorted(missing)}")

    def _validate_seeds(
        self,
        values: Mapping[str, object] | None,
        *,
        allow_none: bool,
    ) -> dict[str, torch.Tensor] | None:
        if values is None:
            return None if allow_none else {}
        if not values:
            raise ValueError("seed_node_ids must not be empty.")
        result: dict[str, torch.Tensor] = {}
        for node_type, raw_ids in values.items():
            if node_type not in self.data.node_types:
                raise KeyError(f"Unknown seed node type: {node_type}")
            ids = torch.as_tensor(raw_ids, dtype=torch.long).flatten()
            if ids.numel() == 0:
                raise ValueError(f"Seed IDs for {node_type} must not be empty.")
            if torch.unique(ids).numel() != ids.numel():
                raise ValueError(f"Seed IDs for {node_type} contain duplicates.")
            total = int(self.data[node_type].num_nodes)
            if bool((ids < 0).any()) or bool((ids >= total).any()):
                raise IndexError(f"Seed ID for {node_type} is out of range.")
            result[node_type] = ids
        return result

    def _resolve_seeds(self, values: Mapping[str, object] | None) -> dict[str, torch.Tensor]:
        resolved = self._validate_seeds(values, allow_none=True)
        if resolved is not None:
            return resolved
        if self._default_seed_node_ids is not None:
            return {key: value.clone() for key, value in self._default_seed_node_ids.items()}
        raise ValueError("FRWRSampler requires seed_node_ids at construction or sampling time.")

    def _build_adjacency(
        self,
    ) -> tuple[dict[TypedNode, list[AdjacencyEntry]], dict[str, torch.Tensor]]:
        adjacency: dict[TypedNode, list[AdjacencyEntry]] = defaultdict(list)
        degree = {
            node_type: torch.zeros(int(self.data[node_type].num_nodes), dtype=torch.float32)
            for node_type in self.data.node_types
        }
        for edge_type, eligible in self._eligible.items():
            source_type, _, target_type = edge_type
            edges = self.data[edge_type].edge_index[:, eligible]
            for global_edge_id, source, target in zip(
                eligible.tolist(), edges[0].tolist(), edges[1].tolist()
            ):
                adjacency[(source_type, int(source))].append(
                    (target_type, int(target), edge_type, int(global_edge_id))
                )
                degree[source_type][int(source)] += 1.0
        return dict(adjacency), degree

    def _walk(
        self,
        seeds: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        visits = {
            node_type: torch.zeros(int(self.data[node_type].num_nodes), dtype=torch.long)
            for node_type in self.data.node_types
        }
        generator = self._generator()
        typed_seeds = [
            (node_type, int(node_id))
            for node_type, ids in seeds.items()
            for node_id in ids.tolist()
        ]
        for seed in typed_seeds:
            for _ in range(self.config.num_walks):
                current = seed
                visits[current[0]][current[1]] += 1
                for _ in range(self.config.walk_length):
                    if float(torch.rand((), generator=generator)) < self.config.restart_prob:
                        current = seed
                    else:
                        neighbors = self._adjacency.get(current, [])
                        if neighbors:
                            index = int(torch.randint(len(neighbors), (1,), generator=generator))
                            target_type, target_id, _, _ = neighbors[index]
                            current = (target_type, target_id)
                        else:
                            current = seed
                    visits[current[0]][current[1]] += 1
        return visits

    def _top_visited_nodes(
        self,
        visits: Mapping[str, torch.Tensor],
        seeds: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        selected: dict[str, torch.Tensor] = {}
        for node_type in self.data.node_types:
            counts = visits[node_type]
            visited_ids = torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()
            visited_ids.sort(key=lambda node_id: (-int(counts[node_id]), node_id))
            cap = self.config.top_nodes_per_type.get(node_type, len(visited_ids))
            retained = visited_ids[:cap]
            retained.extend(seeds.get(node_type, torch.empty(0, dtype=torch.long)).tolist())
            selected[node_type] = torch.as_tensor(sorted(set(retained)), dtype=torch.long)
        return selected

    def _induced_edge_ids(self, node_ids: Mapping[str, torch.Tensor]) -> dict[EdgeType, torch.Tensor]:
        selected_sets = {
            node_type: set(ids.tolist()) for node_type, ids in node_ids.items()
        }
        result: dict[EdgeType, torch.Tensor] = {}
        for edge_type, eligible in self._eligible.items():
            source_type, _, target_type = edge_type
            edges = self.data[edge_type].edge_index[:, eligible]
            keep = [
                local_id
                for local_id, (source, target) in enumerate(edges.t().tolist())
                if int(source) in selected_sets[source_type]
                and int(target) in selected_sets[target_type]
            ]
            result[edge_type] = (
                eligible[torch.as_tensor(keep, dtype=torch.long)]
                if keep
                else torch.empty(0, dtype=torch.long)
            )
        return result

    def _candidate_edges_by_peak(
        self,
        edge_type: EdgeType,
        peak_is_target: bool,
    ) -> dict[int, list[int]]:
        candidates: dict[int, list[int]] = defaultdict(list)
        eligible = self._eligible[edge_type]
        edges = self.data[edge_type].edge_index[:, eligible]
        peak_row = 1 if peak_is_target else 0
        for edge_id, peak_id in zip(eligible.tolist(), edges[peak_row].tolist()):
            candidates[int(peak_id)].append(int(edge_id))
        for values in candidates.values():
            values.sort()
        return dict(candidates)

    def _complete_paths(
        self,
        node_ids: dict[str, torch.Tensor],
        edge_ids: dict[EdgeType, torch.Tensor],
    ) -> dict[str, int]:
        if not self.config.path_completion:
            return {
                "path_completion_added_edges": 0,
                "path_completion_added_nodes": 0,
                "path_completion_completed_peaks": 0,
                "path_completion_uncompletable_peaks": 0,
            }
        tp_type, pg_type = self.config.path_edge_types
        peak_type = tp_type[2]
        candidate_tp = self._candidate_edges_by_peak(tp_type, peak_is_target=True)
        candidate_pg = self._candidate_edges_by_peak(pg_type, peak_is_target=False)
        selected_peaks = node_ids[peak_type].tolist()
        selected_tp = set(edge_ids[tp_type].tolist())
        selected_pg = set(edge_ids[pg_type].tolist())
        before_edges = len(selected_tp) + len(selected_pg)
        before_nodes = sum(ids.numel() for ids in node_ids.values())
        completed = 0
        uncompletable = 0
        for peak in selected_peaks:
            parents = candidate_tp.get(int(peak), [])
            children = candidate_pg.get(int(peak), [])
            if not parents or not children:
                uncompletable += 1
                continue
            selected_parents = [edge_id for edge_id in parents if edge_id in selected_tp]
            selected_children = [edge_id for edge_id in children if edge_id in selected_pg]
            if not selected_parents:
                selected_tp.update(parents[: self.config.path_completion_fanout])
            if not selected_children:
                selected_pg.update(children[: self.config.path_completion_fanout])
            completed += 1
        edge_ids[tp_type] = torch.as_tensor(sorted(selected_tp), dtype=torch.long)
        edge_ids[pg_type] = torch.as_tensor(sorted(selected_pg), dtype=torch.long)

        node_sets = {node_type: set(ids.tolist()) for node_type, ids in node_ids.items()}
        for edge_type in (tp_type, pg_type):
            selected = edge_ids[edge_type]
            edges = self.data[edge_type].edge_index[:, selected]
            node_sets[edge_type[0]].update(map(int, edges[0].tolist()))
            node_sets[edge_type[2]].update(map(int, edges[1].tolist()))
        for node_type, values in node_sets.items():
            node_ids[node_type] = torch.as_tensor(sorted(values), dtype=torch.long)

        if self.config.include_reverse_edges:
            available = set(self.data.edge_types)
            for edge_type in (tp_type, pg_type):
                reverse = self._reverse_edge_type(edge_type, available)
                if reverse is not None:
                    edge_ids[reverse] = self._matching_reverse_ids(
                        edge_type, edge_ids[edge_type], reverse
                    )
        after_edges = edge_ids[tp_type].numel() + edge_ids[pg_type].numel()
        after_nodes = sum(ids.numel() for ids in node_ids.values())
        return {
            "path_completion_added_edges": int(after_edges - before_edges),
            "path_completion_added_nodes": int(after_nodes - before_nodes),
            "path_completion_completed_peaks": completed,
            "path_completion_uncompletable_peaks": uncompletable,
        }

    def _complete_path_count(self, edge_ids: Mapping[EdgeType, torch.Tensor]) -> int:
        if any(edge_type not in edge_ids for edge_type in self.config.path_edge_types):
            return 0
        tp_type, pg_type = self.config.path_edge_types
        parents: dict[int, set[int]] = defaultdict(set)
        children: dict[int, set[int]] = defaultdict(set)
        tp_edges = self.data[tp_type].edge_index[:, edge_ids[tp_type]]
        pg_edges = self.data[pg_type].edge_index[:, edge_ids[pg_type]]
        for tf_id, peak_id in tp_edges.t().tolist():
            parents[int(peak_id)].add(int(tf_id))
        for peak_id, gene_id in pg_edges.t().tolist():
            children[int(peak_id)].add(int(gene_id))
        return sum(len(parents[peak]) * len(children[peak]) for peak in parents.keys() & children.keys())

    @staticmethod
    def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
        if x.numel() < 2:
            return 0.0
        x = x.float() - x.float().mean()
        y = y.float() - y.float().mean()
        denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
        return float(torch.dot(x, y) / denominator) if float(denominator) > 0 else 0.0

    def _degree_bias(
        self,
        visits: Mapping[str, torch.Tensor],
        node_ids: Mapping[str, torch.Tensor],
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for node_type in self.data.node_types:
            degree = self._out_degree[node_type]
            frequency = visits[node_type].float()
            selected = node_ids[node_type]
            n_hubs = max(1, math.ceil(degree.numel() * 0.1))
            hub_ids = torch.argsort(degree, descending=True)[:n_hubs]
            total_visits = float(frequency.sum())
            selected_mean = float(degree[selected].mean()) if selected.numel() else 0.0
            all_mean = float(degree.mean()) if degree.numel() else 0.0
            result[node_type] = {
                "degree_visit_pearson": self._pearson(degree, frequency),
                "top_degree_10pct_visit_share": (
                    float(frequency[hub_ids].sum()) / total_visits if total_visits else 0.0
                ),
                "selected_mean_out_degree": selected_mean,
                "all_mean_out_degree": all_mean,
                "selected_to_all_degree_ratio": selected_mean / all_mean if all_mean else 0.0,
            }
        return result

    def sample(
        self,
        seed_node_ids: Mapping[str, object] | None = None,
    ) -> SubgraphBatch:
        seeds = self._resolve_seeds(seed_node_ids)
        visits = self._walk(seeds)
        node_ids = self._top_visited_nodes(visits, seeds)
        edge_ids = self._induced_edge_ids(node_ids)
        path_diagnostics = self._complete_paths(node_ids, edge_ids)
        counts = {edge_type: int(ids.numel()) for edge_type, ids in edge_ids.items()}
        diagnostics = {
            "sampler": type(self).__name__,
            "seed": self.config.seed,
            "restart_prob": self.config.restart_prob,
            "walk_length": self.config.walk_length,
            "num_walks": self.config.num_walks,
            "seed_node_ids": {key: value.clone() for key, value in seeds.items()},
            "visit_frequency": {key: value.clone() for key, value in visits.items()},
            "unique_visited_nodes": {
                key: int((value > 0).sum()) for key, value in visits.items()
            },
            "selected_node_counts": {key: int(value.numel()) for key, value in node_ids.items()},
            "edge_counts": counts,
            "path_completion": self.config.path_completion,
            "complete_path_count": self._complete_path_count(edge_ids),
            "degree_bias": self._degree_bias(visits, node_ids),
            **path_diagnostics,
        }
        return self._build_batch(node_ids, edge_ids, diagnostics)

