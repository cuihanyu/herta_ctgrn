"""Full-graph and relation-balanced baseline samplers."""

from __future__ import annotations

from typing import Mapping

import torch

from herta.data.sampling.base import BaseSubgraphSampler, EdgeType, SubgraphBatch


class FullGraphSampler(BaseSubgraphSampler):
    """Return every node and every split-eligible message edge."""

    def sample(self) -> SubgraphBatch:
        node_ids = {
            node_type: torch.arange(int(self.data[node_type].num_nodes), dtype=torch.long)
            for node_type in self.data.node_types
        }
        edge_ids = {
            edge_type: self._eligible_edge_ids(edge_type)
            for edge_type in self.data.edge_types
        }
        counts = {edge_type: int(ids.numel()) for edge_type, ids in edge_ids.items()}
        return self._build_batch(
            node_ids,
            edge_ids,
            {
                "sampler": type(self).__name__,
                "seed": self.config.seed,
                "allowed_split_codes": self.config.allowed_split_codes,
                "edge_counts": counts,
                "edge_type_balance": self._edge_balance(counts, self.config.balanced_edge_types),
            },
        )

    @staticmethod
    def _edge_balance(counts: Mapping[EdgeType, int], edge_types: tuple[EdgeType, ...]) -> float:
        values = [counts.get(edge_type, 0) for edge_type in edge_types]
        return float(min(values) / max(values)) if values and max(values) else 0.0


class EdgeTypeBalancedSampler(BaseSubgraphSampler):
    """Sample equal budgets from configured forward edge relations.

    The first baseline is intended for Stage 1, where the defaults balance
    ``cell->gene`` and ``cell->peak`` edges.  Reverse relations are selected by
    endpoint identity, never sampled independently.
    """

    def _sample_relation(
        self,
        edge_type: EdgeType,
        eligible: torch.Tensor,
        budget: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if budget >= eligible.numel():
            return eligible
        edge_index = self.data[edge_type].edge_index[:, eligible]
        if edge_type[0] != "cell":
            order = torch.randperm(eligible.numel(), generator=generator)[:budget]
            return eligible[order]

        # Cover as many source cells as the budget allows before filling the
        # remaining positions uniformly from as-yet-unselected edges.
        source_to_edges: dict[int, list[int]] = {}
        for local_id, source in enumerate(edge_index[0].tolist()):
            source_to_edges.setdefault(int(source), []).append(local_id)
        source_ids = torch.as_tensor(sorted(source_to_edges), dtype=torch.long)
        source_order = source_ids[
            torch.randperm(source_ids.numel(), generator=generator)
        ]
        chosen_local: list[int] = []
        for source in source_order[:budget].tolist():
            candidates = source_to_edges[int(source)]
            offset = int(torch.randint(len(candidates), (1,), generator=generator))
            chosen_local.append(candidates[offset])
        remaining = budget - len(chosen_local)
        if remaining:
            selected_mask = torch.ones(eligible.numel(), dtype=torch.bool)
            selected_mask[torch.as_tensor(chosen_local, dtype=torch.long)] = False
            pool = torch.nonzero(selected_mask, as_tuple=False).flatten()
            fill = pool[torch.randperm(pool.numel(), generator=generator)[:remaining]]
            chosen_local.extend(fill.tolist())
        return eligible[torch.as_tensor(chosen_local, dtype=torch.long)]

    def sample(self) -> SubgraphBatch:
        missing = set(self.config.balanced_edge_types).difference(self.data.edge_types)
        if missing:
            raise ValueError(f"Balanced relations are absent from the graph: {sorted(missing)}")
        eligible = {
            edge_type: self._eligible_edge_ids(edge_type)
            for edge_type in self.config.balanced_edge_types
        }
        empty = [edge_type for edge_type, ids in eligible.items() if ids.numel() == 0]
        if empty:
            raise ValueError(f"Balanced relations have no split-eligible edges: {empty}")
        budget = min(ids.numel() for ids in eligible.values())
        if self.config.edges_per_type is not None:
            budget = min(budget, self.config.edges_per_type)
        generator = self._generator()
        edge_ids: dict[EdgeType, torch.Tensor] = {
            edge_type: torch.empty(0, dtype=torch.long) for edge_type in self.data.edge_types
        }
        for edge_type in self.config.balanced_edge_types:
            edge_ids[edge_type] = self._sample_relation(
                edge_type, eligible[edge_type], int(budget), generator
            )

        available = set(self.data.edge_types)
        if self.config.include_reverse_edges:
            for edge_type in self.config.balanced_edge_types:
                reverse = self._reverse_edge_type(edge_type, available)
                if reverse is not None:
                    edge_ids[reverse] = self._matching_reverse_ids(
                        edge_type, edge_ids[edge_type], reverse
                    )

        node_parts: dict[str, list[torch.Tensor]] = {
            node_type: [] for node_type in self.data.node_types
        }
        if self.config.preserve_all_cells and "cell" in node_parts:
            node_parts["cell"].append(
                torch.arange(int(self.data["cell"].num_nodes), dtype=torch.long)
            )
        for edge_type, selected in edge_ids.items():
            if selected.numel() == 0:
                continue
            global_edges = self.data[edge_type].edge_index[:, selected]
            node_parts[edge_type[0]].append(global_edges[0])
            node_parts[edge_type[2]].append(global_edges[1])
        node_ids = {
            node_type: (
                torch.unique(torch.cat(parts), sorted=True)
                if parts
                else torch.empty(0, dtype=torch.long)
            )
            for node_type, parts in node_parts.items()
        }
        counts = {edge_type: int(ids.numel()) for edge_type, ids in edge_ids.items()}
        forward_counts = [counts[edge_type] for edge_type in self.config.balanced_edge_types]
        balance = float(min(forward_counts) / max(forward_counts)) if max(forward_counts) else 0.0
        return self._build_batch(
            node_ids,
            edge_ids,
            {
                "sampler": type(self).__name__,
                "seed": self.config.seed,
                "allowed_split_codes": self.config.allowed_split_codes,
                "balanced_edge_types": self.config.balanced_edge_types,
                "edges_per_type": int(budget),
                "edge_counts": counts,
                "edge_type_balance": balance,
            },
        )

