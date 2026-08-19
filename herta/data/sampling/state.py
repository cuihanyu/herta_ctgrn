"""Bounded two-hop cell–feature subgraphs for Stage-1 HGT training."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from herta.data.sampling.base import (
    BaseSubgraphSampler,
    EdgeType,
    SamplerConfig,
    SubgraphBatch,
)


CG: EdgeType = ("cell", "expresses", "gene")
CP: EdgeType = ("cell", "accessible", "peak")
STATE_FORWARD_RELATIONS: tuple[EdgeType, EdgeType] = (CG, CP)


@dataclass(frozen=True)
class CellFeatureSubgraphConfig:
    """Hard message-passing budgets independent of the full graph size."""

    message_edges_per_cell_cg: int = 64
    message_edges_per_cell_cp: int = 128
    context_cells_per_gene: int = 4
    context_cells_per_peak: int = 4
    max_context_cells: int = 512

    def __post_init__(self) -> None:
        values = {
            "message_edges_per_cell_cg": self.message_edges_per_cell_cg,
            "message_edges_per_cell_cp": self.message_edges_per_cell_cp,
            "context_cells_per_gene": self.context_cells_per_gene,
            "context_cells_per_peak": self.context_cells_per_peak,
            "max_context_cells": self.max_context_cells,
        }
        if any(not isinstance(value, int) or value < 1 for value in values.values()):
            raise ValueError("Every cell-feature subgraph budget must be a positive integer.")

    @property
    def message_fanouts(self) -> Mapping[EdgeType, int]:
        return MappingProxyType(
            {
                CG: self.message_edges_per_cell_cg,
                CP: self.message_edges_per_cell_cp,
            }
        )

    @property
    def context_fanouts(self) -> Mapping[EdgeType, int]:
        return MappingProxyType(
            {
                CG: self.context_cells_per_gene,
                CP: self.context_cells_per_peak,
            }
        )


class CellFeatureSubgraphSampler(BaseSubgraphSampler):
    """Sample `context cell -> feature -> seed cell` Stage-1 subgraphs.

    Seed cells are supplied by the trainer's cell-balanced sampler. Context
    cells provide the second-hop messages needed by a two-layer HGT, but never
    become supervision sources unless they are also seed cells.
    """

    def __init__(
        self,
        data,
        config: SamplerConfig | None = None,
        *,
        budgets: CellFeatureSubgraphConfig | None = None,
    ) -> None:
        super().__init__(data, config)
        self.budgets = budgets or CellFeatureSubgraphConfig()
        if tuple(self.config.balanced_edge_types) != STATE_FORWARD_RELATIONS:
            raise ValueError(
                "CellFeatureSubgraphSampler requires canonical CG and CP relations."
            )
        self._reverse_types = self._validate_reverse_parity()
        self._source_rows = {
            edge_type: self._group_rows(edge_type, by_source=True)
            for edge_type in STATE_FORWARD_RELATIONS
        }
        self._target_rows = {
            edge_type: self._group_rows(edge_type, by_source=False)
            for edge_type in STATE_FORWARD_RELATIONS
        }

    def _validate_reverse_parity(self) -> Mapping[EdgeType, EdgeType]:
        available = set(self.data.edge_types)
        result: dict[EdgeType, EdgeType] = {}
        for edge_type in STATE_FORWARD_RELATIONS:
            if edge_type not in available:
                raise ValueError(f"Missing Stage-1 relation: {edge_type}")
            reverse = self._reverse_edge_type(edge_type, available)
            if reverse is None:
                raise ValueError(f"Missing reverse relation for {edge_type}.")
            if not torch.equal(
                self.data[reverse].edge_index,
                self.data[edge_type].edge_index.flip(0),
            ):
                raise ValueError(f"Reverse relation {reverse} is not edge-ID aligned.")
            result[edge_type] = reverse
        return MappingProxyType(result)

    def _group_rows(
        self, edge_type: EdgeType, *, by_source: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = self.data[edge_type].edge_index
        axis = 0 if by_source else 1
        node_type = edge_type[0] if by_source else edge_type[2]
        values = edge_index[axis]
        order = torch.argsort(values, stable=True)
        counts = torch.bincount(
            values[order], minlength=int(self.data[node_type].num_nodes)
        )
        row_ptr = torch.zeros(len(counts) + 1, dtype=torch.long)
        row_ptr[1:] = torch.cumsum(counts, dim=0)
        return order, row_ptr

    @staticmethod
    def _row(
        grouped: tuple[torch.Tensor, torch.Tensor], node_id: int
    ) -> torch.Tensor:
        edge_ids, row_ptr = grouped
        return edge_ids[row_ptr[node_id] : row_ptr[node_id + 1]]

    @staticmethod
    def _bounded_union_sample(
        pool: torch.Tensor,
        required: torch.Tensor,
        budget: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        required = torch.unique(required, sorted=True)
        if required.numel() > budget:
            raise ValueError(
                "Supervision positives exceed the configured message fanout."
            )
        if pool.numel() <= budget:
            if required.numel() and not bool(torch.isin(required, pool).all()):
                raise ValueError("Required supervision edge is absent from the source row.")
            return pool
        available = pool[~torch.isin(pool, required)]
        remaining = budget - required.numel()
        if remaining:
            selected = available[
                torch.randperm(available.numel(), generator=generator)[:remaining]
            ]
            return torch.unique(torch.cat([required, selected]), sorted=True)
        return required

    def _first_hop(
        self,
        seed_cell_ids: torch.Tensor,
        required_edge_ids: Mapping[EdgeType, torch.Tensor],
        generator: torch.Generator,
    ) -> dict[EdgeType, torch.Tensor]:
        selected: dict[EdgeType, torch.Tensor] = {}
        for edge_type in STATE_FORWARD_RELATIONS:
            required = torch.as_tensor(
                required_edge_ids.get(edge_type, torch.empty(0)), dtype=torch.long
            )
            required_sources = self.data[edge_type].edge_index[0, required]
            parts: list[torch.Tensor] = []
            for cell_id in seed_cell_ids.tolist():
                pool = self._row(self._source_rows[edge_type], int(cell_id))
                needed = required[required_sources == cell_id]
                parts.append(
                    self._bounded_union_sample(
                        pool,
                        needed,
                        int(self.budgets.message_fanouts[edge_type]),
                        generator,
                    )
                )
            selected[edge_type] = torch.unique(torch.cat(parts), sorted=True)
        return selected

    def _context_hop(
        self,
        seed_cell_ids: torch.Tensor,
        first_hop: Mapping[EdgeType, torch.Tensor],
        generator: torch.Generator,
    ) -> tuple[dict[EdgeType, torch.Tensor], torch.Tensor]:
        candidates: dict[EdgeType, torch.Tensor] = {}
        all_context_sources: list[torch.Tensor] = []
        for edge_type in STATE_FORWARD_RELATIONS:
            targets = torch.unique(
                self.data[edge_type].edge_index[1, first_hop[edge_type]], sorted=True
            )
            parts: list[torch.Tensor] = []
            fanout = int(self.budgets.context_fanouts[edge_type])
            for target_id in targets.tolist():
                pool = self._row(self._target_rows[edge_type], int(target_id))
                if pool.numel() > fanout:
                    pool = pool[
                        torch.randperm(pool.numel(), generator=generator)[:fanout]
                    ]
                parts.append(pool)
            candidate = (
                torch.unique(torch.cat(parts), sorted=True)
                if parts
                else torch.empty(0, dtype=torch.long)
            )
            candidates[edge_type] = candidate
            all_context_sources.append(self.data[edge_type].edge_index[0, candidate])

        context_cells = torch.unique(torch.cat(all_context_sources), sorted=True)
        context_cells = context_cells[~torch.isin(context_cells, seed_cell_ids)]
        if context_cells.numel() > self.budgets.max_context_cells:
            context_cells = context_cells[
                torch.randperm(context_cells.numel(), generator=generator)[
                    : self.budgets.max_context_cells
                ]
            ]
            context_cells = torch.sort(context_cells).values

        final: dict[EdgeType, torch.Tensor] = {}
        allowed_sources = torch.cat([seed_cell_ids, context_cells])
        for edge_type in STATE_FORWARD_RELATIONS:
            candidate = candidates[edge_type]
            sources = self.data[edge_type].edge_index[0, candidate]
            retained = candidate[torch.isin(sources, allowed_sources)]
            final[edge_type] = torch.unique(
                torch.cat([first_hop[edge_type], retained]), sorted=True
            )
        return final, context_cells

    def sample(
        self,
        seed_cell_ids: torch.Tensor,
        *,
        required_edge_ids: Mapping[EdgeType, torch.Tensor],
        extra_node_ids: Mapping[str, torch.Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> SubgraphBatch:
        seeds = torch.unique(torch.as_tensor(seed_cell_ids, dtype=torch.long), sorted=True)
        if seeds.numel() == 0:
            raise ValueError("seed_cell_ids must not be empty.")
        n_cells = int(self.data["cell"].num_nodes)
        if bool((seeds < 0).any()) or bool((seeds >= n_cells).any()):
            raise IndexError("seed_cell_ids contains an out-of-range cell ID.")
        resolved_generator = generator or self._generator()
        first_hop = self._first_hop(seeds, required_edge_ids, resolved_generator)
        forward_ids, context_cells = self._context_hop(
            seeds, first_hop, resolved_generator
        )

        edge_ids: dict[EdgeType, torch.Tensor] = {
            edge_type: torch.empty(0, dtype=torch.long)
            for edge_type in self.data.edge_types
        }
        for edge_type, selected in forward_ids.items():
            edge_ids[edge_type] = selected
            edge_ids[self._reverse_types[edge_type]] = selected.clone()

        node_parts: dict[str, list[torch.Tensor]] = {
            node_type: [] for node_type in self.data.node_types
        }
        node_parts["cell"].extend([seeds, context_cells])
        for edge_type, selected in forward_ids.items():
            edges = self.data[edge_type].edge_index[:, selected]
            node_parts[edge_type[0]].append(edges[0])
            node_parts[edge_type[2]].append(edges[1])
        for node_type, ids in (extra_node_ids or {}).items():
            if node_type not in node_parts:
                raise KeyError(f"Unknown extra node type: {node_type}")
            node_parts[node_type].append(torch.as_tensor(ids, dtype=torch.long).flatten())
        node_ids = {
            node_type: torch.unique(torch.cat(parts), sorted=True)
            if parts
            else torch.empty(0, dtype=torch.long)
            for node_type, parts in node_parts.items()
        }
        batch = self._build_batch(
            node_ids,
            edge_ids,
            {
                "sampler": type(self).__name__,
                "seed": self.config.seed,
                "seed_cell_ids": seeds.clone(),
                "seed_cell_count": int(seeds.numel()),
                "context_cell_ids": context_cells.clone(),
                "context_cell_count": int(context_cells.numel()),
                "graph_batch_mode": "cell_feature_subgraph",
                "subgraph_reverse_parity": True,
            },
        )
        for edge_type, required in required_edge_ids.items():
            if not bool(
                torch.isin(torch.unique(required), batch.global_edge_ids[edge_type]).all()
            ):
                raise RuntimeError("A supervision positive is absent from the message graph.")
        return batch
