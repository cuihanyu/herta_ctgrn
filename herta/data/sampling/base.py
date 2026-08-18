"""Common data contracts for HERTA message-passing subgraph samplers."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, TypeAlias

import torch
from torch_geometric.data import HeteroData


EdgeType: TypeAlias = tuple[str, str, str]


DEFAULT_BALANCED_EDGE_TYPES: tuple[EdgeType, ...] = (
    ("cell", "expresses", "gene"),
    ("cell", "accessible", "peak"),
)


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration shared by the first HERTA subgraph-sampling baselines.

    ``allowed_split_codes`` applies to message edges carrying a ``split_code``
    attribute.  HERTA codes 0/1 denote message/train edges, while 2/3/4 denote
    validation/test/query edges and are excluded by default.
    """

    seed: int = 1
    allowed_split_codes: tuple[int, ...] = (0, 1)
    balanced_edge_types: tuple[EdgeType, ...] = DEFAULT_BALANCED_EDGE_TYPES
    edges_per_type: int | None = None
    preserve_all_cells: bool = True
    include_reverse_edges: bool = True
    n_neighbors: int = 20
    candidate_neighbors: int = 100
    neighbor_temperature: float = 0.2
    restart_prob: float = 0.15
    walk_length: int = 10
    num_walks: int = 20
    top_nodes_per_type: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({"cell": 20, "gene": 64, "peak": 128})
    )
    path_completion: bool = False
    path_completion_fanout: int = 1
    path_edge_types: tuple[EdgeType, EdgeType] = (
        ("gene", "binds", "peak"),
        ("peak", "regulates", "gene"),
    )
    active_genes: int = 64
    active_peaks: int = 128
    path_budget: int = 256
    negative_ratio: int = 1
    use_metacell_context: bool = True
    seed_cell_count: int = 64

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not self.allowed_split_codes:
            raise ValueError("allowed_split_codes must not be empty.")
        if len(set(self.allowed_split_codes)) != len(self.allowed_split_codes):
            raise ValueError("allowed_split_codes contains duplicates.")
        if self.edges_per_type is not None and self.edges_per_type < 1:
            raise ValueError("edges_per_type must be positive when provided.")
        if not self.balanced_edge_types:
            raise ValueError("balanced_edge_types must not be empty.")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        if self.candidate_neighbors < self.n_neighbors:
            raise ValueError("candidate_neighbors must be at least n_neighbors.")
        if self.neighbor_temperature <= 0:
            raise ValueError("neighbor_temperature must be positive.")
        if not 0.0 <= self.restart_prob <= 1.0:
            raise ValueError("restart_prob must be in [0, 1].")
        if self.walk_length < 1 or self.num_walks < 1:
            raise ValueError("walk_length and num_walks must be positive.")
        resolved_top_k = dict(self.top_nodes_per_type)
        if any(not isinstance(key, str) or not key for key in resolved_top_k):
            raise ValueError("top_nodes_per_type keys must be non-empty node type names.")
        if any(not isinstance(value, int) or value < 1 for value in resolved_top_k.values()):
            raise ValueError("top_nodes_per_type values must be positive integers.")
        object.__setattr__(self, "top_nodes_per_type", MappingProxyType(resolved_top_k))
        if self.path_completion_fanout < 1:
            raise ValueError("path_completion_fanout must be positive.")
        if len(self.path_edge_types) != 2:
            raise ValueError("path_edge_types must contain TF-peak and peak-gene edge types.")
        tp_type, pg_type = self.path_edge_types
        if tp_type[2] != pg_type[0]:
            raise ValueError("path_edge_types must share the same mediator node type.")
        if self.active_genes < 1 or self.active_peaks < 1:
            raise ValueError("active_genes and active_peaks must be positive.")
        if self.path_budget < 1:
            raise ValueError("path_budget must be positive.")
        if self.negative_ratio < 1:
            raise ValueError("negative_ratio must be positive.")
        if self.seed_cell_count < 1:
            raise ValueError("seed_cell_count must be positive.")
        if len(set(self.balanced_edge_types)) != len(self.balanced_edge_types):
            raise ValueError("balanced_edge_types contains duplicates.")
        for edge_type in self.balanced_edge_types:
            if len(edge_type) != 3 or not all(isinstance(value, str) and value for value in edge_type):
                raise ValueError("Each balanced edge type must be a non-empty (src, relation, dst) tuple.")


@dataclass
class SubgraphBatch:
    """One sampled HERTA graph with auditable local/global identifiers."""

    data: HeteroData
    global_node_ids: Mapping[str, torch.Tensor]
    global_edge_ids: Mapping[EdgeType, torch.Tensor]
    global_to_local: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    context_features: Mapping[str, torch.Tensor] = field(default_factory=dict)
    negative_edge_index: Mapping[EdgeType, torch.Tensor] = field(default_factory=dict)
    global_negative_edge_index: Mapping[EdgeType, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_types = set(self.data.node_types)
        if set(self.global_node_ids) != node_types or set(self.global_to_local) != node_types:
            raise ValueError("Node mapping keys must exactly match sampled node types.")
        if set(self.global_edge_ids) != set(self.data.edge_types):
            raise ValueError("Edge mapping keys must exactly match sampled edge types.")
        for node_type in self.data.node_types:
            local_to_global = self.global_node_ids[node_type]
            reverse = self.global_to_local[node_type]
            if local_to_global.dtype != torch.long or reverse.dtype != torch.long:
                raise TypeError("Node ID mappings must use torch.long.")
            if local_to_global.ndim != 1 or reverse.ndim != 1:
                raise ValueError("Node ID mappings must be one-dimensional.")
            if local_to_global.numel() != int(self.data[node_type].num_nodes):
                raise ValueError(f"Incorrect local/global node mapping length for {node_type}.")
            expected = torch.arange(local_to_global.numel(), dtype=torch.long)
            if local_to_global.numel() and not torch.equal(reverse[local_to_global], expected):
                raise ValueError(f"Node mapping is not invertible for {node_type}.")
        for edge_type in self.data.edge_types:
            edge_ids = self.global_edge_ids[edge_type]
            if edge_ids.dtype != torch.long or edge_ids.ndim != 1:
                raise TypeError("Global edge IDs must be one-dimensional torch.long tensors.")
            if edge_ids.numel() != self.data[edge_type].edge_index.shape[1]:
                raise ValueError(f"Incorrect global edge mapping length for {edge_type}.")

    def local_ids(self, node_type: str, global_ids: torch.Tensor) -> torch.Tensor:
        """Map global node IDs to local IDs, returning ``-1`` when absent."""

        if node_type not in self.global_to_local:
            raise KeyError(f"Unknown node type: {node_type}")
        ids = torch.as_tensor(global_ids, dtype=torch.long)
        mapping = self.global_to_local[node_type]
        if ids.numel() and (bool((ids < 0).any()) or bool((ids >= mapping.numel()).any())):
            raise IndexError(f"Global {node_type} ID is out of range.")
        return mapping[ids]


class BaseSubgraphSampler(ABC):
    """Base class for deterministic, split-safe HERTA subgraph samplers."""

    def __init__(self, data: HeteroData, config: SamplerConfig | None = None) -> None:
        self.data = data
        self.config = config or SamplerConfig()
        self._validate_graph()

    def _validate_graph(self) -> None:
        if not self.data.node_types:
            raise ValueError("Cannot sample an empty heterogeneous graph.")
        for node_type in self.data.node_types:
            if self.data[node_type].num_nodes is None:
                raise ValueError(f"Node type {node_type} does not define num_nodes.")
        for edge_type in self.data.edge_types:
            edge_index = getattr(self.data[edge_type], "edge_index", None)
            if edge_index is None or edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"Edge type {edge_type} requires edge_index with shape [2, n_edges].")

    def _generator(self) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(self.config.seed)

    def _eligible_edge_ids(self, edge_type: EdgeType) -> torch.Tensor:
        store = self.data[edge_type]
        n_edges = store.edge_index.shape[1]
        split_code = getattr(store, "split_code", None)
        if split_code is None:
            return torch.arange(n_edges, dtype=torch.long)
        if split_code.ndim != 1 or split_code.numel() != n_edges:
            raise ValueError(f"split_code for {edge_type} must align with its edges.")
        allowed = torch.as_tensor(self.config.allowed_split_codes, dtype=split_code.dtype)
        return torch.nonzero(torch.isin(split_code.cpu(), allowed), as_tuple=False).flatten()

    @staticmethod
    def _reverse_edge_type(edge_type: EdgeType, available: set[EdgeType]) -> EdgeType | None:
        source, relation, target = edge_type
        expected_source, expected_target = target, source
        candidates = [
            candidate
            for candidate in available
            if candidate[0] == expected_source
            and candidate[2] == expected_target
            and candidate[1] in {f"rev_{relation}", relation.removeprefix("rev_")}
        ]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous reverse relations for {edge_type}: {candidates}")
        return candidates[0] if candidates else None

    def _matching_reverse_ids(
        self,
        forward_type: EdgeType,
        forward_ids: torch.Tensor,
        reverse_type: EdgeType,
    ) -> torch.Tensor:
        """Return eligible reverse edges matching selected forward endpoint pairs."""

        forward = self.data[forward_type].edge_index[:, forward_ids]
        wanted = {(int(dst), int(src)) for src, dst in forward.t().tolist()}
        eligible = self._eligible_edge_ids(reverse_type)
        reverse = self.data[reverse_type].edge_index[:, eligible]
        keep = [index for index, pair in enumerate(reverse.t().tolist()) if tuple(pair) in wanted]
        return eligible[torch.as_tensor(keep, dtype=torch.long)] if keep else torch.empty(0, dtype=torch.long)

    @staticmethod
    def _copy_node_attributes(source: object, target: object, node_ids: torch.Tensor, total: int) -> None:
        target.num_nodes = int(node_ids.numel())
        for key, value in source.items():
            if key in {"num_nodes", "n_id"}:
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == total:
                target[key] = value[node_ids].clone()
            else:
                target[key] = copy.deepcopy(value)
        target.n_id = node_ids.clone()

    @staticmethod
    def _copy_edge_attributes(
        source: object,
        target: object,
        edge_ids: torch.Tensor,
        remapped_edge_index: torch.Tensor,
        total: int,
    ) -> None:
        target.edge_index = remapped_edge_index
        for key, value in source.items():
            if key in {"edge_index", "e_id"}:
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == total:
                target[key] = value[edge_ids].clone()
            else:
                target[key] = copy.deepcopy(value)
        target.e_id = edge_ids.clone()

    def _build_batch(
        self,
        node_ids: Mapping[str, torch.Tensor],
        edge_ids: Mapping[EdgeType, torch.Tensor],
        diagnostics: Mapping[str, object],
    ) -> SubgraphBatch:
        sampled = HeteroData()
        local_to_global: dict[str, torch.Tensor] = {}
        global_to_local: dict[str, torch.Tensor] = {}
        for node_type in self.data.node_types:
            total = int(self.data[node_type].num_nodes)
            selected = torch.unique(
                torch.as_tensor(node_ids.get(node_type, torch.empty(0)), dtype=torch.long),
                sorted=True,
            )
            if selected.numel() and (bool((selected < 0).any()) or bool((selected >= total).any())):
                raise IndexError(f"Sampled {node_type} node ID is out of range.")
            reverse = torch.full((total,), -1, dtype=torch.long)
            reverse[selected] = torch.arange(selected.numel(), dtype=torch.long)
            self._copy_node_attributes(self.data[node_type], sampled[node_type], selected, total)
            local_to_global[node_type] = selected
            global_to_local[node_type] = reverse

        sampled_edge_ids: dict[EdgeType, torch.Tensor] = {}
        for edge_type in self.data.edge_types:
            source_type, _, target_type = edge_type
            total = self.data[edge_type].edge_index.shape[1]
            selected = torch.unique(
                torch.as_tensor(edge_ids.get(edge_type, torch.empty(0)), dtype=torch.long),
                sorted=True,
            )
            if selected.numel() and (bool((selected < 0).any()) or bool((selected >= total).any())):
                raise IndexError(f"Sampled edge ID is out of range for {edge_type}.")
            global_index = self.data[edge_type].edge_index[:, selected]
            local_index = torch.stack(
                [
                    global_to_local[source_type][global_index[0]],
                    global_to_local[target_type][global_index[1]],
                ]
            )
            if local_index.numel() and bool((local_index < 0).any()):
                raise ValueError(f"Sampled endpoints are missing from node selection for {edge_type}.")
            self._copy_edge_attributes(
                self.data[edge_type], sampled[edge_type], selected, local_index, total
            )
            sampled_edge_ids[edge_type] = selected

        return SubgraphBatch(
            data=sampled,
            global_node_ids=MappingProxyType(local_to_global),
            global_edge_ids=MappingProxyType(sampled_edge_ids),
            global_to_local=MappingProxyType(global_to_local),
            diagnostics=MappingProxyType(dict(diagnostics)),
        )

    @abstractmethod
    def sample(self) -> SubgraphBatch:
        """Construct one deterministic sampled batch."""
