"""Relation-balanced positive edge sampling."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData


def sample_positive_edges(
    data: HeteroData,
    edge_type: tuple[str, str, str],
    batch_size: int,
    generator: torch.Generator | None = None,
    *,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample positive edges and weights for one relation."""

    edge_index = data[edge_type].edge_index
    weights = data[edge_type].edge_weight
    n_edges = edge_index.shape[1]
    if n_edges == 0:
        raise ValueError(f"Relation {edge_type} has no training edges.")
    if batch_size >= n_edges:
        idx = torch.arange(n_edges)
    else:
        idx = torch.randperm(n_edges, generator=generator)[:batch_size]
    sampled = (edge_index[:, idx], weights[idx])
    return (*sampled, idx) if return_indices else sampled


def sample_positive_edges_by_source(
    data: HeteroData,
    edge_type: tuple[str, str, str],
    batch_size_sources: int,
    edges_per_source: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a fixed number of relation edges for a batch of source nodes.

    Stage-1 relations are stored grouped by cell source. Sampling cells first
    prevents large-degree or abundant relations from leaving most cells unseen
    during short training runs.
    """

    if batch_size_sources < 1 or edges_per_source < 1:
        raise ValueError("batch_size_sources and edges_per_source must be positive.")
    edge_index = data[edge_type].edge_index
    weights = data[edge_type].edge_weight
    if edge_index.shape[1] == 0:
        raise ValueError(f"Relation {edge_type} has no training edges.")
    sources = edge_index[0]
    if sources.numel() > 1 and not bool(torch.all(sources[1:] >= sources[:-1])):
        raise ValueError("Source-balanced sampling requires edges grouped by source node.")
    n_source_nodes = int(data[edge_type[0]].num_nodes)
    counts = torch.bincount(sources, minlength=n_source_nodes)
    eligible = torch.nonzero(counts > 0, as_tuple=False).flatten()
    chosen = eligible[
        torch.randperm(len(eligible), generator=generator)[: min(batch_size_sources, len(eligible))]
    ]
    starts = torch.cumsum(counts, dim=0) - counts
    offsets = (
        torch.rand((len(chosen), edges_per_source), generator=generator)
        * counts[chosen, None]
    ).long()
    indices = (starts[chosen, None] + offsets).reshape(-1)
    return edge_index[:, indices], weights[indices]


def sample_cells_with_all_relations(
    data: HeteroData,
    edge_types: tuple[tuple[str, str, str], ...],
    batch_size_cells: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one shared cell batch with positives in every Stage-1 relation."""

    if batch_size_cells < 1:
        raise ValueError("batch_size_cells must be positive.")
    eligible: torch.Tensor | None = None
    n_cells = int(data["cell"].num_nodes)
    for edge_type in edge_types:
        sources = data[edge_type].edge_index[0]
        present = torch.zeros(n_cells, dtype=torch.bool)
        present[sources.unique()] = True
        eligible = present if eligible is None else eligible & present
    assert eligible is not None
    candidates = torch.nonzero(eligible, as_tuple=False).flatten()
    if candidates.numel() == 0:
        raise ValueError("No cells have observed positives in every Stage-1 relation.")
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[: min(batch_size_cells, candidates.numel())]]


def sample_positive_edges_for_cells(
    data: HeteroData,
    edge_type: tuple[str, str, str],
    cell_ids: torch.Tensor,
    edges_per_cell: int,
    generator: torch.Generator | None = None,
    *,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample relation positives for an explicit shared batch of cell IDs."""

    if edges_per_cell < 1:
        raise ValueError("edges_per_cell must be positive.")
    edge_index = data[edge_type].edge_index
    weights = data[edge_type].edge_weight
    sources = edge_index[0]
    if sources.numel() > 1 and not bool(torch.all(sources[1:] >= sources[:-1])):
        raise ValueError("Cell-balanced sampling requires source-grouped edges.")
    counts = torch.bincount(sources, minlength=int(data["cell"].num_nodes))
    if bool((counts[cell_ids] == 0).any()):
        raise ValueError(f"Some sampled cells have no positives for {edge_type}.")
    starts = torch.cumsum(counts, dim=0) - counts
    offsets = (
        torch.rand((cell_ids.numel(), edges_per_cell), generator=generator)
        * counts[cell_ids, None]
    ).long()
    indices = (starts[cell_ids, None] + offsets).reshape(-1)
    sampled = (edge_index[:, indices], weights[indices])
    return (*sampled, indices) if return_indices else sampled
