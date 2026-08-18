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
