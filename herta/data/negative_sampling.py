"""Type-constrained negative sampling."""

from __future__ import annotations

import torch


def positive_lookup(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    """Return a Python set for fast positive-edge exclusion."""

    return {(int(s), int(d)) for s, d in edge_index.t().tolist()}


def sample_negative_targets(
    edge_index: torch.Tensor,
    num_target_nodes: int,
    num_negatives: int,
    generator: torch.Generator | None = None,
    all_positive_edge_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample negative target ids per positive edge without hitting positives."""

    if num_negatives < 1 or num_target_nodes < 1:
        raise ValueError("num_negatives and num_target_nodes must be positive.")
    all_positive = all_positive_edge_index if all_positive_edge_index is not None else edge_index
    all_sources = all_positive[0]
    grouped = all_sources.numel() < 2 or bool(torch.all(all_sources[1:] >= all_sources[:-1]))
    counts = torch.bincount(all_sources) if grouped else None
    starts = torch.cumsum(counts, dim=0) - counts if counts is not None else None

    order = torch.argsort(edge_index[0])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order))
    sorted_sources = edge_index[0, order]
    unique_sources, row_counts = torch.unique_consecutive(sorted_sources, return_counts=True)
    out = torch.empty((edge_index.shape[1], num_negatives), dtype=torch.long)
    row_start = 0
    for source, n_rows_tensor in zip(unique_sources.tolist(), row_counts):
        n_rows = int(n_rows_tensor)
        if grouped and source < len(counts):
            start = int(starts[source])
            positive_targets = all_positive[1, start : start + int(counts[source])]
        else:
            positive_targets = all_positive[1, all_sources == source]
        if len(positive_targets) >= num_target_nodes:
            raise ValueError(f"Source node {source} has no eligible negative targets.")
        candidates = torch.randint(
            num_target_nodes,
            (n_rows, num_negatives),
            generator=generator,
        )
        invalid = torch.isin(candidates, positive_targets)
        while bool(invalid.any()):
            candidates[invalid] = torch.randint(
                num_target_nodes,
                (int(invalid.sum()),),
                generator=generator,
            )
            invalid = torch.isin(candidates, positive_targets)
        out[row_start : row_start + n_rows] = candidates
        row_start += n_rows
    return out[inverse]
