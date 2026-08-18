"""Shared TF-to-peak-to-gene path scoring for training and inference."""

from __future__ import annotations

import math

import numpy as np
import torch


AGGREGATION_MODES = frozenset(
    {"max", "mean", "sum", "logsumexp", "noisy_or", "topk_sum"}
)
DEFAULT_AGGREGATION_MODE = "mean"
DEFAULT_TOP_K = 10


def canonical_aggregation_mode(mode: str) -> str:
    """Validate an aggregation mode and migrate the legacy default."""

    if mode == "normalized_logsumexp":
        # The old implementation aggregated log probabilities and therefore
        # behaved as a geometric mean. Mean is the closest score-preserving
        # migration and keeps old checkpoints/configuration files loadable.
        return "mean"
    if mode not in AGGREGATION_MODES:
        raise ValueError(
            f"aggregation mode must be one of {sorted(AGGREGATION_MODES)}."
        )
    return mode


def path_score_numpy(
    tf_peak_score: np.ndarray,
    peak_gene_score: np.ndarray,
    *context_factors: np.ndarray,
) -> np.ndarray:
    """Return structural or context-conditioned path probabilities."""

    result = np.asarray(tf_peak_score, dtype=float) * np.asarray(
        peak_gene_score, dtype=float
    )
    for factor in context_factors:
        result = result * np.asarray(factor, dtype=float)
    return result


def path_score_tensor(
    tf_peak_score: torch.Tensor,
    peak_gene_score: torch.Tensor,
    *context_factors: torch.Tensor,
) -> torch.Tensor:
    """Differentiable equivalent of :func:`path_score_numpy`."""

    result = tf_peak_score * peak_gene_score
    for factor in context_factors:
        result = result * factor
    return result


def _validate_top_k(top_k: int | None) -> int:
    resolved = DEFAULT_TOP_K if top_k is None else int(top_k)
    if resolved <= 0:
        raise ValueError("top_k must be positive.")
    return resolved


def aggregate_numpy(
    values: np.ndarray,
    mode: str = DEFAULT_AGGREGATION_MODE,
    *,
    axis: int = 0,
    top_k: int | None = None,
) -> np.ndarray:
    """Aggregate paths with the probability-scale shared definition.

    ``logsumexp`` is normalized by path count. ``sum`` and ``topk_sum`` are
    clipped to one so every mode can be compared with a sigmoid direct score.
    """

    mode = canonical_aggregation_mode(mode)
    values = np.asarray(values, dtype=float)
    if values.shape[axis] == 0:
        raise ValueError("Cannot aggregate an empty path dimension.")
    if not np.isfinite(values).all():
        raise ValueError("Path scores must be finite.")
    values = np.clip(values, 0.0, 1.0)
    if mode == "max":
        result = values.max(axis=axis)
    elif mode == "mean":
        result = values.mean(axis=axis)
    elif mode == "sum":
        result = values.sum(axis=axis)
    elif mode == "logsumexp":
        maximum = values.max(axis=axis, keepdims=True)
        result = np.squeeze(maximum, axis=axis) + np.log(
            np.exp(values - maximum).mean(axis=axis)
        )
    elif mode == "noisy_or":
        result = 1.0 - np.prod(1.0 - values, axis=axis)
    else:
        k = min(_validate_top_k(top_k), values.shape[axis])
        partitioned = np.partition(values, values.shape[axis] - k, axis=axis)
        indices = np.arange(values.shape[axis] - k, values.shape[axis])
        result = np.take(partitioned, indices, axis=axis).sum(axis=axis)
    return np.clip(result, 0.0, 1.0)


def aggregate_tensor(
    values: torch.Tensor,
    mode: str = DEFAULT_AGGREGATION_MODE,
    *,
    dim: int = 0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Differentiable equivalent of :func:`aggregate_numpy`."""

    mode = canonical_aggregation_mode(mode)
    if values.shape[dim] == 0:
        raise ValueError("Cannot aggregate an empty path dimension.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Path scores must be finite.")
    values = values.clamp(0.0, 1.0)
    if mode == "max":
        result = values.max(dim=dim).values
    elif mode == "mean":
        result = values.mean(dim=dim)
    elif mode == "sum":
        result = values.sum(dim=dim)
    elif mode == "logsumexp":
        result = torch.logsumexp(values, dim=dim) - math.log(values.shape[dim])
    elif mode == "noisy_or":
        result = 1.0 - torch.prod(1.0 - values, dim=dim)
    else:
        k = min(_validate_top_k(top_k), values.shape[dim])
        result = torch.topk(values, k=k, dim=dim).values.sum(dim=dim)
    return result.clamp(0.0, 1.0)
