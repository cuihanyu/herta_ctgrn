"""Imbalance-aware ranking metrics for regulatory-edge validation."""

from __future__ import annotations

from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd


def _binary_inputs(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_score: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("y_true and y_score must have the same length.")
    if labels.size == 0:
        raise ValueError("y_true and y_score must not be empty.")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain only finite values.")
    unique = set(pd.unique(labels))
    if not unique.issubset({0, 1, False, True}):
        raise ValueError("y_true must contain binary labels 0/1.")
    return labels.astype(int), scores


def _warn_class_counts(labels: np.ndarray, min_positives: int) -> None:
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive < min_positives:
        warnings.warn(
            f"Only {n_positive} positive labels are available; the metric is unstable.",
            RuntimeWarning,
            stacklevel=3,
        )
    if n_negative < 1:
        warnings.warn(
            "No negative/background labels are available; AUROC is undefined.",
            RuntimeWarning,
            stacklevel=3,
        )


def compute_auroc(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_score: Sequence[float] | np.ndarray | pd.Series,
    *,
    min_positives: int = 2,
) -> float:
    """Compute AUROC and return ``nan`` when only one class is present."""

    from sklearn.metrics import roc_auc_score

    labels, scores = _binary_inputs(y_true, y_score)
    _warn_class_counts(labels, min_positives)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_auprc(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_score: Sequence[float] | np.ndarray | pd.Series,
    *,
    min_positives: int = 2,
) -> float:
    """Compute area under the precision-recall curve for imbalanced labels."""

    from sklearn.metrics import average_precision_score

    labels, scores = _binary_inputs(y_true, y_score)
    _warn_class_counts(labels, min_positives)
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def compute_aupr_ratio(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_score: Sequence[float] | np.ndarray | pd.Series,
    *,
    min_positives: int = 2,
) -> float:
    """Return ``AUPRC / positive_fraction`` (random expectation is about one)."""

    labels, scores = _binary_inputs(y_true, y_score)
    positive_fraction = float(labels.mean())
    if positive_fraction <= 0:
        warnings.warn(
            "AUPR ratio is undefined without positive labels.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    auprc = compute_auprc(labels, scores, min_positives=min_positives)
    return float(auprc / positive_fraction)


def compute_f1_at_thresholds(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_score: Sequence[float] | np.ndarray | pd.Series,
    thresholds: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Compute precision, recall, and F1 for a declared threshold grid."""

    from sklearn.metrics import precision_recall_fscore_support

    labels, scores = _binary_inputs(y_true, y_score)
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError("thresholds must contain finite values.")
        predicted = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predicted,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": threshold,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "n_predicted_positive": int(predicted.sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_topk_precision(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_score: Sequence[float] | np.ndarray | pd.Series,
    k: int | Sequence[int],
) -> pd.DataFrame:
    """Compute precision and recall at one or more top-k cutoffs."""

    labels, scores = _binary_inputs(y_true, y_score)
    requested = [int(k)] if isinstance(k, (int, np.integer)) else [int(value) for value in k]
    if not requested or any(value <= 0 for value in requested):
        raise ValueError("k must contain positive integers.")
    order = np.argsort(-scores, kind="stable")
    n_positive = int(labels.sum())
    prevalence = float(labels.mean())
    rows: list[dict[str, float | int]] = []
    for value in dict.fromkeys(requested):
        effective = min(value, len(labels))
        true_positive = int(labels[order[:effective]].sum())
        rows.append(
            {
                "k": int(effective),
                "requested_k": int(value),
                "true_positives": true_positive,
                "precision_at_k": float(true_positive / effective),
                "recall_at_k": (
                    float(true_positive / n_positive) if n_positive else float("nan")
                ),
                "positive_fraction": prevalence,
                "topk_enrichment": (
                    float((true_positive / effective) / prevalence)
                    if prevalence > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_early_precision(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_score: Sequence[float] | np.ndarray | pd.Series,
    *,
    k: int | None = None,
) -> dict[str, float | int]:
    """Compute top-k precision and lift over prevalence.

    By default ``k`` equals the number of positives, matching a common GRN
    early-precision convention.
    """

    labels, scores = _binary_inputs(y_true, y_score)
    n_positive = int(labels.sum())
    requested = max(1, n_positive) if k is None else int(k)
    top = compute_topk_precision(labels, scores, requested).iloc[0]
    prevalence = float(labels.mean())
    precision = float(top["precision_at_k"])
    return {
        "k": int(top["k"]),
        "early_precision": precision,
        "positive_fraction": prevalence,
        "early_precision_ratio": (
            float(precision / prevalence) if prevalence > 0 else float("nan")
        ),
    }
