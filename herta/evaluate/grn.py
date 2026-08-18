"""GRN ranking stability across seeds."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_grn_reference(
    scores: pd.DataFrame,
    reference: pd.DataFrame,
    score_column: str = "score",
) -> dict[str, float]:
    """Evaluate ranked TF-gene edges against a binary reference network."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    keys = ["tf", "gene"]
    if "cluster" in scores and "cluster" in reference:
        keys.insert(0, "cluster")
    missing_scores = set(keys + [score_column]).difference(scores.columns)
    missing_reference = set(keys).difference(reference.columns)
    if missing_scores or missing_reference:
        raise ValueError(
            f"Missing score columns {sorted(missing_scores)} or reference columns {sorted(missing_reference)}."
        )
    ranked = scores.groupby(keys, as_index=False)[score_column].max()
    positives = reference[keys].drop_duplicates().assign(label=1)
    ranked = ranked.merge(positives, on=keys, how="left")
    labels = ranked["label"].fillna(0).to_numpy(dtype=int)
    values = ranked[score_column].to_numpy(dtype=float)
    if labels.min() == labels.max():
        raise ValueError("Reference evaluation requires positive and negative candidate edges.")
    n_positive = int(labels.sum())
    top = np.argsort(values)[::-1][:n_positive]
    early_precision = float(labels[top].mean()) if n_positive else np.nan
    prevalence = float(labels.mean())
    return {
        "auprc": float(average_precision_score(labels, values)),
        "auroc": float(roc_auc_score(labels, values)),
        "early_precision": early_precision,
        "early_precision_ratio": early_precision / prevalence if prevalence > 0 else np.nan,
        "n_candidates": float(len(labels)),
        "n_positives": float(n_positive),
    }


def grn_stability(first: pd.DataFrame, second: pd.DataFrame, top_k: int = 1000) -> dict[str, float]:
    from scipy.stats import spearmanr

    keys = ["cluster", "tf", "gene"] if "cluster" in first and "cluster" in second else ["tf", "gene"]
    merged = first[keys + ["score"]].merge(second[keys + ["score"]], on=keys, suffixes=("_first", "_second"))
    correlation = float(spearmanr(merged["score_first"], merged["score_second"]).statistic) if len(merged) > 1 else np.nan
    first_top = set(map(tuple, first.nlargest(top_k, "score")[keys].to_numpy()))
    second_top = set(map(tuple, second.nlargest(top_k, "score")[keys].to_numpy()))
    union = first_top | second_top
    return {"spearman": correlation, "top_k_jaccard": float(len(first_top & second_top) / len(union)) if union else np.nan}
