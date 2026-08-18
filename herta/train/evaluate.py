"""Evaluation metrics for link prediction."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch


def _read_edge_table(edges: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(edges, pd.DataFrame):
        return edges.copy()
    path = Path(edges)
    name = path.name.lower()
    if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        sep = "\t" if name.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")) else ","
        return pd.read_csv(path, sep=sep)
    return pd.read_parquet(path)


def _target_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"Edge table must contain target column '{requested}'.")
        return requested
    for candidate in ("gene", "target", "Target gene"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Edge table must contain a target column such as 'gene' or 'target'.")


def global_tf_target_edges(
    inferred_edges: pd.DataFrame | str | Path,
    tf_col: str = "tf",
    target_col: str | None = None,
    score_col: str = "score",
) -> pd.DataFrame:
    """Collapse repeated TF-target predictions into one global ranked edge table."""

    edges = _read_edge_table(inferred_edges)
    target_col = _target_column(edges, target_col)
    required = {tf_col, target_col, score_col}
    if missing := required.difference(edges.columns):
        raise ValueError(f"inferred_edges are missing required columns: {sorted(missing)}")
    edges = edges.dropna(subset=[tf_col, target_col, score_col]).copy()
    edges[score_col] = pd.to_numeric(edges[score_col], errors="raise")
    edges = edges.sort_values(score_col, ascending=False).drop_duplicates([tf_col, target_col])
    edges = edges.drop(columns=["cell_type"], errors="ignore")
    return edges.sort_values(score_col, ascending=False).reset_index(drop=True)


def evaluate_tf_target_edges(
    inferred_edges: pd.DataFrame | str | Path,
    gold_standard: pd.DataFrame | str | Path,
    score_col: str = "score",
    tf_col: str = "tf",
    target_col: str | None = None,
    gold_tf_col: str = "tf",
    gold_target_col: str | None = None,
    gold_label_col: str | None = None,
    top_k: int | Sequence[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate a global TF-target ranking against a positive-edge gold standard.

    The candidate universe is the Cartesian product of TFs and target genes in
    the inferred table. Missing predictions receive score zero. Gold-standard
    edges outside this universe are reported through ``gold_coverage`` and are
    excluded from ranking metrics.
    """

    from sklearn.metrics import average_precision_score, roc_auc_score

    inferred = global_tf_target_edges(inferred_edges, tf_col, target_col, score_col)
    if inferred.empty:
        raise ValueError("inferred_edges must contain at least one scored TF-target edge.")
    target_col = _target_column(inferred, target_col)
    gold = _read_edge_table(gold_standard)
    gold_target_col = _target_column(gold, gold_target_col)
    required_gold = {gold_tf_col, gold_target_col}
    if missing := required_gold.difference(gold.columns):
        raise ValueError(f"gold_standard is missing required columns: {sorted(missing)}")
    if gold_label_col is not None:
        if gold_label_col not in gold.columns:
            raise ValueError(f"gold_standard must contain label column '{gold_label_col}'.")
        gold = gold[pd.to_numeric(gold[gold_label_col], errors="coerce").fillna(0) > 0]

    inferred = inferred.rename(columns={tf_col: "tf", target_col: "gene", score_col: "score"})
    inferred["tf"] = inferred["tf"].astype(str)
    inferred["gene"] = inferred["gene"].astype(str)
    gold_pairs = set(
        map(
            tuple,
            gold[[gold_tf_col, gold_target_col]].dropna().astype(str).drop_duplicates().to_numpy(),
        )
    )
    tfs = inferred["tf"].drop_duplicates().tolist()
    genes = inferred["gene"].drop_duplicates().tolist()
    universe = pd.MultiIndex.from_product([tfs, genes], names=["tf", "gene"])
    score_series = inferred.set_index(["tf", "gene"])["score"]
    scored = score_series.reindex(universe, fill_value=0.0).rename("score").reset_index()
    scored["is_gold_edge"] = [pair in gold_pairs for pair in scored[["tf", "gene"]].itertuples(index=False, name=None)]
    scored = scored.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)

    evaluable_gold = gold_pairs.intersection(set(universe.to_list()))
    y_true = scored["is_gold_edge"].astype(int).to_numpy()
    y_score = scored["score"].to_numpy(dtype=float)
    prevalence = float(y_true.mean()) if len(y_true) else np.nan
    has_both_classes = 0 < y_true.sum() < len(y_true)
    auroc = float(roc_auc_score(y_true, y_score)) if has_both_classes else np.nan
    auprc = float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else np.nan

    if top_k is None:
        requested_k = [max(1, len(evaluable_gold))]
    elif isinstance(top_k, int):
        requested_k = [top_k]
    else:
        requested_k = [int(k) for k in top_k]
    requested_k = list(dict.fromkeys(min(k, len(scored)) for k in requested_k if k > 0))
    if not requested_k:
        raise ValueError("top_k must contain at least one positive integer.")

    top_rows: list[dict[str, float | int]] = []
    for requested in requested_k:
        k = requested
        tp = int(scored.head(k)["is_gold_edge"].sum())
        precision = float(tp / k) if k else np.nan
        recall = float(tp / len(evaluable_gold)) if evaluable_gold else np.nan
        if np.isnan(recall):
            f1 = np.nan
        else:
            f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        top_rows.append(
            {
                "k": int(k),
                "true_positives": tp,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "f1_at_k": f1,
                "early_precision_ratio": float(precision / prevalence) if prevalence > 0 else np.nan,
            }
        )
    top_k_metrics = pd.DataFrame(top_rows)
    primary = top_k_metrics.iloc[0]
    summary = pd.DataFrame(
        [
            {
                "auroc": auroc,
                "auprc": auprc,
                "random_auprc": prevalence,
                "precision_at_k": primary["precision_at_k"],
                "recall_at_k": primary["recall_at_k"],
                "f1_at_k": primary["f1_at_k"],
                "early_precision_ratio": primary["early_precision_ratio"],
                "top_k": int(primary["k"]),
                "n_candidate_edges": int(len(scored)),
                "n_predicted_edges": int(len(inferred)),
                "n_gold_edges": int(len(gold_pairs)),
                "n_evaluable_gold_edges": int(len(evaluable_gold)),
                "gold_coverage": float(len(evaluable_gold) / len(gold_pairs)) if gold_pairs else np.nan,
            }
        ]
    )
    return {"summary": summary, "top_k": top_k_metrics, "scored_edges": scored}


def binary_ranking_metrics(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> dict[str, float]:
    """Return simple separability metrics for positives against sampled negatives."""

    pos = pos_logits.detach().cpu().numpy()
    neg = neg_logits.detach().cpu().numpy().reshape(-1)
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    auc = float((ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg) + 1e-8))
    accuracy = float(((pos[:, None] > neg.reshape(1, -1)).mean()))
    return {"sampled_auc": auc, "pairwise_accuracy": accuracy, "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean())}


def embedding_cell_type_metrics(
    embeddings: pd.DataFrame | str | Path,
    labels: pd.Series | list[str] | np.ndarray | dict[str, str],
    name_col: str = "name",
    k_neighbors: int = 10,
) -> dict[str, pd.DataFrame]:
    """Evaluate whether cell embeddings separate known cell types.

    AUROC and AUPRC are computed one-vs-rest from cosine similarity to each
    cell-type centroid. This gives an unsupervised separability diagnostic for
    visual embedding plots without fitting an extra classifier.
    """

    from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    df = pd.read_parquet(embeddings) if isinstance(embeddings, str | Path) else embeddings.copy()
    if name_col not in df.columns:
        raise ValueError(f"Expected a '{name_col}' column in embeddings.")
    value_cols = [c for c in df.columns if c != name_col and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        raise ValueError("No numeric embedding columns found.")

    if isinstance(labels, dict):
        y = df[name_col].astype(str).map(labels).fillna("unknown").astype(str).to_numpy()
    else:
        y = pd.Series(labels).astype(str).to_numpy()
        if len(y) != len(df):
            raise ValueError("labels must have the same length as embeddings.")

    x = normalize(df[value_cols].to_numpy(dtype=np.float32))
    classes = np.array(sorted(pd.unique(y)))
    if len(classes) < 2:
        raise ValueError("At least two cell types are required for AUROC/AUPRC.")

    centroids = np.vstack([x[y == cls].mean(axis=0) for cls in classes])
    centroids = normalize(centroids)
    sim = cosine_similarity(x, centroids)

    per_class_rows: list[dict[str, float | str | int]] = []
    for idx, cls in enumerate(classes):
        truth = (y == cls).astype(int)
        per_class_rows.append(
            {
                "cell_type": cls,
                "n_cells": int(truth.sum()),
                "auroc": float(roc_auc_score(truth, sim[:, idx])),
                "auprc": float(average_precision_score(truth, sim[:, idx])),
                "mean_similarity_positive": float(sim[truth == 1, idx].mean()),
                "mean_similarity_negative": float(sim[truth == 0, idx].mean()),
            }
        )
    per_class = pd.DataFrame(per_class_rows)

    n_neighbors = min(k_neighbors + 1, len(x))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(x)
    indices = nbrs.kneighbors(return_distance=False)
    neighbor_indices = indices[:, 1:] if n_neighbors > 1 else indices
    knn_purity = float(np.mean([np.mean(y[row] == y[i]) for i, row in enumerate(neighbor_indices)]))
    silhouette = float(silhouette_score(x, y, metric="cosine")) if len(classes) < len(x) else np.nan
    summary = pd.DataFrame(
        [
            {
                "macro_auroc": float(per_class["auroc"].mean()),
                "macro_auprc": float(per_class["auprc"].mean()),
                "knn_purity": knn_purity,
                "silhouette_cosine": silhouette,
                "n_cells": int(len(x)),
                "n_cell_types": int(len(classes)),
                "k_neighbors": int(k_neighbors),
            }
        ]
    )
    return {"summary": summary, "per_class": per_class}


def grn_recovery_metrics(
    inferred_edges: pd.DataFrame | str | Path,
    true_paths: pd.DataFrame | str | Path,
    score_col: str = "score",
    top_k: int = 100,
) -> dict[str, pd.DataFrame]:
    """Evaluate inferred TF-gene edges against simulated ground truth paths."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    inferred = pd.read_parquet(inferred_edges) if isinstance(inferred_edges, str | Path) else inferred_edges.copy()
    truth = pd.read_csv(true_paths) if isinstance(true_paths, str | Path) else true_paths.copy()
    required = {"tf", "gene", score_col}
    if not required.issubset(inferred.columns):
        raise ValueError(f"inferred_edges must contain {sorted(required)}.")
    if not {"tf", "gene"}.issubset(truth.columns):
        raise ValueError("true_paths must contain 'tf' and 'gene'.")

    truth_pairs = set(map(tuple, truth[["tf", "gene"]].drop_duplicates().to_numpy()))
    scored = (
        inferred.groupby(["tf", "gene"], as_index=False)[score_col]
        .max()
        .rename(columns={score_col: "score"})
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    scored["is_true_tf_gene"] = [tuple(row) in truth_pairs for row in scored[["tf", "gene"]].to_numpy()]
    y_true = scored["is_true_tf_gene"].astype(int).to_numpy()
    y_score = scored["score"].to_numpy(dtype=float)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
    k = min(top_k, len(scored))
    summary = pd.DataFrame(
        [
            {
                "auroc": auroc,
                "auprc": auprc,
                "precision_at_k": float(scored.head(k)["is_true_tf_gene"].mean()) if k else np.nan,
                "recall_at_k": float(scored.head(k)["is_true_tf_gene"].sum() / max(1, len(truth_pairs))) if k else np.nan,
                "top_k": int(k),
                "n_inferred_pairs": int(len(scored)),
                "n_true_pairs": int(len(truth_pairs)),
            }
        ]
    )
    return {"summary": summary, "scored_edges": scored}
