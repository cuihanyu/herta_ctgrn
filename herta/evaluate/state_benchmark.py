"""Canonical four-representation benchmark for Stage-1 cell states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from herta.evaluate.clustering import clustering_metrics, leiden_clusters


STATE_BENCHMARK_COLUMNS = [
    "representation",
    "n_neighbors",
    "resolution",
    "n_clusters",
    "ARI",
    "NMI",
    "AMI",
    "homogeneity",
    "completeness",
    "v_measure",
    "fowlkes_mallows",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
]


def _validate_representations(
    rna_pca: np.ndarray,
    atac_lsi: np.ndarray,
    herta: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = {
        "RNA_PCA": np.asarray(rna_pca, dtype=np.float32),
        "ATAC_LSI": np.asarray(atac_lsi, dtype=np.float32),
        "HERTA": np.asarray(herta, dtype=np.float32),
    }
    n_cells = arrays["RNA_PCA"].shape[0] if arrays["RNA_PCA"].ndim == 2 else -1
    for name, values in arrays.items():
        if values.ndim != 2 or values.shape[0] != n_cells or values.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty rank-2 matrix aligned by cell.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
    return {
        "RNA_PCA": arrays["RNA_PCA"],
        "ATAC_LSI": arrays["ATAC_LSI"],
        "RNA_ATAC_concat": np.concatenate(
            [arrays["RNA_PCA"], arrays["ATAC_LSI"]], axis=1
        ),
        "HERTA": arrays["HERTA"],
    }


def benchmark_state_representations(
    rna_pca: np.ndarray,
    atac_lsi: np.ndarray,
    herta_cell_embeddings: np.ndarray,
    *,
    labels: np.ndarray | pd.Series | list[str] | None = None,
    cell_ids: list[str] | np.ndarray | pd.Series | None = None,
    random_state: int = 1,
    n_neighbors: int = 20,
    resolution: float = 0.6,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Benchmark RNA, ATAC, concatenated, and HERTA representations identically.

    Biological annotations are passed only to post-clustering metrics. They do
    not affect normalization, neighbor construction, or Leiden assignments.
    """

    representations = _validate_representations(
        rna_pca, atac_lsi, herta_cell_embeddings
    )
    n_cells = next(iter(representations.values())).shape[0]
    label_values = None if labels is None else np.asarray(labels)
    if label_values is not None and len(label_values) != n_cells:
        raise ValueError("labels must align with all four cell representations.")
    if cell_ids is None:
        resolved_ids = np.asarray([f"cell_{index}" for index in range(n_cells)])
    else:
        resolved_ids = np.asarray(cell_ids).astype(str)
        if len(resolved_ids) != n_cells or np.unique(resolved_ids).size != n_cells:
            raise ValueError("cell_ids must be unique and align with the representations.")

    rows: list[dict[str, float | int | str]] = []
    assignments = pd.DataFrame({"cell_id": resolved_ids})
    for name, values in representations.items():
        clusters = leiden_clusters(
            values,
            n_neighbors=n_neighbors,
            resolution=resolution,
            random_state=random_state,
        )
        assignments[name] = clusters
        normalized = normalize(values, norm="l2", axis=1, copy=True)
        try:
            measured = clustering_metrics(
                normalized,
                clusters,
                label_values,
                random_state=random_state,
            )
        except ValueError as error:
            if "between 2 and n_samples - 1 clusters" not in str(error):
                raise
            measured = {}
            if label_values is not None:
                from sklearn.metrics import (
                    adjusted_mutual_info_score,
                    adjusted_rand_score,
                    completeness_score,
                    fowlkes_mallows_score,
                    homogeneity_score,
                    normalized_mutual_info_score,
                    v_measure_score,
                )

                measured = {
                    "ari": adjusted_rand_score(label_values, clusters),
                    "nmi": normalized_mutual_info_score(label_values, clusters),
                    "ami": adjusted_mutual_info_score(label_values, clusters),
                    "homogeneity": homogeneity_score(label_values, clusters),
                    "completeness": completeness_score(label_values, clusters),
                    "v_measure": v_measure_score(label_values, clusters),
                    "fowlkes_mallows": fowlkes_mallows_score(label_values, clusters),
                }
        rows.append(
            {
                "representation": name,
                "n_neighbors": n_neighbors,
                "resolution": resolution,
                "n_clusters": int(np.unique(clusters).size),
                "ARI": measured.get("ari", np.nan),
                "NMI": measured.get("nmi", np.nan),
                "AMI": measured.get("ami", np.nan),
                "homogeneity": measured.get("homogeneity", np.nan),
                "completeness": measured.get("completeness", np.nan),
                "v_measure": measured.get("v_measure", np.nan),
                "fowlkes_mallows": measured.get("fowlkes_mallows", np.nan),
                "silhouette": measured.get("silhouette", np.nan),
                "calinski_harabasz": measured.get("calinski_harabasz", np.nan),
                "davies_bouldin": measured.get("davies_bouldin", np.nan),
            }
        )
    result = pd.DataFrame(rows, columns=STATE_BENCHMARK_COLUMNS)
    if output_dir is not None:
        target = Path(output_dir) / "benchmark"
        target.mkdir(parents=True, exist_ok=True)
        result.to_csv(target / "stage1_representation_metrics.csv", index=False)
        assignments.to_parquet(
            target / "stage1_leiden_clusters.parquet", index=False
        )
    return result
