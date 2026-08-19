"""Clustering evaluation for cell embeddings and biological labels."""

from __future__ import annotations

import numpy as np


def leiden_clusters(
    embeddings: np.ndarray,
    *,
    n_neighbors: int = 20,
    resolution: float = 0.6,
    random_state: int = 0,
) -> np.ndarray:
    """Cluster one cell representation with the canonical Stage-1 Leiden setup."""

    from anndata import AnnData
    import scanpy as sc
    from sklearn.preprocessing import normalize

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] <= n_neighbors:
        raise ValueError(
            "Canonical Stage-1 Leiden requires a rank-2 matrix with more than "
            f"{n_neighbors} cells."
        )
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Leiden embeddings must be non-empty and finite.")
    if n_neighbors != 20 or not np.isclose(resolution, 0.6):
        raise ValueError(
            "The formal Stage-1 benchmark fixes n_neighbors=20 and resolution=0.6."
        )
    adata = AnnData(normalize(values, norm="l2", axis=1, copy=True))
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X",
        random_state=random_state,
    )
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        key_added="cluster",
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )
    return adata.obs["cluster"].astype(str).to_numpy()


def clustering_metrics(
    embeddings: np.ndarray,
    clusters: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    silhouette_sample_size: int | None = None,
    random_state: int = 0,
) -> dict[str, float]:
    """Return common internal and, when labels are supplied, external metrics.

    Larger is better for every metric except ``davies_bouldin``.  Silhouette
    can be evaluated on a deterministic subsample to keep large comparisons
    tractable; all other metrics use every cell.
    """

    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        calinski_harabasz_score,
        completeness_score,
        davies_bouldin_score,
        fowlkes_mallows_score,
        homogeneity_score,
        normalized_mutual_info_score,
        silhouette_score,
        v_measure_score,
    )

    values = np.asarray(embeddings)
    clusters = np.asarray(clusters)
    if values.ndim != 2 or len(values) != len(clusters):
        raise ValueError("embeddings must be a 2D array aligned with clusters.")
    n_clusters = np.unique(clusters).size
    if n_clusters < 2 or n_clusters >= len(clusters):
        raise ValueError("Internal clustering metrics require between 2 and n_samples - 1 clusters.")
    sample_size = silhouette_sample_size
    if sample_size is not None:
        sample_size = min(int(sample_size), len(clusters))
        if sample_size < 2:
            raise ValueError("silhouette_sample_size must be at least 2.")
    result = {
        "silhouette": float(
            silhouette_score(
                values,
                clusters,
                sample_size=sample_size,
                random_state=random_state,
            )
        ),
        "calinski_harabasz": float(calinski_harabasz_score(values, clusters)),
        "davies_bouldin": float(davies_bouldin_score(values, clusters)),
    }
    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) != len(clusters):
            raise ValueError("labels must align with clusters.")
        result["ari"] = float(adjusted_rand_score(labels, clusters))
        result["nmi"] = float(normalized_mutual_info_score(labels, clusters))
        result["ami"] = float(adjusted_mutual_info_score(labels, clusters))
        result["homogeneity"] = float(homogeneity_score(labels, clusters))
        result["completeness"] = float(completeness_score(labels, clusters))
        result["v_measure"] = float(v_measure_score(labels, clusters))
        result["fowlkes_mallows"] = float(fowlkes_mallows_score(labels, clusters))
    return result
