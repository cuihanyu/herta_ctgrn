"""Unsupervised cell neighborhoods from paired RNA PCA and ATAC LSI factors.

The implementation follows the paired-data use of Seurat WNN in ScReNI, while
remaining a small Python-native baseline.  It estimates a cell-specific weight
for each modality and ranks candidate cells by the resulting weighted cosine
similarity.  Cell-type annotations are never consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class WNNResult:
    """Weighted-nearest-neighbor result for paired cells."""

    neighbor_indices: np.ndarray
    neighbor_similarities: np.ndarray
    neighbor_weights: np.ndarray
    rna_weights: np.ndarray
    atac_weights: np.ndarray
    candidate_neighbors: int

    @property
    def n_neighbors(self) -> int:
        return int(self.neighbor_indices.shape[1])

    def to_frame(self, cell_ids: Sequence[str] | None = None) -> pd.DataFrame:
        """Return an inspectable long-form neighbor table."""

        n_cells, k = self.neighbor_indices.shape
        ids = np.asarray(
            [str(i) for i in range(n_cells)] if cell_ids is None else list(map(str, cell_ids)),
            dtype=object,
        )
        if len(ids) != n_cells:
            raise ValueError("cell_ids length must match the WNN cell count.")
        sources = np.repeat(np.arange(n_cells), k)
        targets = self.neighbor_indices.reshape(-1)
        directed = {(int(i), int(j)) for i, j in zip(sources, targets)}
        mutual = np.fromiter(
            ((int(j), int(i)) in directed for i, j in zip(sources, targets)),
            dtype=bool,
            count=len(sources),
        )
        return pd.DataFrame(
            {
                "cell_id": ids[sources],
                "neighbor_id": ids[targets],
                "rank": np.tile(np.arange(1, k + 1), n_cells),
                "similarity": self.neighbor_similarities.reshape(-1),
                "weight": self.neighbor_weights.reshape(-1),
                "mutual": mutual,
                "representation_source": "pca_lsi_wnn",
                "k": k,
                "include_self": False,
            }
        )


def _validate_embedding(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional cell embedding.")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least two cells and one component.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if bool(np.any(np.linalg.norm(array, axis=1) == 0)):
        raise ValueError(f"{name} contains a zero-norm cell vector.")
    return np.ascontiguousarray(array)


def _cosine_knn(unit_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    model = NearestNeighbors(n_neighbors=min(k + 1, len(unit_embedding)), metric="cosine")
    model.fit(unit_embedding)
    distances, indices = model.kneighbors(unit_embedding)
    out_indices = np.empty((len(unit_embedding), k), dtype=np.int64)
    out_similarity = np.empty((len(unit_embedding), k), dtype=np.float32)
    for cell in range(len(unit_embedding)):
        keep = indices[cell] != cell
        cell_indices = indices[cell, keep][:k]
        cell_distances = distances[cell, keep][:k]
        if len(cell_indices) != k:
            raise RuntimeError("Unable to obtain the requested number of non-self neighbors.")
        out_indices[cell] = cell_indices
        out_similarity[cell] = 1.0 - cell_distances
    return out_indices, out_similarity


def _prediction_error(
    unit_embedding: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_similarities: np.ndarray,
) -> np.ndarray:
    affinity = np.clip(neighbor_similarities, 0.0, None)
    row_sum = affinity.sum(axis=1, keepdims=True)
    affinity = np.divide(
        affinity,
        row_sum,
        out=np.full_like(affinity, 1.0 / affinity.shape[1]),
        where=row_sum > 0,
    )
    prediction = np.einsum(
        "nk,nkd->nd", affinity, unit_embedding[neighbor_indices], optimize=True
    )
    prediction = normalize(prediction, norm="l2", axis=1)
    return np.clip(1.0 - np.sum(unit_embedding * prediction, axis=1), 0.0, 2.0)


def _modality_weights(
    rna_unit: np.ndarray,
    atac_unit: np.ndarray,
    rna_neighbors: np.ndarray,
    rna_similarity: np.ndarray,
    atac_neighbors: np.ndarray,
    atac_similarity: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    if temperature <= 0:
        raise ValueError("modality_weight_temperature must be positive.")
    rna_within = _prediction_error(rna_unit, rna_neighbors, rna_similarity)
    rna_cross = _prediction_error(rna_unit, atac_neighbors, atac_similarity)
    atac_within = _prediction_error(atac_unit, atac_neighbors, atac_similarity)
    atac_cross = _prediction_error(atac_unit, rna_neighbors, rna_similarity)
    eps = np.finfo(np.float32).eps
    rna_score = (rna_cross - rna_within) / (rna_cross + rna_within + eps)
    atac_score = (atac_cross - atac_within) / (atac_cross + atac_within + eps)
    logits = np.stack([rna_score, atac_score], axis=1) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights[:, 0].astype(np.float32), weights[:, 1].astype(np.float32)


def build_wnn_neighbors(
    rna_pca: object,
    atac_lsi: object,
    *,
    n_neighbors: int = 20,
    candidate_neighbors: int = 100,
    modality_weight_temperature: float = 0.2,
    neighbor_temperature: float = 0.2,
) -> WNNResult:
    """Build a simplified WNN graph from paired RNA PCA and ATAC LSI vectors.

    Candidate neighbors are the union of modality-specific cosine kNN sets.
    Per-cell RNA/ATAC weights compare within-modality reconstruction against
    reconstruction from the other modality's neighbors.  Final neighbors are
    the highest cell-specific weighted cosine similarities.
    """

    rna = _validate_embedding(rna_pca, "RNA PCA")
    atac = _validate_embedding(atac_lsi, "ATAC LSI")
    if rna.shape[0] != atac.shape[0]:
        raise ValueError("RNA PCA and ATAC LSI must contain the same paired cells.")
    if n_neighbors < 1 or n_neighbors >= rna.shape[0]:
        raise ValueError("n_neighbors must be between 1 and n_cells - 1.")
    if candidate_neighbors < n_neighbors:
        raise ValueError("candidate_neighbors must be at least n_neighbors.")
    if neighbor_temperature <= 0:
        raise ValueError("neighbor_temperature must be positive.")

    resolved_candidates = min(int(candidate_neighbors), rna.shape[0] - 1)
    rna_unit = normalize(rna, norm="l2", axis=1).astype(np.float32, copy=False)
    atac_unit = normalize(atac, norm="l2", axis=1).astype(np.float32, copy=False)
    rna_idx, rna_sim = _cosine_knn(rna_unit, resolved_candidates)
    atac_idx, atac_sim = _cosine_knn(atac_unit, resolved_candidates)
    rna_weight, atac_weight = _modality_weights(
        rna_unit,
        atac_unit,
        rna_idx[:, :n_neighbors],
        rna_sim[:, :n_neighbors],
        atac_idx[:, :n_neighbors],
        atac_sim[:, :n_neighbors],
        modality_weight_temperature,
    )

    final_idx = np.empty((rna.shape[0], n_neighbors), dtype=np.int64)
    final_sim = np.empty((rna.shape[0], n_neighbors), dtype=np.float32)
    for cell in range(rna.shape[0]):
        candidates = np.union1d(rna_idx[cell], atac_idx[cell])
        rna_candidate_sim = rna_unit[candidates] @ rna_unit[cell]
        atac_candidate_sim = atac_unit[candidates] @ atac_unit[cell]
        combined = rna_weight[cell] * rna_candidate_sim + atac_weight[cell] * atac_candidate_sim
        order = np.lexsort((candidates, -combined))[:n_neighbors]
        final_idx[cell] = candidates[order]
        final_sim[cell] = combined[order]

    scaled = final_sim / float(neighbor_temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    neighbor_weight = np.exp(scaled)
    neighbor_weight /= neighbor_weight.sum(axis=1, keepdims=True)
    return WNNResult(
        neighbor_indices=final_idx,
        neighbor_similarities=final_sim,
        neighbor_weights=neighbor_weight.astype(np.float32),
        rna_weights=rna_weight,
        atac_weights=atac_weight,
        candidate_neighbors=resolved_candidates,
    )


def wnn_neighbors_from_anndata(
    rna: object,
    atac: object,
    *,
    n_neighbors: int = 20,
    candidate_neighbors: int = 100,
    n_components: int | None = 256,
    store: bool = True,
    **kwargs: object,
) -> WNNResult:
    """Build WNN from aligned ``X_pca``/``X_lsi`` AnnData representations."""

    if not hasattr(rna, "obsm") or not hasattr(atac, "obsm"):
        raise TypeError("rna and atac must be AnnData-like objects with obsm.")
    if "X_pca" not in rna.obsm or "X_lsi" not in atac.obsm:
        raise ValueError("Expected RNA obsm['X_pca'] and ATAC obsm['X_lsi'].")
    if list(map(str, rna.obs_names)) != list(map(str, atac.obs_names)):
        raise ValueError("RNA and ATAC obs_names must be identically ordered paired cells.")
    rna_pca = np.asarray(rna.obsm["X_pca"])
    atac_lsi = np.asarray(atac.obsm["X_lsi"])
    if n_components is not None:
        if rna_pca.shape[1] < n_components or atac_lsi.shape[1] < n_components:
            raise ValueError(
                f"WNN requires at least {n_components} PCA/LSI components in canonical mode."
            )
        rna_pca = rna_pca[:, :n_components]
        atac_lsi = atac_lsi[:, :n_components]
    result = build_wnn_neighbors(
        rna_pca,
        atac_lsi,
        n_neighbors=n_neighbors,
        candidate_neighbors=candidate_neighbors,
        **kwargs,
    )
    if store:
        metadata = {
            "method": "simplified_pca_lsi_wnn",
            "n_neighbors": result.n_neighbors,
            "candidate_neighbors": result.candidate_neighbors,
            "n_components": n_components,
            "uses_cell_type_labels": False,
        }
        for adata in (rna, atac):
            adata.obsm["wnn_indices"] = result.neighbor_indices.copy()
            adata.obsm["wnn_similarities"] = result.neighbor_similarities.copy()
            adata.obsm["wnn_weights"] = result.neighbor_weights.copy()
            adata.obs["wnn_rna_weight"] = result.rna_weights
            adata.obs["wnn_atac_weight"] = result.atac_weights
            adata.uns["herta_wnn"] = metadata.copy()
    return result
