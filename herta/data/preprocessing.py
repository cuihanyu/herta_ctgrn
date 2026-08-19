"""Scanpy-style preprocessing used before graph construction.

The public functions follow SIMBA's compact preprocessing style: AnnData is the
central container, results are written back in place, and callers decide when to
use the processed matrix for graph construction.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import numpy as np
from anndata import AnnData
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd


RNA_COUNTS_LAYER = "counts"
RNA_LOG_NORMALIZED_LAYER = "log_normalized"
ATAC_RAW_LAYER = "raw"
ATAC_BINARY_LAYER = "binary"
ATAC_LSI_LOADING_KEY = "LSI_loadings"


@dataclass
class MultiomeFactors:
    """Sparse graph weights and low-dimensional factors for two-stage HERTA."""

    rna_norm: object
    atac_tfidf: object
    rna_cell_scores: np.ndarray
    rna_feature_loadings: np.ndarray
    atac_cell_scores: np.ndarray
    atac_feature_loadings: np.ndarray
    rna_highly_variable: np.ndarray | None = None
    atac_highly_variable: np.ndarray | None = None
    atac_binary: object | None = None
    atac_lsi_first_library_size_correlation: float | None = None
    atac_library_size: np.ndarray | None = None


def _validate_anndata_matrix(adata: AnnData, modality: str) -> None:
    """Validate identifiers and numeric values before preprocessing."""

    if adata.n_obs < 2 or adata.n_vars < 2:
        raise ValueError(f"{modality} AnnData must contain at least two cells and two features.")
    if adata.obs_names.has_duplicates:
        raise ValueError(f"{modality} cell identifiers must be unique.")
    if adata.var_names.has_duplicates:
        raise ValueError(f"{modality} feature identifiers must be unique.")
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(values).all():
        raise ValueError(f"{modality} matrix contains NaN or infinite values.")
    if np.any(values < 0):
        raise ValueError(f"{modality} preprocessing expects non-negative input counts.")


def _copy_layer_once(adata: AnnData, layer: str) -> None:
    """Preserve the first observed matrix under ``layer``."""

    if layer not in adata.layers:
        adata.layers[layer] = adata.X.copy()


def _binarize_accessibility(x: object):
    """Return a sparse-safe binary accessibility matrix without mutating input."""

    if sparse.issparse(x):
        binary = sparse.csr_matrix(x, dtype=np.float32, copy=True)
        binary.data = (binary.data > 0).astype(np.float32)
        binary.eliminate_zeros()
        return binary
    return (np.asarray(x) > 0).astype(np.float32)


def atac_peak_support(
    adata: AnnData,
    *,
    min_cells_floor: int = 5,
    min_cell_fraction: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ATAC peak detection counts, support mask, and resolved threshold.

    Detection is defined from the raw accessibility semantics ``X > 0`` and is
    implemented without densifying sparse matrices.
    """

    if min_cells_floor < 1:
        raise ValueError("min_cells_floor must be at least one.")
    if not 0 <= min_cell_fraction <= 1:
        raise ValueError("min_cell_fraction must be between zero and one.")
    _validate_anndata_matrix(adata, "ATAC")
    if sparse.issparse(adata.X):
        matrix = sparse.csr_matrix(adata.X)
        n_cells = np.asarray((matrix > 0).getnnz(axis=0)).ravel().astype(np.int64)
    else:
        n_cells = np.asarray((np.asarray(adata.X) > 0).sum(axis=0)).ravel().astype(np.int64)
    threshold = max(int(min_cells_floor), int(math.ceil(min_cell_fraction * adata.n_obs)))
    return n_cells, n_cells >= threshold, threshold


def filter_low_support_peaks(
    adata: AnnData,
    *,
    min_cells_floor: int = 5,
    min_cell_fraction: float = 0.001,
) -> dict[str, int | float]:
    """Remove peaks below the HERTA minimum cell-support threshold in place."""

    n_input = int(adata.n_vars)
    n_cells, keep, threshold = atac_peak_support(
        adata,
        min_cells_floor=min_cells_floor,
        min_cell_fraction=min_cell_fraction,
    )
    if not bool(keep.any()):
        raise ValueError(
            "No ATAC peaks pass the minimum support threshold "
            f"df >= {threshold}."
        )
    adata.var["n_cells"] = n_cells
    adata.var["low_support_pass"] = keep
    adata._inplace_subset_var(keep)
    audit: dict[str, int | float] = {
        "n_input_peaks": n_input,
        "n_support_pass": int(keep.sum()),
        "n_low_support_removed": int((~keep).sum()),
        "min_peak_cells": int(threshold),
        "min_cells_floor": int(min_cells_floor),
        "min_cell_fraction": float(min_cell_fraction),
    }
    adata.uns["herta_preprocessing"] = {
        **adata.uns.get("herta_preprocessing", {}),
        "peak_support_filter": audit,
    }
    return audit


def _prepare_atac_lsi_matrix(x: object, binarize: bool = True):
    """Apply binary accessibility, Seurat TF-IDF, L1 scaling, and log1p."""

    accessibility = _binarize_accessibility(x) if binarize else (
        x.copy() if hasattr(x, "copy") else np.asarray(x)
    )
    tfidf = tfidf_seurat(accessibility).astype(np.float32)
    transformed = normalize(tfidf, norm="l1")
    transformed = np.log1p(transformed * 1e4)
    return accessibility, tfidf, transformed


def _resolve_components(matrix: object, requested: int, label: str) -> int:
    if requested <= 0:
        raise ValueError(f"{label} components must be positive.")
    maximum = min(matrix.shape[0], matrix.shape[1])
    if maximum < 1:
        raise ValueError(f"{label} matrix is too small for dimensionality reduction.")
    return min(int(requested), int(maximum))


def _compute_lsi(
    matrix: object,
    n_components: int,
    *,
    n_iter: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return component-standardized ``U Sigma``, V loadings, and singular values."""

    resolved = _resolve_components(matrix, n_components, "ATAC LSI")
    resolved_n_iter = max(15, int(n_iter))
    u, singular_values, vt = randomized_svd(
        matrix,
        resolved,
        n_iter=resolved_n_iter,
        random_state=random_state,
    )
    scores = u * singular_values[None, :]
    scores -= scores.mean(axis=0, keepdims=True)
    scale = scores.std(axis=0, ddof=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    scores = np.ascontiguousarray(scores / scale, dtype=np.float32)
    loadings = np.ascontiguousarray(vt.T, dtype=np.float32)
    return scores, loadings, np.asarray(singular_values, dtype=np.float32)


def _library_size(matrix: object) -> np.ndarray:
    return np.asarray(sparse.csr_matrix(matrix).sum(axis=1)).ravel().astype(np.float64)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape or x.size < 2:
        return float("nan")
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return float("nan")
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def factorize_multiome(
    rna: object,
    atac: object,
    n_rna_components: int = 256,
    n_atac_components: int = 256,
    n_hvg: int | None = 2000,
    rna_hvg_flavor: str = "seurat_v3",
    binarize_atac: bool = False,
    drop_first_lsi: bool = True,
    lsi_n_iter: int = 20,
    random_state: int = 0,
) -> MultiomeFactors:
    """Return graph weights and GLUE-style RNA PCA / ATAC LSI factors.

    This is a low-level compatibility helper for isolated callers and tests.
    The formal pipeline uses :func:`herta.data.dataset.prepare_multiome`, and
    graph construction never invokes this function implicitly.
    """

    rna_adata = AnnData(rna.copy() if hasattr(rna, "copy") else np.asarray(rna))
    preprocess_rna(
        rna_adata,
        n_top_genes=rna_adata.n_vars if n_hvg is None else min(n_hvg, rna_adata.n_vars),
        hvg_flavor=rna_hvg_flavor,
        n_comps=n_rna_components,
        compute_highly_variable=n_hvg is not None,
        use_highly_variable=n_hvg is not None and n_hvg < rna_adata.n_vars,
        random_state=random_state,
    )
    rna_norm = rna_adata.layers[RNA_LOG_NORMALIZED_LAYER].copy()

    atac_binary, atac_tfidf, x_lsi = _prepare_atac_lsi_matrix(
        atac, binarize=binarize_atac
    )
    extra = 1 if drop_first_lsi else 0
    n_atac = _resolve_components(x_lsi, n_atac_components + extra, "ATAC LSI")
    if n_atac <= extra:
        raise ValueError("ATAC matrix is too small for the requested LSI factors.")
    atac_scores_all, atac_loadings_all, _ = _compute_lsi(
        x_lsi,
        n_atac,
        n_iter=lsi_n_iter,
        random_state=random_state,
    )
    atac_library_size = _library_size(atac)
    first_lsi_correlation = _safe_pearson(
        atac_scores_all[:, 0], atac_library_size
    )
    start = extra
    atac_scores = atac_scores_all[..., start:]
    atac_loadings = atac_loadings_all[..., start:]

    return MultiomeFactors(
        rna_norm=rna_norm,
        atac_tfidf=atac_tfidf.tocsr() if sparse.issparse(atac_tfidf) else atac_tfidf,
        rna_cell_scores=np.ascontiguousarray(rna_adata.obsm["X_pca"], dtype=np.float32),
        rna_feature_loadings=np.ascontiguousarray(rna_adata.varm["PCs"], dtype=np.float32),
        atac_cell_scores=np.ascontiguousarray(atac_scores, dtype=np.float32),
        atac_feature_loadings=np.ascontiguousarray(atac_loadings, dtype=np.float32),
        rna_highly_variable=np.asarray(rna_adata.var["highly_variable"], dtype=bool),
        atac_highly_variable=np.ones(atac_tfidf.shape[1], dtype=bool),
        atac_binary=(
            atac_binary.tocsr()
            if sparse.issparse(atac_binary)
            else np.asarray(atac_binary, dtype=np.float32)
        ),
        atac_lsi_first_library_size_correlation=first_lsi_correlation,
        atac_library_size=atac_library_size,
    )


def highly_variable_peaks(
    adata: AnnData,
    min_fraction: float = 0.01,
    max_fraction: float = 0.95,
    n_top_peaks: int | None = 20_000,
    selection_method: str = "detection_variance",
    lsi_loading_key: str = "LSI_loadings",
    exclude_first_lsi: bool = True,
) -> None:
    """Mark informative ATAC peaks for LSI and graph construction.

    Peaks are filtered by cell detection rate, then ranked either by binary
    detection variance or by the norm of previously computed LSI loadings.
    Results follow Scanpy's ``adata.var['highly_variable']`` convention.
    """

    if not 0 <= min_fraction < max_fraction <= 1:
        raise ValueError("Expected 0 <= min_fraction < max_fraction <= 1.")
    if n_top_peaks is not None and n_top_peaks <= 0:
        raise ValueError("n_top_peaks must be positive or None.")
    if selection_method not in {"detection_variance", "lsi_loading"}:
        raise ValueError("selection_method must be 'detection_variance' or 'lsi_loading'.")

    if sparse.issparse(adata.X):
        n_cells = np.asarray((adata.X > 0).getnnz(axis=0)).ravel()
    else:
        n_cells = np.asarray((np.asarray(adata.X) > 0).sum(axis=0)).ravel()
    detection_rate = n_cells / max(adata.n_obs, 1)
    if selection_method == "detection_variance":
        variability_score = detection_rate * (1.0 - detection_rate)
    else:
        if lsi_loading_key not in adata.varm:
            raise ValueError(
                f"adata.varm['{lsi_loading_key}'] is required for selection_method='lsi_loading'. "
                "Run lsi(..., store_loadings=True) first."
            )
        loadings = np.asarray(adata.varm[lsi_loading_key], dtype=np.float32)
        start = 1 if exclude_first_lsi and loadings.shape[1] > 1 else 0
        variability_score = np.linalg.norm(loadings[:, start:], axis=1)
    candidates = np.flatnonzero((detection_rate >= min_fraction) & (detection_rate <= max_fraction))
    if candidates.size == 0:
        raise ValueError(
            "No ATAC peaks pass the detection-rate thresholds; loosen "
            "min_fraction or max_fraction."
        )

    if n_top_peaks is not None and candidates.size > n_top_peaks:
        order = np.argsort(-variability_score[candidates], kind="stable")[:n_top_peaks]
        candidates = candidates[order]
    highly_variable = np.zeros(adata.n_vars, dtype=bool)
    highly_variable[candidates] = True

    adata.var["n_cells"] = n_cells
    adata.var["detection_rate"] = detection_rate
    adata.var["variability_score"] = variability_score
    if selection_method == "lsi_loading":
        adata.var["lsi_loading_score"] = variability_score
    adata.var["highly_variable"] = highly_variable
    adata.uns["herta_preprocessing"] = {
        **adata.uns.get("herta_preprocessing", {}),
        "highly_variable_peaks": {
            "min_fraction": min_fraction,
            "max_fraction": max_fraction,
            "n_top_peaks": n_top_peaks,
            "n_selected": int(highly_variable.sum()),
            "selection_method": selection_method,
            "lsi_loading_key": lsi_loading_key if selection_method == "lsi_loading" else None,
            "exclude_first_lsi": exclude_first_lsi if selection_method == "lsi_loading" else None,
        },
    }


def preprocess_rna(
    adata: AnnData,
    n_top_genes: int = 2000,
    hvg_flavor: str = "seurat_v3",
    n_comps: int = 256,
    svd_solver: str = "auto",
    *,
    counts_layer: str = RNA_COUNTS_LAYER,
    log_normalized_layer: str = RNA_LOG_NORMALIZED_LAYER,
    target_sum: float | None = None,
    scale_max_value: float | None = None,
    compute_highly_variable: bool = True,
    use_highly_variable: bool = True,
    random_state: int = 0,
) -> None:
    """Run the GLUE-style scanpy RNA preprocessing workflow in place."""

    import scanpy as sc

    if n_top_genes <= 0:
        raise ValueError("n_top_genes must be positive.")
    _copy_layer_once(adata, counts_layer)
    adata.X = adata.layers[counts_layer].copy()
    _validate_anndata_matrix(adata, "RNA")
    resolved_hvg = min(int(n_top_genes), adata.n_vars)
    count_scale_hvg = hvg_flavor in {"seurat_v3", "seurat_v3_paper"}
    if compute_highly_variable and count_scale_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=resolved_hvg, flavor=hvg_flavor)
    elif not compute_highly_variable:
        adata.var["highly_variable"] = True
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.layers[log_normalized_layer] = adata.X.copy()
    if compute_highly_variable and not count_scale_hvg:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=resolved_hvg,
            flavor=hvg_flavor,
        )
    sc.pp.scale(adata, max_value=scale_max_value)
    pca_vars = int(adata.var["highly_variable"].sum()) if use_highly_variable else adata.n_vars
    n_pca = min(int(n_comps), adata.n_obs - 1, pca_vars - 1)
    if n_pca < 1:
        raise ValueError("RNA matrix is too small for PCA after feature selection.")
    pca_kwargs = {
        "n_comps": n_pca,
        "svd_solver": svd_solver,
        "random_state": random_state,
    }
    if "mask_var" in inspect.signature(sc.tl.pca).parameters:
        pca_kwargs["mask_var"] = "highly_variable" if use_highly_variable else None
    else:
        pca_kwargs["use_highly_variable"] = use_highly_variable
    sc.tl.pca(adata, **pca_kwargs)
    adata.uns["herta_preprocessing"] = {
        **adata.uns.get("herta_preprocessing", {}),
        "rna": {
            "counts_layer": counts_layer,
            "log_normalized_layer": log_normalized_layer,
            "highly_variable_genes": {
                "computed": compute_highly_variable,
                "n_top_genes": resolved_hvg,
                "requested_n_top_genes": n_top_genes,
                "flavor": hvg_flavor,
                "input_scale": "counts" if count_scale_hvg else "log_normalized",
            },
            "normalize_total": {"target_sum": target_sum},
            "log1p": {"base": None},
            "scale": {"max_value": scale_max_value, "zero_center": True},
            "pca": {
                "n_comps": n_pca,
                "requested_n_comps": n_comps,
                "svd_solver": svd_solver,
                "use_highly_variable": use_highly_variable,
                "random_state": random_state,
            },
            "matrix_semantics": {
                "input": "non_negative_counts",
                "x_after_preprocessing": "scaled_log_normalized_expression",
            },
        },
    }


def preprocess_atac(
    adata: AnnData,
    binarize: bool = False,
    eps: float | None = None,
    n_components: int = 256,
    n_iter: int = 20,
    *,
    drop_first_component: bool = True,
    random_state: int = 0,
    raw_layer: str = ATAC_RAW_LAYER,
    binary_layer: str = ATAC_BINARY_LAYER,
    loading_key: str = ATAC_LSI_LOADING_KEY,
) -> None:
    """Compute scGLUE-style TF-IDF/LSI, optionally binarizing first."""

    _copy_layer_once(adata, raw_layer)
    adata.X = adata.layers[raw_layer].copy()
    _validate_anndata_matrix(adata, "ATAC")
    x, x_tfidf, x_lsi_input = _prepare_atac_lsi_matrix(adata.X, binarize=binarize)
    if binarize:
        adata.layers[binary_layer] = x.copy()
    extra = int(drop_first_component)
    resolved = _resolve_components(
        x_lsi_input, n_components + extra, "ATAC LSI"
    )
    if resolved <= extra:
        raise ValueError("ATAC matrix is too small for the requested LSI factors.")
    scores_all, loadings_all, singular_values_all = _compute_lsi(
        x_lsi_input,
        resolved,
        n_iter=n_iter,
        random_state=random_state,
    )
    first_lsi_correlation = _safe_pearson(
        scores_all[:, 0], _library_size(adata.layers[raw_layer])
    )
    x_lsi = scores_all[:, extra:]
    loadings = loadings_all[:, extra:]
    singular_values = singular_values_all[extra:]
    adata.obsm["X_lsi"] = x_lsi
    adata.varm[loading_key] = loadings
    adata.var["highly_variable"] = True
    if sparse.issparse(x_tfidf):
        x_tfidf = x_tfidf.tocsr()
    adata.X = x_tfidf
    adata.obs["n_counts"] = np.asarray(x.sum(axis=1)).ravel()
    adata.var["n_counts"] = np.asarray(x.sum(axis=0)).ravel()
    if sparse.issparse(x):
        adata.obs["n_peaks"] = np.asarray(x.getnnz(axis=1)).ravel()
        adata.var["n_cells"] = np.asarray(x.getnnz(axis=0)).ravel()
    else:
        adata.obs["n_peaks"] = np.asarray((x > 0).sum(axis=1)).ravel()
        adata.var["n_cells"] = np.asarray((x > 0).sum(axis=0)).ravel()
    adata.uns["herta_preprocessing"] = {
        **adata.uns.get("herta_preprocessing", {}),
        "atac": {
            "raw_layer": raw_layer,
            "binary_layer": binary_layer if binarize else None,
            "binarize": binarize,
            "tfidf": {"flavor": "seurat"},
            "lsi": {
                "n_components": int(x_lsi.shape[1]),
                "requested_n_components": n_components,
                "n_iter": n_iter,
                "resolved_n_iter": max(15, int(n_iter)),
                "random_state": random_state,
                "drop_first_component": bool(drop_first_component),
                "cell_score_mode": "component_standardized_u_sigma",
                "feature_loading_mode": "v",
                "first_component_library_size_correlation": first_lsi_correlation,
                "loading_key": loading_key,
                "singular_values": singular_values.tolist(),
                "highly_variable_peak_semantics": "all_peaks_supplied_to_lsi",
            },
        },
    }


def tfidf_seurat(x: object):
    """Seurat v3-style TF-IDF normalization used by GLUE LSI."""

    if sparse.issparse(x):
        matrix = sparse.csr_matrix(x, dtype=np.float32)
        row_sum = np.asarray(matrix.sum(axis=1)).ravel()
        col_sum = np.asarray(matrix.sum(axis=0)).ravel()
        inv_row = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
        idf = np.divide(matrix.shape[0], col_sum, out=np.zeros_like(col_sum), where=col_sum > 0)
        return sparse.diags(inv_row) @ matrix @ sparse.diags(idf)
    matrix = np.asarray(x, dtype=np.float32)
    row_sum = matrix.sum(axis=1, keepdims=True)
    col_sum = matrix.sum(axis=0, keepdims=True)
    tf = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
    idf = np.divide(matrix.shape[0], col_sum, out=np.zeros_like(col_sum), where=col_sum > 0)
    return tf * idf


def lsi(
    adata: AnnData,
    n_components: int = 256,
    use_highly_variable: bool = False,
    n_iter: int = 20,
    binarize: bool = False,
    drop_first_component: bool = True,
    store_loadings: bool = False,
    loading_key: str = ATAC_LSI_LOADING_KEY,
    **kwargs,
) -> None:
    """Run LSI after optional ATAC binarization, using GLUE's TF-IDF workflow."""

    _validate_anndata_matrix(adata, "ATAC")
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0
    if "n_iter" not in kwargs:
        kwargs["n_iter"] = n_iter
    if use_highly_variable and "highly_variable" not in adata.var:
        raise ValueError("use_highly_variable=True requires adata.var['highly_variable'].")
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    _, _, x_norm = _prepare_atac_lsi_matrix(adata_use.X, binarize=binarize)
    extra = int(drop_first_component)
    resolved = _resolve_components(x_norm, n_components + extra, "ATAC LSI")
    if resolved <= extra:
        raise ValueError("ATAC matrix is too small for the requested LSI factors.")
    scores_all, loadings_all, singular_values_all = _compute_lsi(
        x_norm,
        resolved,
        n_iter=int(kwargs["n_iter"]),
        random_state=int(kwargs["random_state"]),
    )
    first_lsi_correlation = _safe_pearson(
        scores_all[:, 0], _library_size(adata_use.X)
    )
    x_lsi = scores_all[:, extra:]
    selected_loadings = loadings_all[:, extra:]
    singular_values = singular_values_all[extra:]
    adata.obsm["X_lsi"] = x_lsi
    if not use_highly_variable:
        adata.var["highly_variable"] = True
    if store_loadings:
        loadings = np.zeros((adata.n_vars, selected_loadings.shape[1]), dtype=np.float32)
        if use_highly_variable:
            loadings[np.asarray(adata.var["highly_variable"], dtype=bool)] = selected_loadings
        else:
            loadings = selected_loadings
        adata.varm[loading_key] = loadings
    adata.uns["herta_preprocessing"] = {
        **adata.uns.get("herta_preprocessing", {}),
        "lsi": {
            "binarize": binarize,
            "use_highly_variable": use_highly_variable,
            "n_components": int(x_lsi.shape[1]),
            "requested_n_components": n_components,
            "n_iter": kwargs["n_iter"],
            "resolved_n_iter": max(15, int(kwargs["n_iter"])),
            "random_state": kwargs["random_state"],
            "drop_first_component": bool(drop_first_component),
            "cell_score_mode": "component_standardized_u_sigma",
            "feature_loading_mode": "v",
            "first_component_library_size_correlation": first_lsi_correlation,
            "singular_values": singular_values.tolist(),
            "store_loadings": store_loadings,
            "loading_key": loading_key if store_loadings else None,
            "highly_variable_peak_semantics": (
                "preexisting_mask" if use_highly_variable else "all_peaks_supplied_to_lsi"
            ),
        },
    }
