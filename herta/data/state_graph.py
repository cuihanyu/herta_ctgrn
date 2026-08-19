"""Stage-1 cell-state graph construction."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import torch
from scipy import sparse
from torch_geometric.data import HeteroData

from herta.data.dataset import MultiomeData
from herta.data.edge_tables import EdgeTableBundle, build_edge_table_bundle
from herta.data.heterogeneous_graph import (
    HeterogeneousGraphBuildResult,
    QueryEdgeTensors,
    edge_tables_to_heterodata,
)
from herta.data.preprocessing import MultiomeFactors


LOGGER = logging.getLogger(__name__)


STATE_RELATIONS = {
    "cg": ("cell", "expresses", "gene"),
    "cp": ("cell", "accessible", "peak"),
}


@dataclass
class StateGraphBuildResult:
    data: HeteroData
    features: dict[str, torch.Tensor]
    factors: MultiomeFactors
    names: dict[str, list[str]]
    metadata: dict[str, object]
    edge_table_bundle: EdgeTableBundle
    query_tensors: dict[str, QueryEdgeTensors]
    audit: dict[str, object]


def _row_topk(matrix: object, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = sparse.csr_matrix(matrix)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row_id in range(x.shape[0]):
        start, end = x.indptr[row_id], x.indptr[row_id + 1]
        idx = x.indices[start:end]
        val = x.data[start:end]
        keep = val > 0
        idx, val = idx[keep], val[keep]
        if len(val) > k:
            chosen = np.argpartition(val, -k)[-k:]
            idx, val = idx[chosen], val[chosen]
        denom = float(val.sum()) + 1e-8
        rows.extend([row_id] * len(idx))
        cols.extend(idx.astype(int).tolist())
        values.extend((val / denom).astype(float).tolist())
    return np.asarray(rows), np.asarray(cols), np.asarray(values, dtype=np.float32)


def _add_relation(data: HeteroData, edge_type: tuple[str, str, str], matrix: object, top_k: int) -> None:
    src, dst, weight = _row_topk(matrix, top_k)
    edge_index = torch.as_tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.as_tensor(weight, dtype=torch.float32)
    data[edge_type].edge_index = edge_index
    data[edge_type].edge_weight = edge_weight
    reverse = (edge_type[2], f"rev_{edge_type[1]}", edge_type[0])
    data[reverse].edge_index = edge_index.flip(0)
    data[reverse].edge_weight = edge_weight.clone()


def build_state_graph(
    multiome: MultiomeData,
    factors: MultiomeFactors | None = None,
    top_cell_gene: int | None = 128,
    top_cell_peak: int | None = 256,
    gene_cell_quantile: float | None = 0.95,
) -> StateGraphBuildResult:
    """Build Stage 1 from canonically prepared data and factors.

    Preprocessing is intentionally not available here.  Call
    :func:`herta.data.dataset.prepare_multiome` once and pass its
    ``metadata["state_factors"]`` result explicitly.
    """

    if factors is None:
        raise ValueError(
            "build_state_graph requires canonical MultiomeFactors; call "
            "prepare_multiome(...) once and pass metadata['state_factors']."
        )
    if not isinstance(factors, MultiomeFactors):
        raise TypeError("factors must be a MultiomeFactors instance.")

    if multiome.rna.shape[0] != multiome.atac.shape[0]:
        raise ValueError("RNA and ATAC matrices must have the same number of cells.")
    if multiome.rna.shape[1] != len(multiome.genes):
        raise ValueError("RNA columns must align with the gene annotation table.")
    if multiome.atac.shape[1] != len(multiome.peaks):
        raise ValueError("ATAC columns must align with the peak annotation table.")
    if "retained_peak" in multiome.peaks and not bool(
        np.asarray(multiome.peaks["retained_peak"], dtype=bool).all()
    ):
        raise ValueError("MultiomeData.peaks contains peaks not selected by the retained union.")
    expected_matrices = {
        "rna_norm": multiome.rna.shape,
        "atac_tfidf": multiome.atac.shape,
    }
    for name, expected_shape in expected_matrices.items():
        value = getattr(factors, name)
        if value.shape != expected_shape:
            raise ValueError(
                f"Precomputed {name} must have shape {expected_shape}; "
                f"observed {value.shape}."
            )
        stored = value.data if sparse.issparse(value) else np.asarray(value)
        if not np.isfinite(stored).all():
            raise ValueError(f"Precomputed {name} contains NaN or infinite values.")
    expected_factors = {
        "rna_cell_scores": multiome.rna.shape[0],
        "rna_feature_loadings": multiome.rna.shape[1],
        "atac_cell_scores": multiome.atac.shape[0],
        "atac_feature_loadings": multiome.atac.shape[1],
    }
    arrays: dict[str, np.ndarray] = {}
    for name, rows in expected_factors.items():
        value = np.asarray(getattr(factors, name))
        if value.ndim != 2 or value.shape[0] != rows:
            raise ValueError(
                f"Precomputed {name} must have shape ({rows}, n_components); "
                f"observed {value.shape}."
            )
        if value.shape[1] < 1:
            raise ValueError(f"Precomputed {name} must contain at least one component.")
        if not np.isfinite(value).all():
            raise ValueError(f"Precomputed {name} contains NaN or infinite values.")
        arrays[name] = value
    if (
        arrays["rna_cell_scores"].shape[1]
        != arrays["rna_feature_loadings"].shape[1]
    ):
        raise ValueError("RNA cell scores and feature loadings must use the same components.")
    if (
        arrays["atac_cell_scores"].shape[1]
        != arrays["atac_feature_loadings"].shape[1]
    ):
        raise ValueError("ATAC cell scores and feature loadings must use the same components.")
    for name, expected_length in (
        ("rna_highly_variable", multiome.rna.shape[1]),
        ("atac_highly_variable", multiome.atac.shape[1]),
    ):
        value = getattr(factors, name)
        if value is not None and np.asarray(value).shape != (expected_length,):
            raise ValueError(f"Precomputed {name} must have length {expected_length}.")
    if (
        factors.atac_binary is not None
        and factors.atac_binary.shape != multiome.atac.shape
    ):
        raise ValueError(
            f"Precomputed atac_binary must have shape {multiome.atac.shape}; "
            f"observed {factors.atac_binary.shape}."
        )
    if factors.atac_library_size is not None:
        library_size = np.asarray(factors.atac_library_size)
        if library_size.shape != (multiome.atac.shape[0],) or not np.isfinite(
            library_size
        ).all():
            raise ValueError("Precomputed atac_library_size must be finite and cell-aligned.")
    gene_names = multiome.genes["gene"].astype(str).tolist()
    peak_names = multiome.peaks["peak"].astype(str).tolist()
    cell_names = list(multiome.cell_names or [f"cell_{i}" for i in range(multiome.rna.shape[0])])
    if len(cell_names) != multiome.rna.shape[0]:
        raise ValueError("cell_names must align with the RNA and ATAC matrices.")
    features = {
        "cell": torch.from_numpy(
            np.ascontiguousarray(
                np.concatenate(
                    [arrays["rna_cell_scores"], arrays["atac_cell_scores"]], axis=1
                ),
                dtype=np.float32,
            )
        ),
        "gene": torch.from_numpy(
            np.ascontiguousarray(arrays["rna_feature_loadings"], dtype=np.float32)
        ),
        "peak": torch.from_numpy(
            np.ascontiguousarray(arrays["atac_feature_loadings"], dtype=np.float32)
        ),
    }
    expression = sparse.csr_matrix(factors.rna_norm)
    # The prepared gene universe is HVG union eligible TF. Do not mask the
    # non-HVG TF columns: observed TF expression is part of the Stage-1 graph.
    expression.eliminate_zeros()
    accessibility = sparse.csr_matrix(factors.atac_tfidf)
    edge_table_bundle = build_edge_table_bundle(
        multiome,
        expression=expression,
        accessibility=accessibility,
        top_cell_gene=top_cell_gene,
        top_cell_peak=top_cell_peak,
        gene_cell_quantile=gene_cell_quantile,
    )
    graph = build_state_heterodata_from_edges(
        multiome,
        edge_table_bundle,
        factors=factors,
        node_features=features,
    )
    rna_hvg = (
        np.asarray(factors.rna_highly_variable, dtype=bool)
        if factors.rna_highly_variable is not None
        else np.ones(multiome.rna.shape[1], dtype=bool)
    )
    atac_hvp = (
        np.asarray(factors.atac_highly_variable, dtype=bool)
        if factors.atac_highly_variable is not None
        else np.ones(multiome.atac.shape[1], dtype=bool)
    )
    tf_mask = (
        np.asarray(multiome.genes["is_tf"], dtype=bool)
        if "is_tf" in multiome.genes
        else np.asarray(
            multiome.genes["gene"].astype(str).isin(set(multiome.tf_names or [])),
            dtype=bool,
        )
    )
    tf_expression_nnz = np.asarray(expression[:, tf_mask].getnnz(axis=0)).ravel()
    n_cells = int(multiome.rna.shape[0])
    rna_library_size = np.asarray(
        sparse.csr_matrix(multiome.rna).sum(axis=1)
    ).ravel()
    atac_library_size = np.asarray(
        sparse.csr_matrix(multiome.atac).sum(axis=1)
    ).ravel()

    def summary(values: object) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64).ravel()
        return {
            "min": float(np.min(array)),
            "median": float(np.median(array)),
            "mean": float(np.mean(array)),
            "max": float(np.max(array)),
        }

    relation_diagnostics: dict[str, object] = {}
    for relation, edge_type in STATE_RELATIONS.items():
        edge_index = graph.data[edge_type].edge_index
        edge_weight = graph.data[edge_type].edge_weight
        degree = torch.bincount(edge_index[0], minlength=n_cells).cpu().numpy()
        relation_diagnostics[f"{relation}_edge_count"] = int(edge_index.shape[1])
        relation_diagnostics[f"{relation}_degree"] = summary(degree)
        relation_diagnostics[f"{relation}_edge_weight"] = summary(
            edge_weight.cpu().numpy()
        )
    cp_edge_type = STATE_RELATIONS["cp"]
    cp_unique_peaks = int(torch.unique(graph.data[cp_edge_type].edge_index[1]).numel())
    feature_nonfinite = {
        node_type: int((~torch.isfinite(values)).sum())
        for node_type, values in features.items()
    }
    feature_all_zero = {
        node_type: int((values.abs().sum(dim=1) == 0).sum())
        for node_type, values in features.items()
    }
    graph.audit.update(
        {
            "node_types": list(graph.data.node_types),
            "rna_hvg_count": int(rna_hvg.sum()),
            "atac_highly_variable_peak_count": int(atac_hvp.sum()),
            "rna_graph_matrix": "log_normalized_expression_hvg_union_tf",
            "tf_gene_node_count": int(tf_mask.sum()),
            "tf_with_observed_expression_count": int((tf_expression_nnz > 0).sum()),
            "all_zero_tf_count": int((tf_expression_nnz == 0).sum()),
            "all_zero_tf_names": np.asarray(gene_names, dtype=object)[tf_mask][
                tf_expression_nnz == 0
            ].astype(str).tolist(),
            "cell_gene_selection": "row_top_k_union_strict_per_gene_quantile",
            "top_cell_gene": top_cell_gene,
            "gene_cell_quantile": gene_cell_quantile,
            "cell_peak_selection": "per_cell_tfidf_top_k",
            "top_cell_peak": top_cell_peak,
            "cp_unique_peak_count": cp_unique_peaks,
            "rna_library_size": summary(rna_library_size),
            "atac_library_size": summary(atac_library_size),
            "atac_lsi_first_library_size_correlation": (
                factors.atac_lsi_first_library_size_correlation
            ),
            "feature_nonfinite_count": feature_nonfinite,
            "feature_all_zero_node_count": feature_all_zero,
            "peak_universe": (
                "coordinate_support_qc_then_lsi_loading_variability"
                if "retained_peak" in multiome.peaks
                else "supplied_peak_universe"
            ),
            "peak_retention_reason_counts": (
                multiome.peaks["retention_reason"].astype(str).value_counts().to_dict()
                if "retention_reason" in multiome.peaks
                else {}
            ),
            "rna_factor_dim": int(factors.rna_cell_scores.shape[1]),
            "atac_factor_dim": int(factors.atac_cell_scores.shape[1]),
            **relation_diagnostics,
        }
    )
    LOGGER.info(
        "CP graph: edges=%d mean_degree=%.3f median_degree=%.3f unique_peaks=%d "
        "weight_min=%.6g weight_mean=%.6g weight_max=%.6g",
        graph.audit["cp_edge_count"],
        graph.audit["cp_degree"]["mean"],
        graph.audit["cp_degree"]["median"],
        cp_unique_peaks,
        graph.audit["cp_edge_weight"]["min"],
        graph.audit["cp_edge_weight"]["mean"],
        graph.audit["cp_edge_weight"]["max"],
    )
    LOGGER.info(
        "CG graph: edges=%d mean_degree=%.3f median_degree=%.3f min_degree=%.0f "
        "max_degree=%.0f",
        graph.audit["cg_edge_count"],
        graph.audit["cg_degree"]["mean"],
        graph.audit["cg_degree"]["median"],
        graph.audit["cg_degree"]["min"],
        graph.audit["cg_degree"]["max"],
    )
    return graph


def build_state_heterodata_from_edges(
    multiome: MultiomeData,
    edge_tables: EdgeTableBundle,
    *,
    factors: MultiomeFactors,
    node_features: dict[str, torch.Tensor],
) -> StateGraphBuildResult:
    """Convert a standardized bundle into the Stage-1 training graph."""

    gene_names = multiome.genes["gene"].astype(str).tolist()
    peak_names = multiome.peaks["peak"].astype(str).tolist()
    cell_names = list(
        multiome.cell_names
        or [f"cell_{index}" for index in range(multiome.rna.shape[0])]
    )
    converted: HeterogeneousGraphBuildResult = edge_tables_to_heterodata(
        edge_tables,
        node_ids={"cell": cell_names, "gene": gene_names, "peak": peak_names},
        node_features=node_features,
        tf_ids=list(multiome.tf_names or []),
        prior_message=False,
    )
    return StateGraphBuildResult(
        data=converted.data,
        features=converted.features,
        factors=factors,
        names=converted.names,
        metadata={
            "genes": multiome.genes.copy(),
            "peaks": multiome.peaks.copy(),
            "tf_names": list(multiome.tf_names or []),
            "motif": multiome.motif,
            "cell_types": multiome.cell_types,
        },
        edge_table_bundle=edge_tables,
        query_tensors=converted.query_tensors,
        audit=converted.audit,
    )
