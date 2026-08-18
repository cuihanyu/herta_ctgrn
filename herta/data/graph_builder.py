"""Build weighted heterogeneous graphs for HERTA-ctGRN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy import sparse
from torch_geometric.data import HeteroData

from herta.data.dataset import MultiomeData
from herta.data.genomics import gene_peak_prior, tf_gene_prior
from herta.data.preprocessing import preprocess_atac, preprocess_rna
from herta.data.heterogeneous_graph import (
    HeterogeneousGraphBuildResult,
    build_heterogeneous_graph,
    edge_tables_to_heterodata,
)


FORWARD_RELATIONS: dict[str, tuple[str, str, str]] = {
    "ct": ("cell", "expresses_tf_activity", "tf"),
    "cg": ("cell", "expresses_gene", "gene"),
    "cp": ("cell", "accessible_peak", "peak"),
    "tp": ("tf", "binds_peak", "peak"),
    "pg": ("peak", "regulates_gene_prior", "gene"),
}


@dataclass
class GraphBuildResult:
    """Graph plus names and processed matrices used downstream."""

    data: HeteroData
    rna_norm: np.ndarray
    atac_tfidf: np.ndarray
    tf_gene_indices: np.ndarray
    names: dict[str, list[str]]
    metadata: dict[str, pd.DataFrame | np.ndarray]


def _add_edges(data: HeteroData, edge_type: tuple[str, str, str], src: np.ndarray, dst: np.ndarray, weight: np.ndarray) -> None:
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(weight, dtype=torch.float32).clamp(0, 1)
    data[edge_type].edge_index = edge_index
    data[edge_type].edge_weight = edge_weight
    rev = (edge_type[2], f"rev_{edge_type[1]}", edge_type[0])
    data[rev].edge_index = edge_index.flip(0)
    data[rev].edge_weight = edge_weight.clone()


def _minmax_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mins = x.min(axis=1, keepdims=True)
    maxs = x.max(axis=1, keepdims=True)
    return (x - mins) / (maxs - mins + eps)


def _matrix_to_numpy(x: object) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _row_topk_weights(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    if k <= 0:
        return np.array(rows), np.array(cols), np.array(vals, dtype=np.float32)
    for i in range(x.shape[0]):
        row = np.asarray(x[i])
        positive = np.flatnonzero(row > 0)
        if positive.size == 0:
            continue
        chosen = positive[np.argsort(row[positive])[-k:]]
        weights = row[chosen].astype(np.float32)
        denom = float(weights.sum()) + 1e-8
        for j, w in zip(chosen, weights / denom):
            rows.append(i)
            cols.append(int(j))
            vals.append(float(w))
    return np.asarray(rows), np.asarray(cols), np.asarray(vals, dtype=np.float32)


def build_heterodata(
    multiome: MultiomeData,
    top_cell_tf: int = 50,
    top_cell_gene: int = 200,
    top_cell_peak: int = 500,
    top_tf_peak: int = 2000,
    pg_window: int = 150000,
    pg_decay: float = 150000.0,
    max_peak_gene_edges_per_peak: int = 20,
    pg_weight_mode: str = "glue",
    peak_tf_links: pd.DataFrame | None = None,
    gene_peak_links: pd.DataFrame | None = None,
) -> GraphBuildResult:
    """Create a four-node-type graph with five weighted prior relations."""

    if pg_weight_mode not in {"glue", "exponential"}:
        raise ValueError("`pg_weight_mode` must be either 'glue' or 'exponential'.")

    rna_adata = AnnData(np.asarray(multiome.rna, dtype=np.float32))
    atac_adata = AnnData(np.asarray(multiome.atac, dtype=np.float32))
    preprocess_rna(rna_adata)
    preprocess_atac(atac_adata)
    rna_norm = _matrix_to_numpy(rna_adata.X)
    atac_tfidf = _matrix_to_numpy(atac_adata.X)
    gene_names = multiome.genes["gene"].astype(str).tolist()
    gene_index = {name: i for i, name in enumerate(gene_names)}
    tf_gene_indices = np.array([gene_index[t] for t in multiome.tf_names if t in gene_index], dtype=np.int64)
    tf_names = [gene_names[i] for i in tf_gene_indices]

    data = HeteroData()
    data["cell"].num_nodes = int(multiome.rna.shape[0])
    data["tf"].num_nodes = int(len(tf_names))
    data["gene"].num_nodes = int(multiome.rna.shape[1])
    data["peak"].num_nodes = int(multiome.atac.shape[1])

    if multiome.motif_activity is not None:
        activity = np.asarray(multiome.motif_activity, dtype=np.float32)[:, : len(tf_names)]
    else:
        activity = rna_norm[:, tf_gene_indices]
    activity_z = (activity - activity.mean(axis=0, keepdims=True)) / (activity.std(axis=0, keepdims=True) + 1e-8)
    activity_w = 1.0 / (1.0 + np.exp(-activity_z))
    _add_edges(data, FORWARD_RELATIONS["ct"], *_row_topk_weights(activity_w, top_cell_tf))
    _add_edges(data, FORWARD_RELATIONS["cg"], *_row_topk_weights(rna_norm, top_cell_gene))
    _add_edges(data, FORWARD_RELATIONS["cp"], *_row_topk_weights(atac_tfidf, top_cell_peak))

    peak_names = multiome.peaks["peak"].astype(str).tolist()
    peak_index = {name: i for i, name in enumerate(peak_names)}
    tf_index = {name: i for i, name in enumerate(tf_names)}
    if peak_tf_links is None:
        motif = np.asarray(multiome.motif, dtype=np.float32)
        if motif.shape == (len(peak_names), len(tf_names)):
            motif = motif.T
        elif motif.shape != (len(tf_names), len(peak_names)):
            raise ValueError("motif must align to the authoritative peak x TF matrix.")
        motif = _minmax_rows(motif)
        tp_src, tp_dst, tp_w = _row_topk_weights(motif, top_tf_peak)
        keep = tp_w > 0
        tp_src = tp_src[keep]
        tp_dst = tp_dst[keep]
        tp_w = tp_w[keep]
        peak_tf_prior = pd.DataFrame(
            {
                "tf": [tf_names[i] for i in tp_src],
                "peak": [peak_names[i] for i in tp_dst],
                "weight": tp_w,
            }
        )
    else:
        required = {"tf", "peak"}
        if missing := required.difference(peak_tf_links.columns):
            raise ValueError(f"Peak-TF prior is missing required columns: {sorted(missing)}")
        peak_tf_prior = peak_tf_links.copy()
        if "weight" not in peak_tf_prior:
            peak_tf_prior["weight"] = 1.0
        peak_tf_prior["weight"] = pd.to_numeric(peak_tf_prior["weight"])
        if not np.isfinite(peak_tf_prior["weight"]).all() or not peak_tf_prior["weight"].between(0, 1).all():
            raise ValueError("Peak-TF prior weights must be finite values between 0 and 1.")
        peak_tf_prior["tf"] = peak_tf_prior["tf"].astype(str)
        peak_tf_prior["peak"] = peak_tf_prior["peak"].astype(str)
        peak_tf_prior = peak_tf_prior.loc[
            peak_tf_prior["tf"].isin(tf_index) & peak_tf_prior["peak"].isin(peak_index)
        ]
        peak_tf_prior = (
            peak_tf_prior.sort_values("weight", ascending=False)
            .groupby("tf", sort=False, group_keys=False)
            .head(top_tf_peak)
            .reset_index(drop=True)
        )
        tp_src = peak_tf_prior["tf"].map(tf_index).to_numpy(dtype=np.int64)
        tp_dst = peak_tf_prior["peak"].map(peak_index).to_numpy(dtype=np.int64)
        tp_w = peak_tf_prior["weight"].to_numpy(dtype=np.float32)
    _add_edges(data, FORWARD_RELATIONS["tp"], tp_src, tp_dst, tp_w)

    if gene_peak_links is None:
        use_gene_regions = {"chromStart", "chromEnd", "strand"}.issubset(multiome.genes.columns) and multiome.genes[
            "strand"
        ].isin({"+", "-"}).all()
        gene_peak = gene_peak_prior(
            multiome.genes,
            multiome.peaks,
            gene_region="combined" if use_gene_regions else "gene_body",
            extend_range=pg_window,
            max_genes_per_peak=max_peak_gene_edges_per_peak,
        )
    else:
        required = {"gene", "peak"}
        if missing := required.difference(gene_peak_links.columns):
            raise ValueError(f"Gene-peak prior is missing required columns: {sorted(missing)}")
        gene_peak = gene_peak_links.copy()
        if "weight" not in gene_peak:
            gene_peak["weight"] = 1.0
        gene_peak["weight"] = pd.to_numeric(gene_peak["weight"])
        if not np.isfinite(gene_peak["weight"]).all() or not gene_peak["weight"].between(0, 1).all():
            raise ValueError("Gene-peak prior weights must be finite values between 0 and 1.")
        if "dist" not in gene_peak:
            gene_peak["dist"] = 0
        if "sign" not in gene_peak:
            gene_peak["sign"] = 1
        gene_peak["gene"] = gene_peak["gene"].astype(str)
        gene_peak["peak"] = gene_peak["peak"].astype(str)
        gene_peak = gene_peak.loc[gene_peak["gene"].isin(gene_index) & gene_peak["peak"].isin(peak_index)]
        gene_peak = (
            gene_peak.sort_values(["peak", "weight"], ascending=[True, False])
            .groupby("peak", sort=False, group_keys=False)
            .head(max_peak_gene_edges_per_peak)
            .reset_index(drop=True)
        )
    if pg_weight_mode == "exponential" and not gene_peak.empty:
        gene_peak["weight"] = np.exp(-gene_peak["dist"].to_numpy(dtype=float) / pg_decay)
    pg_src = np.asarray([peak_index[str(name)] for name in gene_peak["peak"]], dtype=np.int64)
    pg_dst = np.asarray([gene_index[str(name)] for name in gene_peak["gene"]], dtype=np.int64)
    pg_w = gene_peak["weight"].to_numpy(dtype=np.float32)
    _add_edges(data, FORWARD_RELATIONS["pg"], pg_src, pg_dst, pg_w)
    tf_gene = tf_gene_prior(peak_tf_prior, gene_peak)

    data.validate(raise_on_error=True)
    return GraphBuildResult(
        data=data,
        rna_norm=rna_norm,
        atac_tfidf=atac_tfidf,
        tf_gene_indices=tf_gene_indices,
        names={"cell": [f"cell_{i}" for i in range(multiome.rna.shape[0])], "tf": tf_names, "gene": gene_names, "peak": multiome.peaks["peak"].astype(str).tolist()},
        metadata={
            "genes": multiome.genes,
            "peaks": multiome.peaks,
            "cell_types": multiome.cell_types,
            "peak_tf_prior": peak_tf_prior,
            "gene_peak_prior": gene_peak,
            "tf_gene_prior": tf_gene,
        },
    )
