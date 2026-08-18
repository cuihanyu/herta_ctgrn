"""Stage-2 regulatory graph construction and edge splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch_geometric.data import HeteroData

from herta.data.edge_tables import (
    EdgeTableBundle,
    build_peak_gene_edges,
    build_tf_peak_edges,
)
from herta.data.heterogeneous_graph import QueryEdgeTensors, edge_tables_to_heterodata
from herta.data.state_graph import STATE_RELATIONS, StateGraphBuildResult


REGULATORY_RELATIONS = {
    **STATE_RELATIONS,
    "tp": ("gene", "binds", "peak"),
    "pg": ("peak", "regulates", "gene"),
}


@dataclass
class EdgeSplit:
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    edge_features: torch.Tensor | None = None


@dataclass
class RegulatoryGraphBuildResult:
    data: HeteroData
    state: StateGraphBuildResult
    tf_gene_indices: torch.Tensor
    is_tf: torch.Tensor
    splits: dict[str, dict[str, EdgeSplit]]
    candidates: dict[str, pd.DataFrame]
    edge_table_bundle: EdgeTableBundle
    query_tensors: dict[str, QueryEdgeTensors]
    audit: dict[str, object]


def tf_gene_split_table(
    build: RegulatoryGraphBuildResult,
    split: str = "test",
) -> pd.DataFrame:
    """Return one TF-target supervision split with node names restored."""

    if "tg" not in build.splits:
        raise ValueError("No TF-target supervision was supplied to the regulatory graph.")
    if split not in build.splits["tg"]:
        raise ValueError(f"Unknown TF-target split: {split}")
    edge_index = build.splits["tg"][split].edge_index
    gene_names = build.state.names["gene"]
    return pd.DataFrame(
        {
            "tf": [gene_names[index] for index in edge_index[0].tolist()],
            "target": [gene_names[index] for index in edge_index[1].tolist()],
        }
    )


def _split_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, EdgeSplit]:
    n_edges = edge_index.shape[1]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_edges, generator=generator)
    n_test = int(n_edges * test_fraction)
    n_val = int(n_edges * validation_fraction)
    if n_edges - n_test - n_val < 1:
        n_test = 0
        n_val = 0
    indices = {
        "test": order[:n_test],
        "validation": order[n_test : n_test + n_val],
        "train": order[n_test + n_val :],
    }
    return {
        name: EdgeSplit(edge_index[:, idx], edge_weight[idx])
        for name, idx in indices.items()
    }


def _set_train_relation(data: HeteroData, edge_type: tuple[str, str, str], split: EdgeSplit) -> None:
    data[edge_type].edge_index = split.edge_index
    data[edge_type].edge_weight = split.edge_weight
    reverse = (edge_type[2], f"rev_{edge_type[1]}", edge_type[0])
    data[reverse].edge_index = split.edge_index.flip(0)
    data[reverse].edge_weight = split.edge_weight.clone()


def _assign_candidate_splits(
    table: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Assign deterministic splits by peak rather than by candidate edge."""

    result = table.copy()
    if result.empty:
        return result
    current = result["split"].astype(str)
    assign = current.isin({"query", "", "nan", "<NA>"})
    if not bool(assign.any()):
        invalid = set(current).difference({"train", "validation", "test", "message"})
        if invalid:
            raise ValueError(f"Unknown edge split values: {sorted(invalid)}")
        return result
    if set(result["relation"].astype(str)) == {"binds"}:
        peak_column = "target_id"
    elif set(result["relation"].astype(str)) == {"regulates"}:
        peak_column = "source_id"
    else:
        raise ValueError("Candidate split assignment requires one TP or PG relation.")
    peaks = pd.Index(result.loc[assign, peak_column].astype(str).unique())
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(peaks))
    n_test = int(round(len(order) * test_fraction))
    n_validation = int(round(len(order) * validation_fraction))
    if len(order) - n_test - n_validation < 1:
        n_test = 0
        n_validation = 0
    peak_split = {
        **{peaks[index]: "test" for index in order[:n_test]},
        **{
            peaks[index]: "validation"
            for index in order[n_test : n_test + n_validation]
        },
        **{peaks[index]: "train" for index in order[n_test + n_validation :]},
    }
    result.loc[assign, "split"] = result.loc[assign, peak_column].astype(str).map(peak_split)
    return result


def _assign_shared_peak_splits(
    tf_peak: pd.DataFrame,
    peak_gene: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put every TP/PG candidate involving the same peak in one split."""

    tp, pg = tf_peak.copy(), peak_gene.copy()
    all_peaks = pd.Index(
        pd.concat(
            [tp["target_id"].astype(str), pg["source_id"].astype(str)],
            ignore_index=True,
        ).unique()
    )
    order = np.random.default_rng(seed).permutation(len(all_peaks))
    n_test = int(round(len(order) * test_fraction))
    n_validation = int(round(len(order) * validation_fraction))
    if len(order) - n_test - n_validation < 1:
        n_test = 0
        n_validation = 0
    peak_split = {
        **{all_peaks[index]: "test" for index in order[:n_test]},
        **{
            all_peaks[index]: "validation"
            for index in order[n_test : n_test + n_validation]
        },
        **{all_peaks[index]: "train" for index in order[n_test + n_validation :]},
    }
    for table, peak_column in ((tp, "target_id"), (pg, "source_id")):
        table["split"] = table[peak_column].astype(str).map(peak_split)
    cross = pd.concat(
        [
            tp[["target_id", "split"]].rename(columns={"target_id": "peak"}),
            pg[["source_id", "split"]].rename(columns={"source_id": "peak"}),
        ],
        ignore_index=True,
    )
    if bool(cross.groupby("peak")["split"].nunique().gt(1).any()):
        raise RuntimeError("Peak-grouped candidate split assignment failed.")
    return tp, pg


def _compatibility_candidates(bundle: EdgeTableBundle) -> dict[str, pd.DataFrame]:
    tp = bundle.tf_peak.copy().rename(
        columns={"source_id": "tf", "target_id": "peak", "prior_score": "weight"}
    )
    pg = bundle.peak_gene.copy().rename(
        columns={
            "source_id": "peak",
            "target_id": "gene",
            "prior_score": "weight",
            "distance": "dist",
        }
    )
    result = {"tp": tp, "pg": pg}
    if bundle.tf_target is not None:
        result["tg"] = bundle.tf_target.copy().rename(
            columns={"source_id": "tf", "target_id": "target"}
        )
    return result


def build_regulatory_heterodata_from_edges(
    state: StateGraphBuildResult,
    edge_tables: EdgeTableBundle,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 1,
) -> RegulatoryGraphBuildResult:
    """Build query-only Stage 2 tables over the frozen observed graph."""

    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("validation_fraction and test_fraction must be non-negative.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than one.")
    tables = edge_tables.as_dict()
    split_tp, split_pg = _assign_shared_peak_splits(
        tables["tf_peak"],
        tables["peak_gene"],
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    if "tf_target" in tables:
        raise ValueError(
            "TF-target gold-standard tables are outside shared_backbone_minimal_v1."
        )
    split_bundle = EdgeTableBundle(
        cell_gene=tables["cell_gene"],
        cell_peak=tables["cell_peak"],
        peak_gene=split_pg,
        tf_peak=split_tp,
        tf_target=None,
    )
    marked_tfs = {
        state.names["gene"][index]
        for index in torch.nonzero(
            state.data["gene"].is_tf, as_tuple=False
        ).flatten().tolist()
    }
    marked_tfs.update(split_tp["source_id"].astype(str))
    converted = edge_tables_to_heterodata(
        split_bundle,
        node_ids=state.names,
        node_features=state.features,
        tf_ids=[name for name in state.names["gene"] if name in marked_tfs],
        prior_message=False,
    )
    splits: dict[str, dict[str, EdgeSplit]] = {}
    relation_tables = {"tp": split_tp, "pg": split_pg}
    for key, table in relation_tables.items():
        query = converted.query_tensors[key]
        splits[key] = {}
        for split_name in ("train", "validation", "test"):
            mask = torch.as_tensor(
                table["split"].astype(str).eq(split_name).to_numpy(), dtype=torch.bool
            )
            splits[key][split_name] = EdgeSplit(
                query.edge_index[:, mask],
                query.prior_score[mask],
                query.edge_features[mask],
            )
    audit = dict(converted.audit)
    audit["prior_message"] = False
    audit["candidate_split_unit"] = "peak"
    audit["candidate_split_shared_across_tp_pg"] = True
    audit["split_edge_counts"] = {
        key: {
            split_name: int(value.edge_index.shape[1])
            for split_name, value in relation_splits.items()
        }
        for key, relation_splits in splits.items()
    }
    return RegulatoryGraphBuildResult(
        data=converted.data,
        state=state,
        tf_gene_indices=converted.tf_gene_indices,
        is_tf=converted.is_tf,
        splits=splits,
        candidates=_compatibility_candidates(split_bundle),
        edge_table_bundle=split_bundle,
        query_tensors=converted.query_tensors,
        audit=audit,
    )


def _motif_candidates(
    state: StateGraphBuildResult,
    tf_names: list[str],
    top_tf_peak: int | None,
) -> pd.DataFrame:
    motif = state.metadata.get("motif")
    if motif is None:
        raise ValueError("A motif matrix or peak_tf_links table is required for Stage 2.")
    matrix = sparse.csr_matrix(motif)
    peak_names = state.names["peak"]
    if matrix.shape != (len(peak_names), len(tf_names)):
        raise ValueError(
            "The authoritative motif artifact must have shape peak x TF; "
            f"expected {(len(peak_names), len(tf_names))}, observed {matrix.shape}."
        )
    rows: list[dict[str, object]] = []
    for tf_id, tf in enumerate(tf_names):
        column = matrix.getcol(tf_id).tocoo()
        peak_ids = column.row[column.data > 0]
        if top_tf_peak is not None and len(peak_ids) > top_tf_peak:
            peak_ids = np.sort(peak_ids)[:top_tf_peak]
        rows.extend(
            {"tf": tf, "peak": peak_names[p], "weight": 1.0}
            for p in peak_ids
        )
    return pd.DataFrame(rows, columns=["tf", "peak", "weight"])


def _tss_peak_candidates(
    genes: pd.DataFrame,
    peaks: pd.DataFrame,
    *,
    window: int,
    decay: float,
) -> pd.DataFrame:
    """Construct peak-gene candidates from interval-to-strand-aware-TSS distance."""

    required_gene = {"gene", "chrom", "chromStart", "chromEnd", "strand"}
    required_peak = {"peak", "chrom", "chromStart", "chromEnd"}
    if missing := required_gene.difference(genes.columns):
        raise ValueError(f"Gene annotation is missing TSS columns: {sorted(missing)}")
    if missing := required_peak.difference(peaks.columns):
        raise ValueError(f"Peak annotation is missing coordinate columns: {sorted(missing)}")
    if window < 0 or decay <= 0:
        raise ValueError("PG window must be non-negative and decay must be positive.")
    gene_frame = genes.dropna(subset=list(required_gene)).copy()
    peak_frame = peaks.dropna(subset=list(required_peak)).copy()
    gene_frame["tss"] = np.where(
        gene_frame["strand"].astype(str).eq("+"),
        pd.to_numeric(gene_frame["chromStart"]),
        pd.to_numeric(gene_frame["chromEnd"]) - 1,
    ).astype(np.int64)
    rows: list[pd.DataFrame] = []
    for chrom, chrom_genes in gene_frame.groupby("chrom", sort=False):
        chrom_peaks = peak_frame.loc[peak_frame["chrom"].astype(str).eq(str(chrom))]
        if chrom_peaks.empty:
            continue
        starts = pd.to_numeric(chrom_peaks["chromStart"]).to_numpy(dtype=np.int64)
        ends = pd.to_numeric(chrom_peaks["chromEnd"]).to_numpy(dtype=np.int64)
        for gene in chrom_genes.itertuples(index=False):
            tss = int(gene.tss)
            distance = np.where(
                tss < starts,
                starts - tss,
                np.where(tss >= ends, tss - (ends - 1), 0),
            )
            keep = distance <= window
            if not bool(keep.any()):
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "peak": chrom_peaks.loc[keep, "peak"].astype(str).to_numpy(),
                        "gene": str(gene.gene),
                        "dist": distance[keep].astype(float),
                        "weight": np.exp(-distance[keep] / decay),
                        "chrom": str(chrom),
                    }
                )
            )
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["peak", "gene", "dist", "weight", "chrom"])
    )


def _state_tf_universe(
    state: StateGraphBuildResult,
    gene_index: dict[str, int],
) -> list[str]:
    """Return the authoritative Stage-2 TF universe from Stage 1 metadata.

    TF-peak candidate tables are intentionally not used to define this set:
    a motif-supported RNA TF can have no retained peak candidate and must still
    remain marked as a TF for typed negatives, diagnostics, and later evidence
    extensions.
    """

    metadata_tfs = list(
        dict.fromkeys(map(str, state.metadata.get("tf_names", [])))
    )
    marked_tfs = [
        name
        for name, marked in zip(
            state.names["gene"],
            state.data["gene"].is_tf.detach().cpu().bool().tolist(),
        )
        if marked
    ]
    if not metadata_tfs:
        return marked_tfs
    missing_genes = [name for name in metadata_tfs if name not in gene_index]
    if missing_genes:
        raise ValueError(
            "Stage-1 TF metadata contains genes outside the Stage-2 gene universe: "
            f"{missing_genes[:5]}"
        )
    if set(metadata_tfs) != set(marked_tfs):
        raise ValueError(
            "Stage-1 TF metadata and gene is_tf mask disagree; rebuild the state graph."
        )
    return metadata_tfs


def build_regulatory_graph(
    state: StateGraphBuildResult,
    peak_tf_links: pd.DataFrame | None = None,
    gene_peak_links: pd.DataFrame | None = None,
    tf_gene_links: pd.DataFrame | None = None,
    top_tf_peak: int | None = None,
    pg_window: int = 250000,
    pg_decay: float = 25000.0,
    pg_weight_mode: str = "exponential",
    max_peak_gene_edges_per_peak: int | None = None,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 1,
) -> RegulatoryGraphBuildResult:
    """Build binary TP and TSS-distance PG queries over the observed graph."""

    if pg_weight_mode != "exponential":
        raise ValueError("The minimal workflow requires exponential PG distance weights.")
    if tf_gene_links is not None:
        raise ValueError("TF-target gold standards are not read by the minimal workflow.")
    gene_index = {name: idx for idx, name in enumerate(state.names["gene"])}
    peak_index = {name: idx for idx, name in enumerate(state.names["peak"])}
    tf_names = _state_tf_universe(state, gene_index)
    if peak_tf_links is not None:
        if "tf" not in peak_tf_links:
            raise ValueError("TF-peak candidates are missing column: tf")
        unexpected_tfs = sorted(
            set(peak_tf_links["tf"].astype(str)).difference(tf_names)
        )
        if unexpected_tfs:
            raise ValueError(
                "TF-peak candidates contain TFs outside the authoritative Stage-2 "
                f"TF universe: {unexpected_tfs[:5]}"
            )
    tf_gene_indices = torch.tensor([gene_index[name] for name in tf_names], dtype=torch.long)
    is_tf = torch.zeros(len(gene_index), dtype=torch.bool)
    is_tf[tf_gene_indices] = True

    tp = _motif_candidates(state, tf_names, top_tf_peak) if peak_tf_links is None else peak_tf_links.copy()
    required_tp = {"tf", "peak"}
    if missing := required_tp.difference(tp.columns):
        raise ValueError(f"TF-peak candidates are missing columns: {sorted(missing)}")
    tp = tp[tp["tf"].astype(str).isin(gene_index) & tp["peak"].astype(str).isin(peak_index)].copy()
    if "weight" not in tp:
        tp["weight"] = 1.0
    tp["weight"] = pd.to_numeric(tp["weight"], errors="coerce")
    tp = tp[np.isfinite(tp["weight"]) & (tp["weight"] > 0)].copy()
    tp = tp.sort_values("weight", ascending=False).drop_duplicates(["tf", "peak"])
    tp["weight"] = 1.0
    if top_tf_peak is not None:
        tp = tp.groupby("tf", sort=False, group_keys=False).head(top_tf_peak)
    tp = tp.reset_index(drop=True)

    if gene_peak_links is None:
        pg = _tss_peak_candidates(
            state.metadata["genes"],
            state.metadata["peaks"],
            window=pg_window,
            decay=pg_decay,
        )
    else:
        pg = gene_peak_links.copy()
    required_pg = {"gene", "peak"}
    if missing := required_pg.difference(pg.columns):
        raise ValueError(f"Peak-gene candidates are missing columns: {sorted(missing)}")
    pg = pg[pg["gene"].astype(str).isin(gene_index) & pg["peak"].astype(str).isin(peak_index)].copy()
    if "dist" in pg:
        distance = np.abs(pd.to_numeric(pg["dist"]).to_numpy(dtype=float))
        pg = pg.loc[distance <= pg_window].copy()
        pg["weight"] = np.exp(-distance[distance <= pg_window] / pg_decay)
    elif "weight" not in pg:
        pg["weight"] = 1.0
    pg["weight"] = pd.to_numeric(pg["weight"], errors="coerce")
    pg = pg[np.isfinite(pg["weight"]) & (pg["weight"] > 0)].copy()
    pg = pg.sort_values("weight", ascending=False).drop_duplicates(["peak", "gene"])
    if max_peak_gene_edges_per_peak is not None:
        pg = pg.sort_values(["peak", "weight"], ascending=[True, False]).groupby(
            "peak", sort=False, group_keys=False
        ).head(max_peak_gene_edges_per_peak)
    pg = pg.reset_index(drop=True)
    if "dist" not in pg and "distance" not in pg:
        pg["dist"] = 0.0

    if tp.empty or pg.empty:
        empty = "TF-peak" if tp.empty else "peak-gene"
        raise ValueError(f"No valid {empty} candidates remain after filtering.")

    genome_build = "unspecified"
    for candidate in (tp, pg, state.metadata.get("genes"), state.metadata.get("peaks")):
        if isinstance(candidate, pd.DataFrame) and "genome_build" in candidate:
            values = candidate["genome_build"].dropna().astype(str)
            if not values.empty:
                genome_build = values.iloc[0]
                break
    standardized_tp = build_tf_peak_edges(
        tp,
        tf_genes=tf_names,
        genome_build=genome_build,
    )
    standardized_pg = build_peak_gene_edges(pg, genome_build=genome_build)
    bundle = EdgeTableBundle(
        cell_gene=state.edge_table_bundle.cell_gene,
        cell_peak=state.edge_table_bundle.cell_peak,
        peak_gene=standardized_pg,
        tf_peak=standardized_tp,
        tf_target=None,
    )
    result = build_regulatory_heterodata_from_edges(
        state,
        bundle,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    candidate_tf_names = list(dict.fromkeys(tp["tf"].astype(str)))
    candidate_tf_set = set(candidate_tf_names)
    result.audit["tf_universe"] = {
        "definition": "stage1_rna_gene_intersection_motif_supported_tf",
        "n_tf_universe": len(tf_names),
        "n_tf_with_peak_candidates": len(candidate_tf_names),
        "n_tf_without_peak_candidates": len(tf_names) - len(candidate_tf_names),
        "tf_without_peak_candidates": [
            name for name in tf_names if name not in candidate_tf_set
        ],
        "top_tf_peak_is_per_tf": top_tf_peak is not None,
        "top_tf_peak_per_tf": None if top_tf_peak is None else int(top_tf_peak),
    }
    if int(result.tf_gene_indices.numel()) != len(tf_names):
        raise RuntimeError(
            "Stage-2 graph silently changed the authoritative TF universe: "
            f"expected {len(tf_names)}, observed {int(result.tf_gene_indices.numel())}."
        )
    return result
