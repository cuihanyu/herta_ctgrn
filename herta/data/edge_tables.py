"""Inspectable edge-table contracts for the HERTA three-node graph schema.

This module stops at pandas tables. Conversion to PyG tensors and use as
message-passing edges belong to the graph-builder stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from herta.data.genomics import dist_power_decay


NODE_TYPES = ("cell", "gene", "peak")
TF_REPRESENTATION = "gene_subset"

RELATIONS: Mapping[str, tuple[str, str, str]] = {
    "cell_gene": ("cell", "expresses", "gene"),
    "cell_peak": ("cell", "accessible", "peak"),
    "peak_gene": ("peak", "regulates", "gene"),
    "tf_peak": ("gene", "binds", "peak"),
    "tf_target": ("gene", "gold_regulates", "gene"),
}

MESSAGE_RELATIONS = ("cell_gene", "cell_peak")
QUERY_RELATIONS = ("peak_gene", "tf_peak", "tf_target")

STATE_EDGE_TABLE_COLUMNS = [
    "source_id",
    "target_id",
    "source_type",
    "target_type",
    "relation",
    "raw_value",
    "edge_weight",
    "selection_method",
]

RELATION_EXTRA_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "cell_gene": (
        "edge_weight",
        "raw_value",
        "normalized_value",
        "normalized_weight",
        "selection_method",
        "expression_threshold",
        "percentile",
    ),
    "cell_peak": (
        "edge_weight",
        "raw_value",
        "normalized_value",
        "normalized_weight",
        "selection_method",
    ),
    "peak_gene": ("n_evidence",),
    "tf_peak": (
        "n_evidence",
        "motif_score",
        "pvalue",
        "overlap_bp",
        "source",
        "source_version",
    ),
    "tf_target": (
        "supervision_role",
        "mor",
        "mor_conflict",
        "confidence",
        "source",
        "evidence",
    ),
}

EDGE_TABLE_COLUMNS = [
    "source_id",
    "target_id",
    "source_type",
    "target_type",
    "relation",
    "prior_score",
    "evidence_type",
    "evidence_id",
    "distance",
    "chrom",
    "genome_build",
    "label_status",
    "split",
]

LABEL_STATUSES = {
    "observed",
    "candidate",
    "weak_positive",
    "gold_positive",
    "unknown",
    "negative",
}

GOLD_ROLES = {
    "training_weak_supervision": "train",
    "validation_model_selection": "validation",
    "held_out_final_evaluation": "test",
}


def _ids(values: pd.Index | list[str], name: str) -> pd.Index:
    result = pd.Index(values, dtype="object").astype(str)
    if result.empty:
        raise ValueError(f"{name} must not be empty.")
    if result.has_duplicates:
        raise ValueError(f"{name} must contain unique identifiers.")
    if (result.str.strip() == "").any():
        raise ValueError(f"{name} must not contain empty identifiers.")
    return result


def _matrix(value: object, shape: tuple[int, int], name: str) -> sparse.csr_matrix:
    result = sparse.csr_matrix(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} shape {result.shape} does not match identifiers {shape}.")
    result.sum_duplicates()
    result.eliminate_zeros()
    if result.data.size and (
        not np.isfinite(result.data).all() or np.any(result.data < 0)
    ):
        raise ValueError(f"{name} must contain finite, non-negative values.")
    return result


def _observed_matrix(
    value: object, shape: tuple[int, int], name: str
) -> sparse.csr_matrix:
    """Return positive finite observed values, filtering invalid sparse entries."""

    result = sparse.csr_matrix(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} shape {result.shape} does not match identifiers {shape}.")
    result.sum_duplicates()
    if result.data.size and np.any(result.data[np.isfinite(result.data)] < 0):
        raise ValueError(f"{name} must not contain negative values.")
    if result.data.size:
        result.data[~np.isfinite(result.data) | (result.data <= 0)] = 0.0
    result.eliminate_zeros()
    return result


def _join_unique(values: pd.Series) -> str:
    unique = sorted(
        {
            str(value).strip()
            for value in values
            if pd.notna(value) and str(value).strip() not in {"", "nan", "None"}
        }
    )
    return ";".join(unique)


def _nullable_string(value: object, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.astype("string")
    return pd.Series(value, index=index, dtype="string")


def _finalize_edge_table(
    table: pd.DataFrame,
    *,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    result = table.copy()
    for column in EDGE_TABLE_COLUMNS:
        if column not in result:
            result[column] = pd.NA
    for column in extra_columns or []:
        if column not in result:
            result[column] = pd.NA
    string_columns = [
        "source_id",
        "target_id",
        "source_type",
        "target_type",
        "relation",
        "evidence_type",
        "evidence_id",
        "chrom",
        "genome_build",
        "label_status",
        "split",
    ]
    for column in string_columns:
        result[column] = _nullable_string(result[column], result.index)
    result["prior_score"] = pd.to_numeric(
        result["prior_score"], errors="coerce"
    ).astype(float)
    result["distance"] = pd.to_numeric(
        result["distance"], errors="coerce"
    ).astype("Float64")
    ordered = EDGE_TABLE_COLUMNS + [
        column for column in (extra_columns or []) if column not in EDGE_TABLE_COLUMNS
    ]
    result = result.loc[:, ordered].reset_index(drop=True)
    validate_edge_table(result)
    return result


def validate_edge_table(
    table: pd.DataFrame,
    *,
    expected_relation: str | None = None,
) -> None:
    """Validate the shared HERTA edge-table contract."""

    if not isinstance(table, pd.DataFrame):
        raise TypeError("Every HERTA edge table must be a pandas DataFrame.")
    missing = set(EDGE_TABLE_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(f"Edge table is missing required columns: {sorted(missing)}")
    if table.empty:
        return
    for column in (
        "source_id",
        "target_id",
        "source_type",
        "target_type",
        "relation",
        "evidence_type",
        "evidence_id",
        "label_status",
        "split",
    ):
        values = table[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"Edge column '{column}' must contain non-empty values.")
    observed_schema = set(
        table[["source_type", "relation", "target_type"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    known_schema = set(RELATIONS.values())
    if unknown_schema := observed_schema.difference(known_schema):
        raise ValueError(f"Edge table contains unknown relation schemas: {unknown_schema}")
    scores = pd.to_numeric(table["prior_score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores).all():
        raise ValueError("prior_score must contain finite values.")
    if not scores.between(0, 1).all():
        raise ValueError("prior_score must be between 0 and 1.")
    if "edge_weight" in table and set(table["label_status"].astype(str)) == {"observed"}:
        edge_weights = pd.to_numeric(table["edge_weight"], errors="coerce")
        if (
            edge_weights.isna().any()
            or not np.isfinite(edge_weights).all()
            or bool((edge_weights <= 0).any())
        ):
            raise ValueError("Observed edge_weight must contain finite positive values.")
    distance = pd.to_numeric(table["distance"], errors="coerce")
    if (distance.dropna() < 0).any():
        raise ValueError("distance must be non-negative when present.")
    invalid_status = set(table["label_status"].dropna().astype(str)).difference(
        LABEL_STATUSES
    )
    if invalid_status:
        raise ValueError(f"Unknown label_status values: {sorted(invalid_status)}")
    if table.duplicated(["source_id", "target_id", "relation"]).any():
        raise ValueError("Edge table contains duplicate biological edges.")
    if expected_relation is not None:
        if expected_relation not in RELATIONS:
            raise ValueError(f"Unknown expected relation: {expected_relation}")
        source_type, relation, target_type = RELATIONS[expected_relation]
        expected = {(source_type, relation, target_type)}
        if observed_schema != expected:
            raise ValueError(
                f"{expected_relation} table has endpoint schema {observed_schema}, expected {expected}."
            )


def validate_state_edge_table(
    table: pd.DataFrame,
    *,
    expected_relation: str,
) -> None:
    """Validate the compact Stage-1 observed-edge contract."""

    if expected_relation not in MESSAGE_RELATIONS:
        raise ValueError("State edge tables support only cell_gene and cell_peak.")
    if list(table.columns) != STATE_EDGE_TABLE_COLUMNS:
        raise ValueError(
            "Stage-1 edge columns must exactly match "
            f"{STATE_EDGE_TABLE_COLUMNS}; observed {list(table.columns)}."
        )
    if table.empty:
        return
    source_type, relation, target_type = RELATIONS[expected_relation]
    observed = set(
        table[["source_type", "relation", "target_type"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    if observed != {(source_type, relation, target_type)}:
        raise ValueError(
            f"Stage-1 relation mismatch for {expected_relation}: {sorted(observed)}."
        )
    for column in (
        "source_id",
        "target_id",
        "source_type",
        "target_type",
        "relation",
        "selection_method",
    ):
        values = table[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"Stage-1 edge column '{column}' must be non-empty.")
    for column in ("raw_value", "edge_weight"):
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(
                f"Stage-1 edge column '{column}' must be finite and positive."
            )
    if table.duplicated(["source_id", "target_id"]).any():
        raise ValueError("Stage-1 observed edges must have unique endpoints.")


def _finalize_state_edge_table(
    rows: list[dict[str, object]],
    *,
    relation_key: str,
) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=STATE_EDGE_TABLE_COLUMNS)
    for column in (
        "source_id",
        "target_id",
        "source_type",
        "target_type",
        "relation",
        "selection_method",
    ):
        result[column] = result[column].astype("string")
    for column in ("raw_value", "edge_weight"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    validate_state_edge_table(result, expected_relation=relation_key)
    return result.reset_index(drop=True)


def empty_edge_table(relation_key: str) -> pd.DataFrame:
    """Return an empty table that still exposes the complete relation schema."""

    if relation_key not in RELATIONS:
        raise ValueError(f"Unknown relation key: {relation_key}")
    return _finalize_edge_table(
        pd.DataFrame(),
        extra_columns=list(RELATION_EXTRA_COLUMNS.get(relation_key, ())),
    )


def _build_observed_edges(
    values: object,
    source_ids: pd.Index | list[str],
    target_ids: pd.Index | list[str],
    *,
    relation_key: str,
    raw_values: object | None,
    top_k: int | None,
) -> pd.DataFrame:
    if relation_key == "cell_gene":
        return _build_cell_gene_union_edges(
            values,
            source_ids,
            target_ids,
            raw_values=raw_values,
            top_k=top_k,
            gene_cell_quantile=0.95,
        )
    sources = _ids(source_ids, "source_ids")
    targets = _ids(target_ids, "target_ids")
    matrix = _observed_matrix(values, (len(sources), len(targets)), "values")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when specified.")

    source_type, relation, target_type = RELATIONS[relation_key]
    rows: list[dict[str, object]] = []
    for source_index in range(matrix.shape[0]):
        row = matrix.getrow(source_index)
        positive = row.data > 0
        indices = row.indices[positive]
        selected = row.data[positive]
        if top_k is not None and len(selected) > top_k:
            order = np.lexsort((indices, -selected))[:top_k]
            indices, selected = indices[order], selected[order]
        denominator = float(selected.sum())
        normalized = (
            selected / denominator if denominator > 0 else np.zeros_like(selected)
        )
        for target_index, value, weight in zip(indices, selected, normalized):
            rows.append(
                {
                    "source_id": sources[source_index],
                    "target_id": targets[target_index],
                    "source_type": source_type,
                    "target_type": target_type,
                    "relation": relation,
                    "raw_value": float(value),
                    "edge_weight": float(weight),
                    "selection_method": (
                        "all_positive" if top_k is None else f"row_top_{top_k}"
                    ),
                }
            )
    return _finalize_state_edge_table(
        rows,
        relation_key=relation_key,
    )


def _build_cell_gene_union_edges(
    values: object,
    source_ids: pd.Index | list[str],
    target_ids: pd.Index | list[str],
    *,
    raw_values: object | None,
    top_k: int | None,
    gene_cell_quantile: float | None,
) -> pd.DataFrame:
    """Build the union of row Top-K and strict per-gene quantile CG edges."""

    sources = _ids(source_ids, "source_ids")
    targets = _ids(target_ids, "target_ids")
    matrix = _observed_matrix(values, (len(sources), len(targets)), "values")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when specified.")
    if gene_cell_quantile is not None and not 0 <= gene_cell_quantile <= 1:
        raise ValueError("gene_cell_quantile must be between zero and one.")
    if top_k is None and gene_cell_quantile is None:
        raise ValueError("CG construction requires top_k or gene_cell_quantile.")

    percentile = None if gene_cell_quantile is None else 100.0 * gene_cell_quantile
    csc = matrix.tocsc()
    thresholds = np.full(csc.shape[1], np.inf, dtype=np.float64)
    if percentile is not None:
        rank = (csc.shape[0] - 1) * gene_cell_quantile
        lower_rank, upper_rank = int(np.floor(rank)), int(np.ceil(rank))
        fraction = rank - lower_rank
        for column in range(csc.shape[1]):
            column_values = np.sort(
                csc.data[csc.indptr[column] : csc.indptr[column + 1]]
            )
            n_zeros = csc.shape[0] - column_values.size

            def value_at(position: int) -> float:
                return 0.0 if position < n_zeros else float(
                    column_values[position - n_zeros]
                )

            lower = value_at(lower_rank)
            upper = value_at(upper_rank)
            thresholds[column] = lower + fraction * (upper - lower)

    rows: list[dict[str, object]] = []
    csr = matrix.tocsr()
    for source_index in range(csr.shape[0]):
        row = csr.getrow(source_index)
        quantile_targets = set(
            row.indices[row.data > thresholds[row.indices]].astype(int).tolist()
        )
        top_targets: set[int] = set()
        if top_k is not None and row.data.size:
            order = np.lexsort((row.indices, -row.data))[: min(top_k, row.data.size)]
            top_targets = set(row.indices[order].astype(int).tolist())
        selected_targets = top_targets | quantile_targets
        if not selected_targets:
            continue
        keep = np.asarray(
            [int(target) in selected_targets for target in row.indices], dtype=bool
        )
        indices, selected = row.indices[keep], row.data[keep]
        denominator = float(selected.sum())
        weights = selected / denominator if denominator > 0 else np.zeros_like(selected)
        for target_index, value, weight in zip(indices, selected, weights):
            in_top = int(target_index) in top_targets
            in_quantile = int(target_index) in quantile_targets
            selection_method = (
                f"row_top_{top_k}+per_gene_quantile_{gene_cell_quantile:g}"
                if in_top and in_quantile
                else (
                    f"row_top_{top_k}"
                    if in_top
                    else f"per_gene_quantile_{gene_cell_quantile:g}"
                )
            )
            rows.append(
                {
                    "source_id": sources[source_index],
                    "target_id": targets[target_index],
                    "source_type": "cell",
                    "target_type": "gene",
                    "relation": "expresses",
                    "raw_value": float(value),
                    "edge_weight": float(weight),
                    "selection_method": selection_method,
                }
            )
    return _finalize_state_edge_table(
        rows,
        relation_key="cell_gene",
    )


def build_cell_gene_edges(
    expression: object,
    cell_ids: pd.Index | list[str],
    gene_ids: pd.Index | list[str],
    *,
    raw_counts: object | None = None,
    top_k: int | None = None,
    gene_cell_quantile: float | None = 0.95,
) -> pd.DataFrame:
    """Build observed cell→gene expression edges from a non-negative matrix."""

    return _build_cell_gene_union_edges(
        expression,
        cell_ids,
        gene_ids,
        raw_values=raw_counts,
        top_k=top_k,
        gene_cell_quantile=gene_cell_quantile,
    )


def build_cell_peak_edges(
    accessibility: object,
    cell_ids: pd.Index | list[str],
    peak_ids: pd.Index | list[str],
    *,
    raw_accessibility: object | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Build observed cell→peak accessibility edges."""

    return _build_observed_edges(
        accessibility,
        cell_ids,
        peak_ids,
        relation_key="cell_peak",
        raw_values=raw_accessibility,
        top_k=top_k,
    )


def _resolve_genome_build(
    table: pd.DataFrame,
    genome_build: str | None,
) -> pd.Series:
    if genome_build is not None:
        value = str(genome_build).strip()
        if not value:
            raise ValueError("genome_build must be a non-empty string.")
        if "genome_build" in table:
            existing = table["genome_build"].dropna().astype(str)
            mismatch = existing[existing.str.strip().ne(value)]
            if not mismatch.empty:
                raise ValueError("Input edge evidence does not match genome_build.")
        return pd.Series(value, index=table.index, dtype="string")
    if "genome_build" not in table:
        raise ValueError("genome_build is required for genomic candidate edges.")
    values = table["genome_build"].astype("string")
    if values.isna().any() or values.str.strip().eq("").any():
        raise ValueError("Genomic candidate evidence contains missing genome_build values.")
    unique = values.drop_duplicates()
    if len(unique) > 1:
        raise ValueError(
            f"Genomic candidate evidence mixes genome builds: {sorted(unique.astype(str))}"
        )
    return values


def build_peak_gene_edges(
    candidates: pd.DataFrame,
    *,
    genome_build: str | None = None,
) -> pd.DataFrame:
    """Standardize coordinate/distance-supported peak→gene candidates."""

    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("candidates must be a pandas DataFrame.")
    required = {"peak", "gene"}
    if missing := required.difference(candidates.columns):
        raise ValueError(f"Peak-gene candidates are missing columns: {sorted(missing)}")
    frame = candidates.copy()
    frame["genome_build"] = _resolve_genome_build(frame, genome_build)
    distance_column = "distance" if "distance" in frame else "dist"
    if distance_column not in frame:
        raise ValueError("Peak-gene candidates require a distance or dist column.")
    frame["distance"] = pd.to_numeric(frame[distance_column], errors="coerce")
    if frame["distance"].isna().any() or (frame["distance"] < 0).any():
        raise ValueError("Peak-gene distance must contain non-negative values.")
    if "prior_score" in frame:
        frame["prior_score"] = pd.to_numeric(frame["prior_score"], errors="coerce")
    elif "weight" in frame:
        frame["prior_score"] = pd.to_numeric(frame["weight"], errors="coerce")
    else:
        frame["prior_score"] = dist_power_decay(frame["distance"].to_numpy())
    if "chrom" not in frame:
        frame["chrom"] = pd.NA
    if "evidence_id" not in frame:
        frame["evidence_id"] = (
            frame["chrom"].fillna("unknown").astype(str)
            + ":"
            + frame["peak"].astype(str)
            + "->"
            + frame["gene"].astype(str)
        )

    rows: list[dict[str, object]] = []
    for (peak, gene), group in frame.groupby(["peak", "gene"], sort=False):
        best = group.sort_values(["prior_score", "distance"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "source_id": str(peak),
                "target_id": str(gene),
                "source_type": "peak",
                "target_type": "gene",
                "relation": "regulates",
                "prior_score": float(best["prior_score"]),
                "evidence_type": "genomic_distance",
                "evidence_id": _join_unique(group["evidence_id"]),
                "distance": float(group["distance"].min()),
                "chrom": best["chrom"],
                "genome_build": best["genome_build"],
                "label_status": "candidate",
                "split": "query",
                "n_evidence": int(len(group)),
            }
        )
    result = _finalize_edge_table(
        pd.DataFrame(rows), extra_columns=["n_evidence"]
    )
    validate_edge_table(result, expected_relation="peak_gene")
    return result


def build_tf_peak_edges(
    motif_candidates: pd.DataFrame,
    *,
    tf_genes: pd.Index | list[str],
    genome_build: str | None = None,
) -> pd.DataFrame:
    """Standardize motif-supported TF(gene)→peak candidate edges."""

    if not isinstance(motif_candidates, pd.DataFrame):
        raise TypeError("motif_candidates must be a pandas DataFrame.")
    required = {"tf", "peak"}
    if missing := required.difference(motif_candidates.columns):
        raise ValueError(f"TF-peak candidates are missing columns: {sorted(missing)}")
    allowed_tfs = set(_ids(tf_genes, "tf_genes"))
    frame = motif_candidates.copy()
    frame["tf"] = frame["tf"].astype(str)
    frame["peak"] = frame["peak"].astype(str)
    frame = frame[frame["tf"].isin(allowed_tfs)].copy()
    frame["genome_build"] = _resolve_genome_build(frame, genome_build)
    if "prior_score" in frame:
        score = pd.to_numeric(frame["prior_score"], errors="coerce")
    elif "weight" in frame:
        score = pd.to_numeric(frame["weight"], errors="coerce")
    elif "motif_score" in frame:
        raw_score = pd.to_numeric(frame["motif_score"], errors="coerce")
        maximum = raw_score.groupby(frame["tf"]).transform("max")
        score = raw_score / maximum.where(maximum > 0, 1.0)
    else:
        score = pd.Series(1.0, index=frame.index)
    frame["prior_score"] = score
    evidence_column = next(
        (
            column
            for column in ("evidence_id", "motif_ids", "motif_id_or_site_id")
            if column in frame
        ),
        None,
    )
    frame["evidence_id"] = (
        frame[evidence_column].astype("string")
        if evidence_column is not None
        else frame["tf"].astype("string")
    )
    if "chrom" not in frame:
        frame["chrom"] = pd.NA

    rows: list[dict[str, object]] = []
    for (tf, peak), group in frame.groupby(["tf", "peak"], sort=False):
        best = group.sort_values("prior_score", ascending=False).iloc[0]
        rows.append(
            {
                "source_id": str(tf),
                "target_id": str(peak),
                "source_type": "gene",
                "target_type": "peak",
                "relation": "binds",
                "prior_score": float(best["prior_score"]),
                "evidence_type": "motif_overlap",
                "evidence_id": _join_unique(group["evidence_id"]),
                "distance": pd.NA,
                "chrom": best["chrom"],
                "genome_build": best["genome_build"],
                "label_status": "weak_positive",
                "split": "query",
                "n_evidence": int(
                    pd.to_numeric(
                        group.get("n_motif_hits", pd.Series(1, index=group.index)),
                        errors="coerce",
                    )
                    .fillna(1)
                    .sum()
                ),
                "motif_score": pd.to_numeric(
                    group.get("motif_score", pd.Series(np.nan, index=group.index)),
                    errors="coerce",
                ).max(),
                "pvalue": pd.to_numeric(
                    group.get("pvalue", pd.Series(np.nan, index=group.index)),
                    errors="coerce",
                ).min(),
                "overlap_bp": pd.to_numeric(
                    group.get("overlap_bp", pd.Series(np.nan, index=group.index)),
                    errors="coerce",
                ).max(),
                "source": _join_unique(
                    group.get("source", pd.Series("JASPAR", index=group.index))
                ),
                "source_version": _join_unique(
                    group.get(
                        "source_version",
                        pd.Series("unspecified", index=group.index),
                    )
                ),
            }
        )
    result = _finalize_edge_table(
        pd.DataFrame(rows),
        extra_columns=[
            "n_evidence",
            "motif_score",
            "pvalue",
            "overlap_bp",
            "source",
            "source_version",
        ],
    )
    validate_edge_table(result, expected_relation="tf_peak")
    return result


def build_tf_target_gold_edges(
    gold_standard: pd.DataFrame,
    *,
    role: str,
    tf_genes: pd.Index | list[str] | None = None,
    gene_ids: pd.Index | list[str] | None = None,
) -> pd.DataFrame:
    """Standardize optional curated TF→target edges for one explicit role.

    Absence from the curated table remains unknown and is never emitted as a
    biological negative.
    """

    if role not in GOLD_ROLES:
        raise ValueError(f"role must be one of {sorted(GOLD_ROLES)}")
    if not isinstance(gold_standard, pd.DataFrame):
        raise TypeError("gold_standard must be a pandas DataFrame.")
    required = {"tf", "target"}
    if missing := required.difference(gold_standard.columns):
        raise ValueError(f"TF-target gold standard is missing columns: {sorted(missing)}")
    frame = gold_standard.copy()
    frame["tf"] = frame["tf"].astype(str)
    frame["target"] = frame["target"].astype(str)
    if tf_genes is not None:
        frame = frame[frame["tf"].isin(set(_ids(tf_genes, "tf_genes")))]
    if gene_ids is not None:
        frame = frame[frame["target"].isin(set(_ids(gene_ids, "gene_ids")))]
    frame = frame[frame["tf"] != frame["target"]].copy()
    for column, default in (
        ("source", "curated"),
        ("evidence", ""),
        ("confidence", ""),
        ("mor", 0),
        ("mor_conflict", False),
    ):
        if column not in frame:
            frame[column] = default

    rows: list[dict[str, object]] = []
    for (tf, target), group in frame.groupby(["tf", "target"], sort=False):
        evidence = _join_unique(group["evidence"])
        source = _join_unique(group["source"])
        mor_values = set(
            pd.to_numeric(group["mor"], errors="coerce").fillna(0).map(np.sign).astype(int)
        )
        nonzero_mor = mor_values.difference({0})
        mor_conflict = len(nonzero_mor) > 1 or bool(
            pd.Series(group["mor_conflict"]).fillna(False).astype(bool).any()
        )
        mor = 0 if mor_conflict or len(nonzero_mor) != 1 else next(iter(nonzero_mor))
        rows.append(
            {
                "source_id": str(tf),
                "target_id": str(target),
                "source_type": "gene",
                "target_type": "gene",
                "relation": "gold_regulates",
                "prior_score": 1.0,
                "evidence_type": "curated_tf_target",
                "evidence_id": evidence or source,
                "distance": pd.NA,
                "chrom": pd.NA,
                "genome_build": pd.NA,
                "label_status": "gold_positive",
                "split": GOLD_ROLES[role],
                "supervision_role": role,
                "mor": int(mor),
                "mor_conflict": bool(mor_conflict),
                "confidence": _join_unique(group["confidence"]),
                "source": source,
                "evidence": evidence,
            }
        )
    result = _finalize_edge_table(
        pd.DataFrame(rows),
        extra_columns=[
            "supervision_role",
            "mor",
            "mor_conflict",
            "confidence",
            "source",
            "evidence",
        ],
    )
    validate_edge_table(result, expected_relation="tf_target")
    return result


@dataclass(frozen=True)
class EdgeTableBundle:
    """The five inspectable tables that precede HERTA graph conversion."""

    cell_gene: pd.DataFrame
    cell_peak: pd.DataFrame
    peak_gene: pd.DataFrame
    tf_peak: pd.DataFrame
    tf_target: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        for relation_key in MESSAGE_RELATIONS:
            validate_state_edge_table(
                getattr(self, relation_key), expected_relation=relation_key
            )
        for relation_key in ("peak_gene", "tf_peak"):
            validate_edge_table(
                getattr(self, relation_key), expected_relation=relation_key
            )
        if self.tf_target is not None:
            validate_edge_table(self.tf_target, expected_relation="tf_target")
            duplicated = self.tf_target.duplicated(
                ["source_id", "target_id"], keep=False
            )
            if duplicated.any() and (
                self.tf_target.loc[duplicated]
                .groupby(["source_id", "target_id"])["split"]
                .nunique()
                .gt(1)
                .any()
            ):
                raise ValueError("TF-target gold edges leak across split roles.")

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return named tables without converting them to graph tensors."""

        tables = {
            "cell_gene": self.cell_gene,
            "cell_peak": self.cell_peak,
            "peak_gene": self.peak_gene,
            "tf_peak": self.tf_peak,
        }
        if self.tf_target is not None:
            tables["tf_target"] = self.tf_target
        return tables


def build_edge_table_bundle(
    multiome: object,
    *,
    expression: object | None = None,
    accessibility: object | None = None,
    peak_gene_candidates: pd.DataFrame | None = None,
    tf_peak_candidates: pd.DataFrame | None = None,
    tf_target_gold: pd.DataFrame | None = None,
    tf_target_role: str = "training_weak_supervision",
    top_cell_gene: int | None = None,
    top_cell_peak: int | None = None,
    gene_cell_quantile: float | None = 0.95,
    genome_build: str | None = None,
) -> EdgeTableBundle:
    """Build the canonical five-table bundle around one aligned ``MultiomeData``.

    Regulatory candidate tables are optional so the same entry point can create
    a Stage-1 observation bundle.  Missing candidate relations are represented
    by typed empty tables, not by a second graph-building convention.
    """

    required_attributes = ("rna", "atac", "genes", "peaks")
    missing_attributes = [
        name for name in required_attributes if not hasattr(multiome, name)
    ]
    if missing_attributes:
        raise TypeError(
            "multiome must provide aligned RNA/ATAC matrices and annotations; "
            f"missing={missing_attributes}"
        )
    genes = getattr(multiome, "genes")
    peaks = getattr(multiome, "peaks")
    if not isinstance(genes, pd.DataFrame) or "gene" not in genes:
        raise ValueError("multiome.genes must be a DataFrame with a 'gene' column.")
    if not isinstance(peaks, pd.DataFrame) or "peak" not in peaks:
        raise ValueError("multiome.peaks must be a DataFrame with a 'peak' column.")
    gene_ids = genes["gene"].astype(str).tolist()
    peak_ids = peaks["peak"].astype(str).tolist()
    n_cells = int(getattr(multiome, "rna").shape[0])
    cell_names = getattr(multiome, "cell_names", None)
    cell_ids = list(cell_names or [f"cell_{index}" for index in range(n_cells)])
    if len(cell_ids) != n_cells:
        raise ValueError("multiome.cell_names must align with the input matrices.")

    cell_gene = build_cell_gene_edges(
        getattr(multiome, "rna") if expression is None else expression,
        cell_ids,
        gene_ids,
        raw_counts=getattr(multiome, "rna"),
        top_k=top_cell_gene,
        gene_cell_quantile=gene_cell_quantile,
    )
    cell_peak = build_cell_peak_edges(
        getattr(multiome, "atac") if accessibility is None else accessibility,
        cell_ids,
        peak_ids,
        raw_accessibility=getattr(multiome, "atac"),
        top_k=top_cell_peak,
    )
    peak_gene = (
        empty_edge_table("peak_gene")
        if peak_gene_candidates is None
        else build_peak_gene_edges(
            peak_gene_candidates,
            genome_build=genome_build,
        )
    )
    tf_names = list(getattr(multiome, "tf_names", None) or [])
    tf_peak = (
        empty_edge_table("tf_peak")
        if tf_peak_candidates is None
        else build_tf_peak_edges(
            tf_peak_candidates,
            tf_genes=tf_names,
            genome_build=genome_build,
        )
    )
    tf_target = (
        None
        if tf_target_gold is None
        else build_tf_target_gold_edges(
            tf_target_gold,
            role=tf_target_role,
            tf_genes=tf_names,
            gene_ids=gene_ids,
        )
    )
    return EdgeTableBundle(
        cell_gene=cell_gene,
        cell_peak=cell_peak,
        peak_gene=peak_gene,
        tf_peak=tf_peak,
        tf_target=tf_target,
    )
