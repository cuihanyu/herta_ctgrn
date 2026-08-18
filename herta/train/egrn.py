"""Cell-type eGRN inference and cell regulatory-state construction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from herta.model.regulatory_scoring import (
    AGGREGATION_MODES,
    DEFAULT_AGGREGATION_MODE,
    aggregate_numpy,
    canonical_aggregation_mode,
    path_score_numpy,
)


@dataclass(frozen=True)
class RegulatoryStateConfig:
    """Configuration for context scoring and regulatory-state normalization."""

    mediator_aggregation: str = DEFAULT_AGGREGATION_MODE
    aggregation_top_k: int = 10
    activity_normalization: str = "cell_max"
    use_target_gene_activity: bool = False
    state_library_normalize: bool = True
    state_log1p: bool = False
    state_log1p_scale: float = 1e4
    state_robust_scale: bool = True
    state_iqr_floor: float = 1e-3
    min_tf_variance: float = 1e-8
    include_tf_target_activity: bool = False
    top_mediators_per_tf_target: int | None = None
    path_chunk_size: int = 4096
    eps: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mediator_aggregation",
            canonical_aggregation_mode(self.mediator_aggregation),
        )
        if self.mediator_aggregation not in AGGREGATION_MODES:
            raise ValueError("Unsupported mediator_aggregation.")
        if self.aggregation_top_k <= 0:
            raise ValueError("aggregation_top_k must be positive.")
        if self.activity_normalization not in {"cell_max", "library", "none"}:
            raise ValueError(
                "activity_normalization must be 'cell_max', 'library', or 'none'."
            )
        if self.state_log1p_scale <= 0 or self.eps <= 0 or self.state_iqr_floor <= 0:
            raise ValueError(
                "state_log1p_scale, state_iqr_floor, and eps must be positive."
            )
        if self.min_tf_variance < 0:
            raise ValueError("min_tf_variance must be non-negative.")
        if (
            self.top_mediators_per_tf_target is not None
            and self.top_mediators_per_tf_target <= 0
        ):
            raise ValueError("top_mediators_per_tf_target must be positive.")
        if self.path_chunk_size <= 0:
            raise ValueError("path_chunk_size must be positive.")

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, object] | None,
    ) -> "RegulatoryStateConfig":
        if config is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unknown = set(config).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown regulatory-state config keys: {sorted(unknown)}")
        return cls(**dict(config))


@dataclass(frozen=True)
class RegulatoryStateResult:
    """Raw TF regulon activity and its clustering representation."""

    raw: pd.DataFrame
    representation: pd.DataFrame
    metadata: dict[str, object]
    tf_target_activity: pd.DataFrame | None = None
    selected_tfs: tuple[str, ...] = ()

    @property
    def matrix(self) -> pd.DataFrame:
        """Return the normalized cell-by-TF matrix used by HERTA clustering."""

        return self.representation


@dataclass(frozen=True)
class EGRNInferenceResult:
    """Inspectable link tables, GRN aggregation, and regulatory state."""

    tf_peak_links: pd.DataFrame
    peak_gene_links: pd.DataFrame
    tf_target_egrn: pd.DataFrame
    tf_target_grn: pd.DataFrame
    regulatory_state: RegulatoryStateResult
    output_paths: dict[str, Path]


def _score_column(table: pd.DataFrame) -> str:
    for column in (
        "calibrated_score",
        "model_score",
        "score",
        "prior_score",
        "weight",
    ):
        if column in table:
            return column
    raise ValueError(
        "Link table requires calibrated_score, model_score, score, prior_score, or weight."
    )


def _standardize_tf_peak(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    if "tf" not in frame and "source_id" in frame:
        frame["tf"] = frame["source_id"]
    if "peak" not in frame and "target_id" in frame:
        frame["peak"] = frame["target_id"]
    if missing := {"tf", "peak"}.difference(frame.columns):
        raise ValueError(f"TF-peak links are missing columns: {sorted(missing)}")
    score_col = _score_column(frame)
    frame["tf"] = frame["tf"].astype(str)
    frame["peak"] = frame["peak"].astype(str)
    frame["structural_score"] = pd.to_numeric(frame[score_col], errors="coerce")
    _validate_probability(frame["structural_score"], "TF-peak structural scores")
    evidence_col = next(
        (
            column
            for column in ("evidence_id", "motif_ids", "motif_id_or_site_id")
            if column in frame
        ),
        None,
    )
    frame["evidence_id"] = (
        frame[evidence_col].fillna("").astype(str) if evidence_col else ""
    )
    rows: list[dict[str, object]] = []
    for (tf, peak), group in frame.groupby(["tf", "peak"], sort=False):
        best = group.sort_values("structural_score", ascending=False).iloc[0]
        rows.append(
            {
                "tf": tf,
                "peak": peak,
                "structural_score": float(best["structural_score"]),
                "evidence_id": _join_unique(group["evidence_id"]),
                "genome_build": _first_or_empty(group, "genome_build"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "tf",
            "peak",
            "structural_score",
            "evidence_id",
            "genome_build",
        ],
    )


def _standardize_peak_gene(table: pd.DataFrame) -> pd.DataFrame:
    frame = table.copy()
    if "peak" not in frame and "source_id" in frame:
        frame["peak"] = frame["source_id"]
    if "gene" not in frame:
        if "target_gene" in frame:
            frame["gene"] = frame["target_gene"]
        elif "target_id" in frame:
            frame["gene"] = frame["target_id"]
    if missing := {"peak", "gene"}.difference(frame.columns):
        raise ValueError(f"Peak-gene links are missing columns: {sorted(missing)}")
    score_col = _score_column(frame)
    frame["peak"] = frame["peak"].astype(str)
    frame["gene"] = frame["gene"].astype(str)
    frame["structural_score"] = pd.to_numeric(frame[score_col], errors="coerce")
    _validate_probability(frame["structural_score"], "Peak-gene structural scores")
    frame["distance"] = pd.to_numeric(
        frame.get("distance", frame.get("dist", np.nan)),
        errors="coerce",
    )
    if bool((frame["distance"].dropna() < 0).any()):
        raise ValueError("Peak-gene distances must be non-negative.")
    frame["evidence_id"] = (
        frame["evidence_id"].fillna("").astype(str)
        if "evidence_id" in frame
        else pd.Series("", index=frame.index, dtype=str)
    )
    rows: list[dict[str, object]] = []
    for (peak, gene), group in frame.groupby(["peak", "gene"], sort=False):
        best = group.sort_values("structural_score", ascending=False).iloc[0]
        rows.append(
            {
                "peak": peak,
                "target_gene": gene,
                "structural_score": float(best["structural_score"]),
                "distance": float(group["distance"].min())
                if group["distance"].notna().any()
                else np.nan,
                "evidence_id": _join_unique(group["evidence_id"]),
                "genome_build": _first_or_empty(group, "genome_build"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "peak",
            "target_gene",
            "structural_score",
            "distance",
            "evidence_id",
            "genome_build",
        ],
    )


def _validate_probability(values: pd.Series, name: str) -> None:
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must be finite.")
    if not values.between(0, 1).all():
        raise ValueError(f"{name} must be in [0, 1].")


def _join_unique(values: pd.Series) -> str:
    return ";".join(
        sorted(
            {
                str(value).strip()
                for value in values
                if pd.notna(value) and str(value).strip()
            }
        )
    )


def _first_or_empty(group: pd.DataFrame, column: str) -> str:
    if column not in group:
        return ""
    values = [
        str(value)
        for value in group[column]
        if pd.notna(value) and str(value).strip()
    ]
    return values[0] if values else ""


def _compose_paths(
    tf_peak: pd.DataFrame,
    peak_gene: pd.DataFrame,
    config: RegulatoryStateConfig,
) -> pd.DataFrame:
    builds = {
        value
        for value in pd.concat(
            [tf_peak["genome_build"], peak_gene["genome_build"]],
            ignore_index=True,
        ).astype(str)
        if value.strip()
    }
    if len(builds) > 1:
        raise ValueError(f"eGRN links mix genome builds: {sorted(builds)}")
    paths = tf_peak.rename(
        columns={
            "structural_score": "tf_peak_score",
            "evidence_id": "tf_peak_evidence_id",
            "genome_build": "tf_peak_genome_build",
        }
    ).merge(
        peak_gene.rename(
            columns={
                "structural_score": "peak_gene_score",
                "evidence_id": "peak_gene_evidence_id",
                "genome_build": "peak_gene_genome_build",
            }
        ),
        on="peak",
        how="inner",
        validate="many_to_many",
    )
    if paths.empty:
        raise ValueError("TF-peak and peak-gene links do not share any mediator peaks.")
    paths["structural_path_score"] = path_score_numpy(
        paths["tf_peak_score"].to_numpy(dtype=float),
        paths["peak_gene_score"].to_numpy(dtype=float),
    )
    if config.top_mediators_per_tf_target is not None:
        paths = (
            paths.sort_values("structural_path_score", ascending=False)
            .groupby(["tf", "target_gene"], sort=False, group_keys=False)
            .head(config.top_mediators_per_tf_target)
        )
    return paths.sort_values(
        ["tf", "target_gene", "structural_path_score"],
        ascending=[True, True, False],
        ignore_index=True,
    )


def _activity_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    cell_index: pd.Index,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame indexed by cell ID.")
    result = frame.copy()
    result.index = result.index.astype(str)
    result.columns = result.columns.astype(str)
    if result.index.has_duplicates or result.columns.has_duplicates:
        raise ValueError(f"{name} cell and feature identifiers must be unique.")
    if set(result.index) != set(cell_index):
        raise ValueError(f"{name} cell IDs do not align with TF activity.")
    result = result.loc[cell_index]
    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name} must contain finite, non-negative activity values.")
    return result.astype(float)


def _normalize_activity(
    frame: pd.DataFrame,
    method: str,
    eps: float,
) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    if method == "cell_max":
        denominator = values.max(axis=1, keepdims=True)
    elif method == "library":
        denominator = values.sum(axis=1, keepdims=True)
    else:
        return frame.copy()
    normalized = np.divide(
        values,
        denominator,
        out=np.zeros_like(values),
        where=denominator > eps,
    )
    return pd.DataFrame(normalized, index=frame.index, columns=frame.columns)


def _cell_types(
    values: pd.Series | Sequence[str],
    cell_index: pd.Index,
) -> pd.Series:
    if isinstance(values, pd.Series):
        result = values.copy()
        result.index = result.index.astype(str)
        if set(result.index) == set(cell_index):
            result = result.loc[cell_index]
        elif len(result) == len(cell_index):
            result.index = cell_index
        else:
            raise ValueError("cell_types do not align with activity rows.")
    else:
        if len(values) != len(cell_index):
            raise ValueError("cell_types do not align with activity rows.")
        result = pd.Series(values, index=cell_index)
    if result.isna().any():
        raise ValueError("cell_types contains missing values.")
    return result.astype(str)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing activity columns: {sorted(missing)[:5]}")


def _path_activity(
    paths: pd.DataFrame,
    tf_values: np.ndarray,
    peak_values: np.ndarray,
    tf_index: Mapping[str, int],
    peak_index: Mapping[str, int],
    target_values: np.ndarray | None,
    target_index: Mapping[str, int] | None,
    selected_cells: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    current = paths.iloc[start:stop]
    tf_columns = np.fromiter(
        (tf_index[name] for name in current["tf"]),
        dtype=int,
        count=len(current),
    )
    peak_columns = np.fromiter(
        (peak_index[name] for name in current["peak"]),
        dtype=int,
        count=len(current),
    )
    activity = path_score_numpy(
        current["tf_peak_score"].to_numpy(dtype=float)[None, :],
        current["peak_gene_score"].to_numpy(dtype=float)[None, :],
        tf_values[np.ix_(selected_cells, tf_columns)],
        peak_values[np.ix_(selected_cells, peak_columns)],
    )
    if target_values is not None and target_index is not None:
        target_columns = np.fromiter(
            (target_index[name] for name in current["target_gene"]),
            dtype=int,
            count=len(current),
        )
        activity = path_score_numpy(
            activity,
            np.ones_like(activity),
            target_values[np.ix_(selected_cells, target_columns)],
        )
    return activity


def _regulatory_state_from_paths(
    paths: pd.DataFrame,
    tf_activity: pd.DataFrame,
    peak_accessibility: pd.DataFrame,
    target_gene_activity: pd.DataFrame | None,
    config: RegulatoryStateConfig,
) -> RegulatoryStateResult:
    tf_names = list(dict.fromkeys(paths["tf"].astype(str)))
    tf_output_index = {name: index for index, name in enumerate(tf_names)}
    tf_index = {name: index for index, name in enumerate(tf_activity.columns)}
    peak_index = {
        name: index for index, name in enumerate(peak_accessibility.columns)
    }
    target_index = (
        {name: index for index, name in enumerate(target_gene_activity.columns)}
        if target_gene_activity is not None
        else None
    )
    tf_values = tf_activity.to_numpy(dtype=float)
    peak_values = peak_accessibility.to_numpy(dtype=float)
    target_values = (
        target_gene_activity.to_numpy(dtype=float)
        if target_gene_activity is not None
        else None
    )
    all_cells = np.arange(len(tf_activity), dtype=int)
    raw = np.zeros((len(tf_activity), len(tf_names)), dtype=float)
    tf_target_values: list[np.ndarray] = []
    tf_target_pairs: list[tuple[str, str]] = []
    for (tf, _target_gene), group in paths.groupby(
        ["tf", "target_gene"],
        sort=False,
    ):
        indices = group.index.to_numpy(dtype=int)
        values = _path_activity(
            paths,
            tf_values,
            peak_values,
            tf_index,
            peak_index,
            target_values,
            target_index,
            all_cells,
            int(indices.min()),
            int(indices.max()) + 1,
        )
        # Paths are sorted contiguously by TF and target gene.
        aggregate = aggregate_numpy(
            values,
            config.mediator_aggregation,
            axis=1,
            top_k=config.aggregation_top_k,
        )
        raw[:, tf_output_index[str(tf)]] += aggregate
        if config.include_tf_target_activity:
            tf_target_values.append(np.asarray(aggregate, dtype=float))
            tf_target_pairs.append((str(tf), str(_target_gene)))
    raw_frame = pd.DataFrame(
        raw,
        index=tf_activity.index.copy(),
        columns=tf_names,
    )
    representation = raw.copy()
    if config.state_library_normalize:
        denominator = representation.sum(axis=1, keepdims=True)
        representation = np.divide(
            representation,
            denominator,
            out=np.zeros_like(representation),
            where=denominator > config.eps,
        )
    if config.state_log1p:
        representation = np.log1p(representation * config.state_log1p_scale)
    variances = representation.var(axis=0)
    keep = variances > config.min_tf_variance
    kept_tfs = [name for name, selected in zip(tf_names, keep) if selected]
    dropped_tfs = [name for name, selected in zip(tf_names, keep) if not selected]
    dropped_low_iqr_tfs: list[str] = []
    representation = representation[:, keep]
    if config.state_robust_scale and representation.shape[1]:
        median = np.median(representation, axis=0, keepdims=True)
        q25 = np.quantile(representation, 0.25, axis=0, keepdims=True)
        q75 = np.quantile(representation, 0.75, axis=0, keepdims=True)
        iqr = (q75 - q25).reshape(-1)
        stable = iqr >= config.state_iqr_floor
        dropped_low_iqr_tfs = [
            name for name, selected in zip(kept_tfs, stable) if not selected
        ]
        kept_tfs = [name for name, selected in zip(kept_tfs, stable) if selected]
        representation = representation[:, stable]
        if representation.shape[1]:
            representation = (representation - median[:, stable]) / iqr[stable][None, :]
    representation_frame = pd.DataFrame(
        representation,
        index=tf_activity.index.copy(),
        columns=kept_tfs,
    )
    tf_target_activity = None
    if config.include_tf_target_activity:
        tf_target_activity = pd.DataFrame(
            np.column_stack(tf_target_values)
            if tf_target_values
            else np.empty((len(tf_activity), 0), dtype=float),
            index=tf_activity.index.copy(),
            columns=pd.MultiIndex.from_tuples(
                tf_target_pairs,
                names=["tf", "target_gene"],
            ),
        )
    metadata = {
        **asdict(config),
        "formula": (
            "s_tf_peak * s_peak_gene * a_tf * a_peak"
            + (" * a_target" if config.use_target_gene_activity else "")
        ),
        "raw_state_aggregation": (
            "sum over target genes after mediator_aggregation per TF-target"
        ),
        "n_cells": len(raw_frame),
        "n_tfs_raw": len(tf_names),
        "n_tfs_representation": len(kept_tfs),
        "selected_tfs": kept_tfs,
        "dropped_low_variance_tfs": dropped_tfs,
        "dropped_low_iqr_tfs": dropped_low_iqr_tfs,
        "clustering_role": "primary_herta_representation",
        "cluster_loss": "experimental_optional_disabled_by_default",
    }
    return RegulatoryStateResult(
        raw=raw_frame,
        representation=representation_frame,
        metadata=metadata,
        tf_target_activity=tf_target_activity,
        selected_tfs=tuple(kept_tfs),
    )


def build_regulatory_state(
    tf_peak_links: pd.DataFrame,
    peak_gene_links: pd.DataFrame,
    tf_activity: pd.DataFrame,
    peak_accessibility: pd.DataFrame,
    *,
    target_gene_activity: pd.DataFrame | None = None,
    config: RegulatoryStateConfig | Mapping[str, object] | None = None,
) -> RegulatoryStateResult:
    """Build per-cell TF regulon activity from scored TF→peak→gene paths."""

    resolved = (
        config
        if isinstance(config, RegulatoryStateConfig)
        else RegulatoryStateConfig.from_mapping(config)
    )
    tf_peak = _standardize_tf_peak(tf_peak_links)
    peak_gene = _standardize_peak_gene(peak_gene_links)
    paths = _compose_paths(tf_peak, peak_gene, resolved)
    cell_index = pd.Index(tf_activity.index.astype(str))
    tf_activity = _activity_frame(
        tf_activity,
        name="tf_activity",
        cell_index=cell_index,
    )
    peak_accessibility = _activity_frame(
        peak_accessibility,
        name="peak_accessibility",
        cell_index=cell_index,
    )
    _require_columns(tf_activity, set(paths["tf"]), "tf_activity")
    _require_columns(
        peak_accessibility,
        set(paths["peak"]),
        "peak_accessibility",
    )
    tf_activity = _normalize_activity(
        tf_activity,
        resolved.activity_normalization,
        resolved.eps,
    )
    peak_accessibility = _normalize_activity(
        peak_accessibility,
        resolved.activity_normalization,
        resolved.eps,
    )
    target = None
    if resolved.use_target_gene_activity:
        if target_gene_activity is None:
            raise ValueError(
                "target_gene_activity is required when its gate is enabled."
            )
        target = _activity_frame(
            target_gene_activity,
            name="target_gene_activity",
            cell_index=cell_index,
        )
        _require_columns(target, set(paths["target_gene"]), "target_gene_activity")
        target = _normalize_activity(
            target,
            resolved.activity_normalization,
            resolved.eps,
        )
    return _regulatory_state_from_paths(
        paths,
        tf_activity,
        peak_accessibility,
        target,
        resolved,
    )


def _context_path_scores(
    paths: pd.DataFrame,
    tf_activity: pd.DataFrame,
    peak_accessibility: pd.DataFrame,
    target_gene_activity: pd.DataFrame | None,
    members: np.ndarray,
) -> np.ndarray:
    tf_mean = tf_activity.iloc[members].mean(axis=0)
    peak_mean = peak_accessibility.iloc[members].mean(axis=0)
    context_factors = [
        paths["tf"].map(tf_mean).to_numpy(dtype=float),
        paths["peak"].map(peak_mean).to_numpy(dtype=float),
    ]
    if target_gene_activity is not None:
        target_mean = target_gene_activity.iloc[members].mean(axis=0)
        context_factors.append(
            paths["target_gene"].map(target_mean).to_numpy(dtype=float)
        )
    return path_score_numpy(
        paths["tf_peak_score"].to_numpy(dtype=float),
        paths["peak_gene_score"].to_numpy(dtype=float),
        *context_factors,
    )


def _context_link_tables(
    tf_peak: pd.DataFrame,
    peak_gene: pd.DataFrame,
    tf_activity: pd.DataFrame,
    peak_accessibility: pd.DataFrame,
    target_gene_activity: pd.DataFrame | None,
    context: str,
    members: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tf_mean = tf_activity.iloc[members].mean(axis=0)
    peak_mean = peak_accessibility.iloc[members].mean(axis=0)
    tp = tf_peak.copy()
    tp["cell_type"] = context
    tp["n_cells"] = len(members)
    tp["tf_activity_mean"] = tp["tf"].map(tf_mean).astype(float)
    tp["peak_accessibility_mean"] = tp["peak"].map(peak_mean).astype(float)
    tp["context_activity_mean"] = (
        tp["tf_activity_mean"] * tp["peak_accessibility_mean"]
    )
    tp["context_score"] = (
        tp["structural_score"] * tp["context_activity_mean"]
    )
    pg = peak_gene.copy()
    pg["cell_type"] = context
    pg["n_cells"] = len(members)
    pg["peak_accessibility_mean"] = pg["peak"].map(peak_mean).astype(float)
    pg["target_gene_activity_mean"] = np.nan
    pg["context_activity_mean"] = pg["peak_accessibility_mean"]
    if target_gene_activity is not None:
        target_mean = target_gene_activity.iloc[members].mean(axis=0)
        pg["target_gene_activity_mean"] = (
            pg["target_gene"].map(target_mean).astype(float)
        )
        pg["context_activity_mean"] = (
            pg["peak_accessibility_mean"]
            * pg["target_gene_activity_mean"]
        )
    pg["context_score"] = (
        pg["structural_score"] * pg["context_activity_mean"]
    )
    return tp, pg


def infer_cell_type_egrn(
    tf_peak_links: pd.DataFrame,
    peak_gene_links: pd.DataFrame,
    tf_activity: pd.DataFrame,
    peak_accessibility: pd.DataFrame,
    cell_types: pd.Series | Sequence[str],
    *,
    target_gene_activity: pd.DataFrame | None = None,
    config: RegulatoryStateConfig | Mapping[str, object] | None = None,
    output_dir: str | Path | None = None,
) -> EGRNInferenceResult:
    """Infer cell-type-specific TF→peak→target paths and regulatory state."""

    resolved = (
        config
        if isinstance(config, RegulatoryStateConfig)
        else RegulatoryStateConfig.from_mapping(config)
    )
    tf_peak = _standardize_tf_peak(tf_peak_links)
    peak_gene = _standardize_peak_gene(peak_gene_links)
    paths = _compose_paths(tf_peak, peak_gene, resolved)
    cell_index = pd.Index(tf_activity.index.astype(str))
    tf_activity = _activity_frame(
        tf_activity,
        name="tf_activity",
        cell_index=cell_index,
    )
    peak_accessibility = _activity_frame(
        peak_accessibility,
        name="peak_accessibility",
        cell_index=cell_index,
    )
    _require_columns(tf_activity, set(paths["tf"]), "tf_activity")
    _require_columns(
        peak_accessibility,
        set(paths["peak"]),
        "peak_accessibility",
    )
    tf_activity = _normalize_activity(
        tf_activity,
        resolved.activity_normalization,
        resolved.eps,
    )
    peak_accessibility = _normalize_activity(
        peak_accessibility,
        resolved.activity_normalization,
        resolved.eps,
    )
    target = None
    if resolved.use_target_gene_activity:
        if target_gene_activity is None:
            raise ValueError(
                "target_gene_activity is required when its gate is enabled."
            )
        target = _activity_frame(
            target_gene_activity,
            name="target_gene_activity",
            cell_index=cell_index,
        )
        _require_columns(target, set(paths["target_gene"]), "target_gene_activity")
        target = _normalize_activity(
            target,
            resolved.activity_normalization,
            resolved.eps,
        )
    labels = _cell_types(cell_types, cell_index)
    state = _regulatory_state_from_paths(
        paths,
        tf_activity,
        peak_accessibility,
        target,
        resolved,
    )

    tp_tables: list[pd.DataFrame] = []
    pg_tables: list[pd.DataFrame] = []
    path_tables: list[pd.DataFrame] = []
    grn_rows: list[dict[str, object]] = []
    for context in pd.unique(labels):
        members = np.flatnonzero(labels.to_numpy() == context)
        tp_current, pg_current = _context_link_tables(
            tf_peak,
            peak_gene,
            tf_activity,
            peak_accessibility,
            target,
            str(context),
            members,
        )
        tp_tables.append(tp_current)
        pg_tables.append(pg_current)
        current = paths.copy()
        current["cell_type"] = str(context)
        current["n_cells"] = len(members)
        current["tf_activity_mean"] = current["tf"].map(
            tf_activity.iloc[members].mean(axis=0)
        )
        current["peak_accessibility_mean"] = current["peak"].map(
            peak_accessibility.iloc[members].mean(axis=0)
        )
        current["target_gene_activity_mean"] = np.nan
        if target is not None:
            current["target_gene_activity_mean"] = current["target_gene"].map(
                target.iloc[members].mean(axis=0)
            )
        current["path_score"] = _context_path_scores(
            paths,
            tf_activity,
            peak_accessibility,
            target,
            members,
        )
        for (tf, gene), group in current.groupby(
            ["tf", "target_gene"],
            sort=False,
        ):
            values = group["path_score"].to_numpy(dtype=float)
            aggregated = float(
                aggregate_numpy(
                    values,
                    resolved.mediator_aggregation,
                    axis=0,
                    top_k=resolved.aggregation_top_k,
                )
            )
            mediator_count = int(group["peak"].nunique())
            supporting_peaks = ";".join(
                group.sort_values("path_score", ascending=False)["peak"]
                .astype(str)
                .drop_duplicates()
            )
            tf_peak_mean = float(group["tf_peak_score"].mean())
            peak_gene_mean = float(group["peak_gene_score"].mean())
            current.loc[group.index, "tf_target_score"] = aggregated
            current.loc[group.index, "tf_target_score_sum"] = float(values.sum())
            current.loc[group.index, "n_mediator_peaks"] = mediator_count
            current.loc[group.index, "egrn_score"] = aggregated
            current.loc[group.index, "n_supporting_peaks"] = mediator_count
            current.loc[group.index, "supporting_peaks"] = supporting_peaks
            current.loc[group.index, "tf_peak_score_mean"] = tf_peak_mean
            current.loc[group.index, "peak_gene_score_mean"] = peak_gene_mean
            current.loc[group.index, "aggregation_mode"] = (
                resolved.mediator_aggregation
            )
            grn_rows.append(
                {
                    "cell_type": str(context),
                    "tf": str(tf),
                    "target_gene": str(gene),
                    "egrn_score": aggregated,
                    "n_supporting_peaks": mediator_count,
                    "supporting_peaks": supporting_peaks,
                    "tf_peak_score_mean": tf_peak_mean,
                    "peak_gene_score_mean": peak_gene_mean,
                    "aggregation_mode": resolved.mediator_aggregation,
                    # Backward-compatible aliases used by plotting/evaluation.
                    "gene": str(gene),
                    "score": aggregated,
                    "score_sum": float(values.sum()),
                    "n_mediator_peaks": mediator_count,
                    "top_mediator_peaks": ";".join(
                        supporting_peaks.split(";")[:10]
                    ),
                    "n_cells": len(members),
                    "aggregation": resolved.mediator_aggregation,
                    "sign": 0,
                    "sign_source": "unsigned",
                }
            )
        current["rank"] = (
            current.groupby(["cell_type", "tf"])["path_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        current["aggregation"] = resolved.mediator_aggregation
        path_tables.append(current)
    tf_peak_context = pd.concat(tp_tables, ignore_index=True)
    peak_gene_context = pd.concat(pg_tables, ignore_index=True)
    tf_target_egrn = pd.concat(path_tables, ignore_index=True)
    tf_target_egrn = tf_target_egrn.sort_values(
        ["cell_type", "tf_target_score", "path_score"],
        ascending=[True, False, False],
        ignore_index=True,
    )
    tf_target_grn = pd.DataFrame(grn_rows).sort_values(
        ["cell_type", "score"],
        ascending=[True, False],
        ignore_index=True,
    )
    output_paths: dict[str, Path] = {}
    result = EGRNInferenceResult(
        tf_peak_links=tf_peak_context,
        peak_gene_links=peak_gene_context,
        tf_target_egrn=tf_target_egrn,
        tf_target_grn=tf_target_grn,
        regulatory_state=state,
        output_paths=output_paths,
    )
    if output_dir is not None:
        output_paths.update(write_egrn_outputs(result, output_dir))
    return result


def write_egrn_outputs(
    result: EGRNInferenceResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the required CSV artifacts without changing reference inputs."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "tf_peak_links": target / "tf_peak_links.csv",
        "peak_gene_links": target / "peak_gene_links.csv",
        "tf_target_egrn": target / "tf_target_egrn.csv",
        "tf_target_grn": target / "tf_target_grn.csv",
    }
    result.tf_peak_links.to_csv(paths["tf_peak_links"], index=False)
    result.peak_gene_links.to_csv(paths["peak_gene_links"], index=False)
    result.tf_target_egrn.to_csv(paths["tf_target_egrn"], index=False)
    result.tf_target_grn.to_csv(paths["tf_target_grn"], index=False)
    paths.update(write_regulatory_state_outputs(result.regulatory_state, target))
    return paths


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def write_regulatory_state_outputs(
    result: RegulatoryStateResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a standard, reloadable post-training regulatory-state bundle."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "regulatory_state": target / "regulatory_state.csv",
        "regulatory_state_raw": target / "regulatory_state_raw.csv",
        "regulatory_state_npz": target / "regulatory_state.npz",
        "regulatory_state_metadata": target / "regulatory_state_metadata.json",
        "selected_tfs": target / "selected_tfs.tsv",
    }
    result.matrix.rename_axis("cell_id").reset_index().to_csv(
        paths["regulatory_state"],
        index=False,
    )
    result.raw.rename_axis("cell_id").reset_index().to_csv(
        paths["regulatory_state_raw"],
        index=False,
    )
    npz_payload: dict[str, np.ndarray] = {
        "matrix": result.matrix.to_numpy(dtype=np.float32),
        "raw": result.raw.to_numpy(dtype=np.float32),
        "cell_ids": result.matrix.index.astype(str).to_numpy(),
        "selected_tfs": np.asarray(result.selected_tfs, dtype=str),
        "raw_tfs": result.raw.columns.astype(str).to_numpy(),
    }
    if result.tf_target_activity is not None:
        tf_target_path = target / "tf_target_activity.csv"
        flat = result.tf_target_activity.copy()
        flat.columns = [
            f"{tf}::{target_gene}"
            for tf, target_gene in flat.columns.to_list()
        ]
        flat.rename_axis("cell_id").reset_index().to_csv(
            tf_target_path,
            index=False,
        )
        paths["tf_target_activity"] = tf_target_path
        npz_payload["tf_target_activity"] = (
            result.tf_target_activity.to_numpy(dtype=np.float32)
        )
        npz_payload["tf_target_tfs"] = np.asarray(
            result.tf_target_activity.columns.get_level_values("tf"),
            dtype=str,
        )
        npz_payload["tf_target_genes"] = np.asarray(
            result.tf_target_activity.columns.get_level_values("target_gene"),
            dtype=str,
        )
    np.savez_compressed(paths["regulatory_state_npz"], **npz_payload)
    with paths["regulatory_state_metadata"].open("w", encoding="utf-8") as stream:
        json.dump(
            result.metadata,
            stream,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    pd.DataFrame({"tf": list(result.selected_tfs)}).to_csv(
        paths["selected_tfs"],
        sep="\t",
        index=False,
    )
    return paths
