"""LINGER-style layered validation for HERTA regulatory predictions."""

from __future__ import annotations

from collections.abc import Sequence
import re
import warnings

import numpy as np
import pandas as pd

from herta.evaluate.gold import (
    GoldStandardResult,
    load_chipseq_peaks,
    load_eqtl_links,
    load_peak_gene_gold,
    load_tf_target_gold,
)
from herta.evaluate.metrics import (
    compute_aupr_ratio,
    compute_auprc,
    compute_auroc,
    compute_early_precision,
    compute_f1_at_thresholds,
    compute_topk_precision,
)


DISTANCE_BINS = (
    "0-5kb",
    "5-10kb",
    "10-20kb",
    "20-50kb",
    "50-100kb",
    "100-200kb",
    "200kb-1Mb",
)


def _column(frame: pd.DataFrame, names: Sequence[str], *, required: bool = True) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    if required:
        raise ValueError(f"Expected one of these columns: {list(names)}")
    return None


def _result(
    status: str,
    reason: str | None = None,
    **tables: object,
) -> dict[str, object]:
    return {
        "status": status,
        "reason": reason,
        "warnings": [],
        **tables,
    }


def _classification_status(labels: pd.Series | np.ndarray) -> tuple[str, str | None]:
    values = np.asarray(labels, dtype=int)
    n_positive = int(values.sum())
    if n_positive == 0:
        return "warning", "no positive evidence overlaps the prediction universe"
    if n_positive == len(values):
        return "warning", "no negative/background candidates are available"
    return "ok", None


def _reported_warnings(metrics: pd.DataFrame) -> list[str]:
    if "warning" not in metrics:
        return []
    return list(
        dict.fromkeys(
            value
            for value in metrics["warning"].dropna().astype(str)
            if value.strip()
        )
    )


def _gold_result(
    value: GoldStandardResult | pd.DataFrame | str | None,
    loader: object,
    **kwargs: object,
) -> GoldStandardResult:
    if isinstance(value, GoldStandardResult):
        return value
    return loader(value, **kwargs)


def _peak_coordinates(frame: pd.DataFrame, peak_col: str = "peak") -> pd.DataFrame:
    result = frame.copy()
    chrom = _column(result, ("chrom", "chr", "chromosome"), required=False)
    start = _column(result, ("start", "chromStart", "chrom_start"), required=False)
    end = _column(result, ("end", "chromEnd", "chrom_end"), required=False)
    if chrom and start and end:
        result["_chrom"] = result[chrom].astype(str)
        result["_start"] = pd.to_numeric(result[start], errors="coerce")
        result["_end"] = pd.to_numeric(result[end], errors="coerce")
        return result
    parsed = result[peak_col].astype(str).str.extract(
        r"^(?P<_chrom>.+?)(?::|-)(?P<_start>\d+)-(?P<_end>\d+)$"
    )
    result["_chrom"] = parsed["_chrom"]
    result["_start"] = pd.to_numeric(parsed["_start"], errors="coerce")
    result["_end"] = pd.to_numeric(parsed["_end"], errors="coerce")
    return result


def _standardize_tf_peak(frame: pd.DataFrame, score_col: str | None = None) -> pd.DataFrame:
    tf_col = _column(frame, ("tf", "source_id", "gene", "regulator"))
    peak_col = _column(frame, ("peak", "target_id", "peak_id"))
    score_col = score_col or _column(
        frame,
        (
            "tf_peak_score",
            "context_score",
            "structural_score",
            "calibrated_score",
            "model_score",
            "score",
            "probability",
            "prior_score",
            "weight",
            "motif_score",
        ),
    )
    result = frame.copy()
    result["tf"] = frame[tf_col].astype(str)
    result["peak"] = frame[peak_col].astype(str)
    result["score"] = pd.to_numeric(frame[score_col], errors="raise").astype(float)
    if not np.isfinite(result["score"]).all():
        raise ValueError("TF-peak scores must be finite.")
    return _peak_coordinates(result)


def _standardize_peak_gene(frame: pd.DataFrame, score_col: str | None = None) -> pd.DataFrame:
    peak_col = _column(frame, ("peak", "source_id", "peak_id"))
    gene_col = _column(frame, ("gene", "target_gene", "target", "target_id"))
    score_col = score_col or _column(
        frame,
        (
            "peak_gene_score",
            "context_score",
            "structural_score",
            "calibrated_score",
            "model_score",
            "score",
            "probability",
            "prior_score",
            "weight",
        ),
    )
    result = frame.copy()
    result["peak"] = frame[peak_col].astype(str)
    result["gene"] = frame[gene_col].astype(str)
    result["score"] = pd.to_numeric(frame[score_col], errors="raise").astype(float)
    if not np.isfinite(result["score"]).all():
        raise ValueError("Peak-gene scores must be finite.")
    return _peak_coordinates(result)


def _standardize_tf_gene(frame: pd.DataFrame, score_col: str | None = None) -> pd.DataFrame:
    tf_col = _column(frame, ("tf", "source_id", "regulator"))
    gene_col = _column(frame, ("gene", "target_gene", "target", "target_id"))
    score_col = score_col or _column(
        frame,
        ("tf_target_score", "egrn_score", "tf_gene_score", "score", "path_score"),
    )
    result = frame.copy()
    result["tf"] = frame[tf_col].astype(str)
    result["gene"] = frame[gene_col].astype(str)
    result["score"] = pd.to_numeric(frame[score_col], errors="raise").astype(float)
    if not np.isfinite(result["score"]).all():
        raise ValueError("TF-gene scores must be finite.")
    context = _column(result, ("cell_type", "cluster", "context"), required=False)
    if context and context != "cell_type":
        result = result.rename(columns={context: "cell_type"})
    return result


def _interval_labels(
    predictions: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> np.ndarray:
    labels = np.zeros(len(predictions), dtype=bool)
    valid_gold = evidence.dropna(subset=["chrom", "start", "end"]).copy()
    if valid_gold.empty:
        return labels
    valid_gold["start"] = pd.to_numeric(valid_gold["start"], errors="coerce")
    valid_gold["end"] = pd.to_numeric(valid_gold["end"], errors="coerce")
    valid_gold = valid_gold.dropna(subset=["start", "end"])
    grouped: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
    full_group = [*group_columns, "chrom"]
    for key, group in valid_gold.groupby(full_group, sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        ordered = group.sort_values("start")
        starts = ordered["start"].to_numpy(dtype=float)
        prefix_max_end = np.maximum.accumulate(ordered["end"].to_numpy(dtype=float))
        grouped[tuple(map(str, key_tuple))] = starts, prefix_max_end
    for position in range(len(predictions)):
        start = predictions["_start"].iloc[position]
        end = predictions["_end"].iloc[position]
        chrom = predictions["_chrom"].iloc[position]
        if pd.isna(start) or pd.isna(end) or pd.isna(chrom):
            continue
        key = tuple(
            str(predictions[column].iloc[position]) for column in group_columns
        ) + (str(chrom),)
        interval = grouped.get(key)
        if interval is None:
            continue
        starts, prefix_max_end = interval
        upper = int(np.searchsorted(starts, float(end), side="left"))
        labels[position] = upper > 0 and prefix_max_end[upper - 1] > float(start)
    return labels


def _label_tf_peak(predictions: pd.DataFrame, evidence: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(len(predictions), dtype=bool)
    if {"tf", "peak"}.issubset(evidence.columns):
        pairs = set(
            map(tuple, evidence[["tf", "peak"]].dropna().astype(str).drop_duplicates().to_numpy())
        )
        labels |= np.array(
            [
                pair in pairs
                for pair in predictions[["tf", "peak"]].astype(str).itertuples(index=False, name=None)
            ]
        )
    if {"tf", "chrom", "start", "end"}.issubset(evidence.columns):
        labels |= _interval_labels(predictions, evidence, group_columns=("tf",))
    return labels


def _label_peak_gene(predictions: pd.DataFrame, evidence: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(len(predictions), dtype=bool)
    if {"peak", "gene"}.issubset(evidence.columns):
        pairs = set(
            map(
                tuple,
                evidence[["peak", "gene"]].dropna().astype(str).drop_duplicates().to_numpy(),
            )
        )
        labels |= np.array(
            [
                pair in pairs
                for pair in predictions[["peak", "gene"]]
                .astype(str)
                .itertuples(index=False, name=None)
            ]
        )
    if {"gene", "chrom", "start", "end"}.issubset(evidence.columns):
        labels |= _interval_labels(predictions, evidence, group_columns=("gene",))
    return labels


def _label_tf_gene(predictions: pd.DataFrame, evidence: pd.DataFrame) -> np.ndarray:
    if not {"tf", "gene"}.issubset(evidence.columns):
        raise ValueError("TF-gene evidence must contain tf and gene columns.")
    positive = evidence.copy()
    if "label" in positive:
        positive = positive[pd.to_numeric(positive["label"], errors="coerce").fillna(0) > 0]
    contextual = "cell_type" in predictions and "cell_type" in positive
    keys = ["cell_type", "tf", "gene"] if contextual else ["tf", "gene"]
    pairs = set(map(tuple, positive[keys].dropna().astype(str).drop_duplicates().to_numpy()))
    return np.array(
        [
            pair in pairs
            for pair in predictions[keys].astype(str).itertuples(index=False, name=None)
        ]
    )


def _metric_row(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    score_name: str,
    scope: str = "global",
    thresholds: Sequence[float] = (0.5,),
    top_k: Sequence[int] = (10, 50, 100),
    **context: object,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        auroc = compute_auroc(labels, scores)
        auprc = compute_auprc(labels, scores)
        ratio = compute_aupr_ratio(labels, scores)
    f1 = compute_f1_at_thresholds(labels, scores, thresholds)
    top = compute_topk_precision(labels, scores, top_k)
    early = compute_early_precision(labels, scores)
    warning_text = "; ".join(dict.fromkeys(str(item.message) for item in caught))
    row: dict[str, object] = {
        "scope": scope,
        "score_name": score_name,
        "n_candidates": int(len(labels)),
        "n_positive": int(labels.sum()),
        "positive_fraction": float(labels.mean()) if len(labels) else np.nan,
        "auroc": auroc,
        "auprc": auprc,
        "random_auprc": float(labels.mean()) if len(labels) else np.nan,
        "aupr_ratio": ratio,
        "f1": float(f1.iloc[0]["f1"]),
        "f1_threshold": float(f1.iloc[0]["threshold"]),
        "early_precision": early["early_precision"],
        "early_precision_ratio": early["early_precision_ratio"],
        "warning": warning_text,
        **context,
    }
    for key, value in context.items():
        top[key] = value
        f1[key] = value
    top["scope"] = scope
    top["score_name"] = score_name
    f1["scope"] = scope
    f1["score_name"] = score_name
    return row, top, f1


def evaluate_tf_peak_links(
    tf_peak_links: pd.DataFrame,
    chipseq_peaks: GoldStandardResult | pd.DataFrame | str | None = None,
    motif_baseline: pd.DataFrame | None = None,
    *,
    score_col: str | None = None,
    species: str = "mouse",
    thresholds: Sequence[float] = (0.5,),
    top_k: Sequence[int] = (10, 50, 100),
) -> dict[str, object]:
    """Validate TF-peak links against independent binding data or motif sanity."""

    predictions = _standardize_tf_peak(tf_peak_links, score_col)
    gold = _gold_result(chipseq_peaks, load_chipseq_peaks, species=species)
    validation_type = "independent_binding_validation"
    evidence: pd.DataFrame
    if gold.available:
        evidence = gold.data
    elif motif_baseline is not None:
        evidence = _standardize_tf_peak(motif_baseline)
        validation_type = "motif_prior_sanity_check"
    else:
        return _result(
            "skipped",
            gold.reason or "independent binding and motif evidence are both unavailable",
            validation_type="none",
            metrics=pd.DataFrame(),
            per_tf=pd.DataFrame(),
            topk=pd.DataFrame(),
            thresholds=pd.DataFrame(),
            scored_edges=predictions,
        )
    predictions = predictions.copy()
    predictions["label"] = _label_tf_peak(predictions, evidence).astype(int)
    rows: list[dict[str, object]] = []
    top_tables: list[pd.DataFrame] = []
    threshold_tables: list[pd.DataFrame] = []
    row, top, f1 = _metric_row(
        predictions["label"].to_numpy(),
        predictions["score"].to_numpy(),
        score_name="tf_peak_score",
        thresholds=thresholds,
        top_k=top_k,
        validation_type=validation_type,
    )
    rows.append(row)
    top_tables.append(top)
    threshold_tables.append(f1)
    per_tf_rows: list[dict[str, object]] = []
    for tf, group in predictions.groupby("tf", sort=True):
        tf_row, tf_top, tf_f1 = _metric_row(
            group["label"].to_numpy(),
            group["score"].to_numpy(),
            score_name="tf_peak_score",
            scope="per_tf",
            thresholds=thresholds,
            top_k=top_k,
            tf=tf,
            validation_type=validation_type,
        )
        per_tf_rows.append(tf_row)
        top_tables.append(tf_top)
        threshold_tables.append(tf_f1)
    status, reason = _classification_status(predictions["label"])
    metrics = pd.DataFrame(rows)
    per_tf_metrics = pd.DataFrame(per_tf_rows)
    result = _result(
        status,
        reason,
        validation_type=validation_type,
        metrics=metrics,
        per_tf=per_tf_metrics,
        topk=pd.concat(top_tables, ignore_index=True),
        thresholds=pd.concat(threshold_tables, ignore_index=True),
        scored_edges=predictions,
    )
    result["warnings"] = _reported_warnings(
        pd.concat([metrics, per_tf_metrics], ignore_index=True, sort=False)
    )
    if validation_type == "motif_prior_sanity_check":
        result["disclaimer"] = (
            "Motif evidence is a candidate prior, not an independent TF-binding gold standard."
        )
    return result


def _distance_bin(value: object) -> str | None:
    if pd.isna(value):
        return None
    distance = abs(float(value))
    limits = (5_000, 10_000, 20_000, 50_000, 100_000, 200_000)
    for label, upper in zip(DISTANCE_BINS[:-1], limits, strict=True):
        if distance < upper:
            return label
    if distance <= 1_000_000:
        return DISTANCE_BINS[-1]
    return "outside_1Mb"


def evaluate_peak_gene_links(
    peak_gene_links: pd.DataFrame,
    peak_gene_gold: GoldStandardResult | pd.DataFrame | str | None = None,
    eqtl_links: GoldStandardResult | pd.DataFrame | str | None = None,
    *,
    score_col: str | None = None,
    species: str = "mouse",
    distance_scale: float = 25_000.0,
) -> dict[str, object]:
    """Validate peak-gene links globally and within genomic-distance bins."""

    predictions = _standardize_peak_gene(peak_gene_links, score_col)
    direct = _gold_result(peak_gene_gold, load_peak_gene_gold, species=species)
    eqtl = _gold_result(eqtl_links, load_eqtl_links, species=species)
    available = [item.data for item in (direct, eqtl) if item.available]
    if not available:
        reasons = "; ".join(filter(None, (direct.reason, eqtl.reason)))
        return _result(
            "skipped",
            reasons or "peak-gene gold standards are unavailable",
            metrics=pd.DataFrame(),
            scored_edges=predictions,
        )
    evidence = pd.concat(available, ignore_index=True, sort=False)
    predictions = predictions.copy()
    predictions["label"] = _label_peak_gene(predictions, evidence).astype(int)
    distance_col = _column(predictions, ("distance", "distance_bp"), required=False)
    if distance_col:
        predictions["distance"] = pd.to_numeric(predictions[distance_col], errors="coerce").abs()
    else:
        predictions["distance"] = np.nan
    predictions["distance_bin"] = predictions["distance"].map(_distance_bin)
    predictions["distance_only_score"] = np.exp(
        -predictions["distance"].fillna(np.inf).to_numpy(dtype=float) / float(distance_scale)
    )
    pcc_col = _column(
        predictions,
        ("pcc", "correlation", "pcc_score", "expression_accessibility_pcc"),
        required=False,
    )
    score_specs = [("model", "score"), ("distance_only", "distance_only_score")]
    if pcc_col:
        predictions["pcc_baseline_score"] = pd.to_numeric(
            predictions[pcc_col], errors="coerce"
        )
        score_specs.append(("pcc_baseline", "pcc_baseline_score"))
    rows: list[dict[str, object]] = []
    for score_name, column in score_specs:
        valid = predictions[column].notna()
        if valid.any():
            row, _, _ = _metric_row(
                predictions.loc[valid, "label"].to_numpy(),
                predictions.loc[valid, column].to_numpy(),
                score_name=score_name,
                evidence_sources=";".join(sorted(evidence["source"].dropna().astype(str).unique())),
            )
            row["distance_bin"] = "all"
            rows.append(row)
        for distance_bin in DISTANCE_BINS:
            selected = valid & predictions["distance_bin"].eq(distance_bin)
            if not selected.any():
                continue
            row, _, _ = _metric_row(
                predictions.loc[selected, "label"].to_numpy(),
                predictions.loc[selected, column].to_numpy(),
                score_name=score_name,
                scope="distance_bin",
                distance_bin=distance_bin,
                evidence_sources=";".join(sorted(evidence["source"].dropna().astype(str).unique())),
            )
            rows.append(row)
    status, reason = _classification_status(predictions["label"])
    metrics = pd.DataFrame(rows)
    result = _result(
        status,
        reason,
        metrics=metrics,
        scored_edges=predictions,
        distance_bins=list(DISTANCE_BINS),
    )
    result["warnings"] = _reported_warnings(metrics)
    return result


def evaluate_tf_gene_links(
    tf_target_egrn: pd.DataFrame,
    tf_target_gold: (
        GoldStandardResult
        | pd.DataFrame
        | str
        | Sequence[GoldStandardResult | pd.DataFrame | str]
        | None
    ) = None,
    *,
    score_col: str | None = None,
    species: str = "mouse",
    top_k: Sequence[int] = (10, 50, 100),
    thresholds: Sequence[float] = (0.5,),
) -> dict[str, object]:
    """Validate TF-target links globally, per TF, and per cell type."""

    predictions = _standardize_tf_gene(tf_target_egrn, score_col)
    gold = _gold_result(tf_target_gold, load_tf_target_gold, species=species)
    if not gold.available:
        return _result(
            "skipped",
            gold.reason or "TF-target gold standard unavailable",
            metrics=pd.DataFrame(),
            global_metrics=pd.DataFrame(),
            per_tf=pd.DataFrame(),
            per_cell_type=pd.DataFrame(),
            topk=pd.DataFrame(),
            scored_edges=predictions,
        )
    predictions = predictions.copy()
    predictions["label"] = _label_tf_gene(predictions, gold.data).astype(int)
    global_predictions = (
        predictions.groupby(["tf", "gene"], as_index=False)["score"].max()
        if "cell_type" in predictions and "cell_type" not in gold.data
        else predictions
    )
    global_predictions["label"] = _label_tf_gene(global_predictions, gold.data).astype(int)
    rows: list[dict[str, object]] = []
    top_tables: list[pd.DataFrame] = []
    row, top, _ = _metric_row(
        global_predictions["label"].to_numpy(),
        global_predictions["score"].to_numpy(),
        score_name="tf_target_score",
        top_k=top_k,
        thresholds=thresholds,
        evidence_sources=";".join(sorted(gold.data["source"].dropna().astype(str).unique())),
    )
    rows.append(row)
    top_tables.append(top)
    per_tf: list[dict[str, object]] = []
    for tf, group in global_predictions.groupby("tf", sort=True):
        tf_row, tf_top, _ = _metric_row(
            group["label"].to_numpy(),
            group["score"].to_numpy(),
            score_name="tf_target_score",
            scope="per_tf",
            top_k=top_k,
            thresholds=thresholds,
            tf=tf,
        )
        per_tf.append(tf_row)
        top_tables.append(tf_top)
    per_cell_type: list[dict[str, object]] = []
    if "cell_type" in predictions:
        for cell_type, group in predictions.groupby("cell_type", sort=True):
            cell_row, cell_top, _ = _metric_row(
                group["label"].to_numpy(),
                group["score"].to_numpy(),
                score_name="tf_target_score",
                scope="per_cell_type",
                top_k=top_k,
                thresholds=thresholds,
                cell_type=cell_type,
            )
            per_cell_type.append(cell_row)
            top_tables.append(cell_top)
    status, reason = _classification_status(global_predictions["label"])
    metrics = pd.concat(
        [
            pd.DataFrame(rows),
            pd.DataFrame(per_tf),
            pd.DataFrame(per_cell_type),
        ],
        ignore_index=True,
        sort=False,
    )
    result = _result(
        status,
        reason,
        metrics=metrics,
        global_metrics=pd.DataFrame(rows),
        per_tf=pd.DataFrame(per_tf),
        per_cell_type=pd.DataFrame(per_cell_type),
        topk=pd.concat(top_tables, ignore_index=True),
        scored_edges=predictions,
    )
    result["warnings"] = _reported_warnings(metrics)
    return result


def _as_evidence_data(
    value: GoldStandardResult | pd.DataFrame | None,
) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, GoldStandardResult):
        return value.data if value.available else None
    return value.copy()


def evaluate_regulatory_paths(
    paths: pd.DataFrame,
    tf_peak_evidence: GoldStandardResult | pd.DataFrame | None = None,
    peak_gene_evidence: GoldStandardResult | pd.DataFrame | None = None,
    tf_gene_evidence: GoldStandardResult | pd.DataFrame | None = None,
) -> dict[str, object]:
    """Annotate each TF→peak→gene path with up to three evidence layers."""

    tf_col = _column(paths, ("tf", "regulator"))
    peak_col = _column(paths, ("peak", "peak_id"))
    gene_col = _column(paths, ("gene", "target_gene", "target"))
    annotated = paths.copy().rename(columns={tf_col: "tf", peak_col: "peak", gene_col: "gene"})
    annotated["tf"] = annotated["tf"].astype(str)
    annotated["peak"] = annotated["peak"].astype(str)
    annotated["gene"] = annotated["gene"].astype(str)
    annotated = _peak_coordinates(annotated)
    tp = _as_evidence_data(tf_peak_evidence)
    pg = _as_evidence_data(peak_gene_evidence)
    tg = _as_evidence_data(tf_gene_evidence)
    if tp is None and pg is None and tg is None:
        return _result(
            "skipped",
            "no path-level evidence was provided",
            annotated_paths=annotated,
            summary=pd.DataFrame(),
            disclaimer=(
                "Composite support combines evidence layers and is not an independent gold standard."
            ),
        )
    annotated["tf_peak_supported"] = _label_tf_peak(annotated, tp) if tp is not None else False
    annotated["peak_gene_supported"] = (
        _label_peak_gene(annotated, pg) if pg is not None else False
    )
    annotated["tf_gene_supported"] = _label_tf_gene(annotated, tg) if tg is not None else False
    support_columns = ["tf_peak_supported", "peak_gene_supported", "tf_gene_supported"]
    annotated["n_evidence_layers"] = annotated[support_columns].sum(axis=1).astype(int)
    annotated["support_class"] = np.select(
        [
            annotated["n_evidence_layers"].eq(3),
            annotated["tf_peak_supported"] & annotated["peak_gene_supported"],
            annotated["tf_gene_supported"],
            annotated["n_evidence_layers"].eq(1),
        ],
        ["fully_triangulated", "both_legs", "tf_gene_only", "single_layer"],
        default="unsupported",
    )
    summary = (
        annotated.groupby(["support_class", "n_evidence_layers"], as_index=False)
        .size()
        .rename(columns={"size": "n_paths"})
    )
    summary["fraction"] = summary["n_paths"] / max(1, len(annotated))
    return _result(
        "ok",
        annotated_paths=annotated,
        summary=summary,
        disclaimer=(
            "Composite support combines non-independent evidence layers; it is not an "
            "independent gold standard and does not by itself prove causality."
        ),
    )
