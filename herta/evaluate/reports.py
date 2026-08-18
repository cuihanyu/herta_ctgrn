"""Writers for layered HERTA regulatory-validation reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _table(result: dict[str, Any] | None, key: str) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame()
    value = result.get(key)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _summary_row(name: str, result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {"module": name, "status": "skipped", "reason": "result was not provided"}
    row: dict[str, Any] = {
        "module": name,
        "status": result.get("status", "unclear"),
        "reason": result.get("reason"),
    }
    metrics = _table(result, "global_metrics")
    if metrics.empty:
        metrics = _table(result, "metrics")
    if metrics.empty:
        metrics = _table(result, "summary")
    if not metrics.empty:
        first = metrics.iloc[0]
        for column in (
            "n_candidates",
            "n_positive",
            "positive_fraction",
            "auroc",
            "auprc",
            "random_auprc",
            "aupr_ratio",
            "f1",
            "early_precision",
            "early_precision_ratio",
        ):
            if column in first:
                row[column] = first[column]
    return row


def _display_metric(value: Any) -> str:
    return "" if pd.isna(value) else f"{float(value):.6g}"


def write_validation_report(
    output_dir: str | Path,
    *,
    tf_peak: dict[str, Any] | None = None,
    peak_gene: dict[str, Any] | None = None,
    tf_gene: dict[str, Any] | None = None,
    paths: dict[str, Any] | None = None,
    title: str = "HERTA regulatory validation report",
) -> dict[str, Path]:
    """Write standard CSV tables and a concise Markdown status report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            _summary_row("tf_peak", tf_peak),
            _summary_row("peak_gene", peak_gene),
            _summary_row("tf_gene", tf_gene),
            _summary_row("regulatory_paths", paths),
        ]
    )
    tables = {
        "validation_summary": summary,
        "tf_peak_metrics": pd.concat(
            [_table(tf_peak, "metrics"), _table(tf_peak, "per_tf")],
            ignore_index=True,
            sort=False,
        ).drop_duplicates(),
        "peak_gene_metrics": _table(peak_gene, "metrics"),
        "tf_gene_metrics": _table(tf_gene, "metrics"),
        "path_support_summary": _table(paths, "summary"),
    }
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path

    lines = [
        f"# {title}",
        "",
        "## Validation status",
        "",
        "| Layer | Status | AUROC | AUPRC | AUPR ratio | EPR | Reason |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        reason = "" if pd.isna(row.reason) else str(row.reason).replace("|", "\\|")
        lines.append(
            f"| {row.module} | {row.status} | {_display_metric(getattr(row, 'auroc', None))} "
            f"| {_display_metric(getattr(row, 'auprc', None))} "
            f"| {_display_metric(getattr(row, 'aupr_ratio', None))} "
            f"| {_display_metric(getattr(row, 'early_precision_ratio', None))} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Missing optional gold standards are reported as `skipped`; no labels are fabricated.",
            "- Motif-only TF-peak evaluation is a prior sanity check, not independent binding validation.",
            "- Composite path support combines evidence layers and is not an independent gold standard.",
            "- Unknown or unrecorded edges are candidate background, not proven biological negatives.",
            "",
            "## Output tables",
            "",
        ]
    )
    for name, path in written.items():
        lines.append(f"- `{path.name}`: {len(tables[name])} rows")
    report = output / "VALIDATION_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written["report"] = report
    return written
