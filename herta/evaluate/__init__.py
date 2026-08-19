"""Evaluation utilities for two-stage HERTA and regulatory validation."""

from herta.evaluate.clustering import clustering_metrics, leiden_clusters
from herta.evaluate.state_benchmark import (
    STATE_BENCHMARK_COLUMNS,
    benchmark_state_representations,
)
from herta.evaluate.gold import (
    GoldStandardResult,
    load_chipseq_peaks,
    load_dorothea,
    load_eqtl_links,
    load_peak_gene_gold,
    load_perturbation_targets,
    load_tf_target_gold,
    load_trrust,
)
from herta.evaluate.grn import evaluate_grn_reference, grn_stability
from herta.evaluate.metrics import (
    compute_aupr_ratio,
    compute_auprc,
    compute_auroc,
    compute_early_precision,
    compute_f1_at_thresholds,
    compute_topk_precision,
)
from herta.evaluate.regulatory import (
    DISTANCE_BINS,
    evaluate_peak_gene_links,
    evaluate_regulatory_paths,
    evaluate_tf_gene_links,
    evaluate_tf_peak_links,
)
from herta.evaluate.reports import write_validation_report

__all__ = [
    "DISTANCE_BINS",
    "GoldStandardResult",
    "clustering_metrics",
    "leiden_clusters",
    "STATE_BENCHMARK_COLUMNS",
    "benchmark_state_representations",
    "compute_aupr_ratio",
    "compute_auprc",
    "compute_auroc",
    "compute_early_precision",
    "compute_f1_at_thresholds",
    "compute_topk_precision",
    "evaluate_grn_reference",
    "evaluate_peak_gene_links",
    "evaluate_regulatory_paths",
    "evaluate_tf_gene_links",
    "evaluate_tf_peak_links",
    "grn_stability",
    "load_chipseq_peaks",
    "load_dorothea",
    "load_eqtl_links",
    "load_peak_gene_gold",
    "load_perturbation_targets",
    "load_tf_target_gold",
    "load_trrust",
    "write_validation_report",
]
