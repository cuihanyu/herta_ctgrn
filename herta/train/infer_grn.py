"""Cluster-conditioned TF-to-peak-to-gene inference for Stage 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from herta.data.regulatory_graph import RegulatoryGraphBuildResult
from herta.model.grn_model import GRNModel
from herta.model.regulatory_scoring import (
    DEFAULT_AGGREGATION_MODE,
    aggregate_numpy,
    canonical_aggregation_mode,
    path_score_numpy,
)
from herta.train.cluster_cells import cluster_cells
from herta.train.egrn import (
    RegulatoryStateConfig,
    build_regulatory_state,
    write_regulatory_state_outputs,
)


def _cluster_mean(matrix: object, cell_ids: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    subset = matrix[cell_ids]
    if weights is None:
        return np.asarray(subset.mean(axis=0)).ravel()
    denominator = float(weights.sum()) + 1e-8
    return np.asarray(subset.T @ weights).ravel() / denominator


def _unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    minimum = float(values.min()) if len(values) else 0.0
    maximum = float(values.max()) if len(values) else 0.0
    return ((values - minimum) / (maximum - minimum + 1e-8)).astype(np.float32)


def _selected_activity_frame(
    matrix: object,
    feature_names: list[str],
    selected_names: list[str],
    cell_names: list[str],
) -> pd.DataFrame:
    """Materialize only regulatory features, avoiding a full dense matrix."""

    feature_index = {str(name): index for index, name in enumerate(feature_names)}
    missing = set(selected_names).difference(feature_index)
    if missing:
        raise ValueError(
            f"Regulatory-state activity is missing features: {sorted(missing)[:5]}"
        )
    indices = [feature_index[name] for name in selected_names]
    selected = matrix[:, indices]
    values = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
    return pd.DataFrame(
        values,
        index=pd.Index(cell_names, name="cell_id"),
        columns=selected_names,
        dtype=float,
    )


def score_regulatory_candidates(
    model: GRNModel,
    embeddings: dict[str, torch.Tensor],
    build: RegulatoryGraphBuildResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return trained TF-peak and peak-gene scores as inspectable tables."""

    gene_index = {name: idx for idx, name in enumerate(build.state.names["gene"])}
    peak_index = {name: idx for idx, name in enumerate(build.state.names["peak"])}
    tp = build.candidates["tp"].copy().reset_index(drop=True)
    pg = build.candidates["pg"].copy().reset_index(drop=True)
    tp_edge = torch.tensor(
        [[gene_index[str(x)] for x in tp["tf"]], [peak_index[str(x)] for x in tp["peak"]]],
        dtype=torch.long,
    )
    pg_edge = torch.tensor(
        [[peak_index[str(x)] for x in pg["peak"]], [gene_index[str(x)] for x in pg["gene"]]],
        dtype=torch.long,
    )
    with torch.no_grad():
        predictions = model.decode(
            embeddings,
            query_tensors=build.query_tensors,
            tf_peak_edge_index=tp_edge,
            peak_gene_edge_index=pg_edge,
            is_tf=build.is_tf,
            aggregate_tf_gene=False,
        )
        tp["score"] = predictions.tf_peak_scores.cpu().numpy()
        pg["score"] = predictions.peak_gene_scores.cpu().numpy()
        tp["logit"] = predictions.tf_peak_logits.cpu().numpy()
        pg["logit"] = predictions.peak_gene_logits.cpu().numpy()
    tp["motif_hit"] = 1
    if "dist" in pg:
        pg["distance_bp"] = pd.to_numeric(pg["dist"], errors="raise")
    elif "distance" in pg:
        pg["distance_bp"] = pd.to_numeric(pg["distance"], errors="raise")
    if "weight" in pg:
        pg["distance_weight"] = pd.to_numeric(pg["weight"], errors="raise")
    for table in (tp, pg):
        table["score_semantics"] = "shared_regulatory_backbone_compatibility"
        table["calibrated"] = False
    return tp, pg


# Backward-compatible private alias retained for older tests and callers.
_score_candidates = score_regulatory_candidates


def infer_shared_backbone(
    model: GRNModel,
    build: RegulatoryGraphBuildResult,
    config: dict,
    output_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
) -> dict[str, object]:
    """Write only the current shared TP/PG regulatory-backbone outputs."""

    output_dir = Path(output_dir)
    preprocessing_dir = output_dir / "preprocessing"
    candidate_dir = output_dir / "candidates"
    score_dir = output_dir / "edge_scores"
    for directory in (preprocessing_dir, candidate_dir, score_dir):
        directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(build.data, build.state.features)
    tp, pg = score_regulatory_candidates(model, embeddings, build)

    genes = build.state.metadata["genes"].copy()
    peaks = build.state.metadata["peaks"].copy()
    tfs = (
        genes.loc[genes["is_tf"].astype(bool)].copy()
        if "is_tf" in genes
        else genes.loc[genes["gene"].astype(str).isin(build.state.metadata["tf_names"])].copy()
    )
    candidate_tp = pd.DataFrame(
        {"tf": tp["tf"].astype(str), "peak": tp["peak"].astype(str), "motif_hit": 1}
    )
    candidate_pg = pd.DataFrame(
        {
            "peak": pg["peak"].astype(str),
            "gene": pg["gene"].astype(str),
            "distance_bp": pd.to_numeric(pg.get("dist", pg.get("distance"))),
            "distance_weight": pd.to_numeric(pg["weight"]),
        }
    )
    predicted_tp = tp[["tf", "peak", "score"]].copy()
    predicted_pg = pg[["peak", "gene", "score"]].copy()

    genes.to_parquet(preprocessing_dir / "genes.parquet", index=False)
    tfs.to_parquet(preprocessing_dir / "tfs.parquet", index=False)
    peaks.to_parquet(preprocessing_dir / "peaks.parquet", index=False)
    build.edge_table_bundle.tf_peak.to_parquet(
        candidate_dir / "tf_peak_candidates.parquet", index=False
    )
    build.edge_table_bundle.peak_gene.to_parquet(
        candidate_dir / "peak_gene_candidates.parquet", index=False
    )
    tp.to_parquet(score_dir / "tf_peak_scores.parquet", index=False)
    pg.to_parquet(score_dir / "peak_gene_scores.parquet", index=False)
    genes.to_csv(output_dir / "genes.tsv", sep="\t", index=False)
    tfs.to_csv(output_dir / "tfs.tsv", sep="\t", index=False)
    peaks.to_csv(output_dir / "peaks.tsv", sep="\t", index=False)
    candidate_tp.to_csv(output_dir / "candidate_tf_peak.tsv", sep="\t", index=False)
    candidate_pg.to_csv(output_dir / "candidate_peak_gene.tsv", sep="\t", index=False)
    predicted_tp.to_csv(output_dir / "predicted_tf_peak.tsv", sep="\t", index=False)
    predicted_pg.to_csv(output_dir / "predicted_peak_gene.tsv", sep="\t", index=False)

    manifest = {
        "profile": "shared_backbone_minimal_v1",
        "seed": int(config.get("seed", 1)),
        "resolved_config": config,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "node_counts": {
            node_type: int(build.data[node_type].num_nodes)
            for node_type in ("cell", "gene", "peak")
        },
        "candidate_counts": {"tf_peak": int(len(tp)), "peak_gene": int(len(pg))},
        "split_policy": "deterministic_peak_grouped_80_10_10",
        "candidate_truncation": {"tf_peak": False, "peak_gene": False},
        "prior_message": False,
        "score_semantics": "shared regulatory-backbone compatibility score",
        "calibrated": False,
        "cell_type_specific": False,
        "regulatory_state_output": False,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, default=str)
    return {"tf_peak": tp, "peak_gene": pg, "manifest": manifest}


def infer_cluster_grn(
    model: GRNModel,
    build: RegulatoryGraphBuildResult,
    clusters: pd.DataFrame | str | Path,
    config: dict,
    output_dir: str | Path,
) -> dict[str, object]:
    """Apply cluster RNA/ATAC context to shared learned TP/PG scores."""

    output_dir = Path(output_dir)
    (output_dir / "edge_scores").mkdir(parents=True, exist_ok=True)
    (output_dir / "grn").mkdir(parents=True, exist_ok=True)
    assignments = pd.read_parquet(clusters) if isinstance(clusters, str | Path) else clusters.copy()
    required = {"cell_id", "cluster"}
    if missing := required.difference(assignments.columns):
        raise ValueError(f"clusters are missing required columns: {sorted(missing)}")
    cell_index = {name: idx for idx, name in enumerate(build.state.names["cell"])}
    assignments = assignments[assignments["cell_id"].astype(str).isin(cell_index)].copy()
    assignments["cell_index"] = assignments["cell_id"].astype(str).map(cell_index).astype(int)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(build.data, build.state.features)
    tp, pg = _score_candidates(model, embeddings, build)
    tp.to_parquet(output_dir / "edge_scores" / "tf_peak_scores.parquet", index=False)
    pg.to_parquet(output_dir / "edge_scores" / "peak_gene_scores.parquet", index=False)

    paths = tp[["tf", "peak", "score"]].rename(columns={"score": "tf_peak_score"}).merge(
        pg[["peak", "gene", "score"]].rename(columns={"score": "peak_gene_score"}),
        on="peak",
        how="inner",
    )
    gene_index = {name: idx for idx, name in enumerate(build.state.names["gene"])}
    peak_index = {name: idx for idx, name in enumerate(build.state.names["peak"])}
    inference_cfg = config.get("grn_inference", config.get("inference", {}))
    use_target_gate = bool(inference_cfg.get("use_target_expression_gate", False))
    aggregation_mode = canonical_aggregation_mode(
        str(
            inference_cfg.get(
                "aggregation_mode",
                config.get("model", {}).get(
                    "aggregator_mode",
                    DEFAULT_AGGREGATION_MODE,
                ),
            )
        )
    )
    aggregation_top_k = int(inference_cfg.get("aggregation_top_k", 10))
    if aggregation_top_k <= 0:
        raise ValueError("aggregation_top_k must be positive.")
    soft_columns = [column for column in assignments if column.startswith("q_")]
    if soft_columns:
        contexts = [
            (
                column[2:],
                assignments["cell_index"].to_numpy(dtype=int),
                assignments[column].to_numpy(dtype=float),
            )
            for column in soft_columns
        ]
    else:
        contexts = [
            (str(cluster), members["cell_index"].to_numpy(dtype=int), None)
            for cluster, members in assignments.groupby("cluster", sort=True)
        ]

    path_rows: list[pd.DataFrame] = []
    for cluster, member_ids, member_weights in contexts:
        rna_mean = _unit_scale(_cluster_mean(build.state.factors.rna_norm, member_ids, member_weights))
        atac_mean = _unit_scale(_cluster_mean(build.state.factors.atac_tfidf, member_ids, member_weights))
        current = paths.copy()
        current["tf_activity"] = current["tf"].map(lambda name: rna_mean[gene_index[str(name)]])
        current["peak_accessibility"] = current["peak"].map(lambda name: atac_mean[peak_index[str(name)]])
        context_factors = [
            current["tf_activity"].to_numpy(dtype=float),
            current["peak_accessibility"].to_numpy(dtype=float),
        ]
        if use_target_gate:
            current["target_expression"] = current["gene"].map(lambda name: rna_mean[gene_index[str(name)]])
            context_factors.append(
                current["target_expression"].to_numpy(dtype=float)
            )
        current["path_score"] = path_score_numpy(
            current["tf_peak_score"].to_numpy(dtype=float),
            current["peak_gene_score"].to_numpy(dtype=float),
            *context_factors,
        )
        current["cluster"] = str(cluster)
        current["aggregation_mode"] = aggregation_mode
        path_rows.append(current)
    path_scores = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()

    rows: list[dict[str, object]] = []
    if not path_scores.empty:
        for (cluster, tf, gene), group in path_scores.groupby(["cluster", "tf", "gene"], sort=False):
            ranked = group.sort_values("path_score", ascending=False)
            supporting_peaks = ";".join(
                ranked["peak"]
                .astype(str)
                .drop_duplicates()
            )
            egrn_score = float(
                aggregate_numpy(
                    group["path_score"].to_numpy(dtype=float),
                    aggregation_mode,
                    top_k=aggregation_top_k,
                )
            )
            rows.append(
                {
                    "cluster": cluster,
                    "tf": tf,
                    "target_gene": gene,
                    "egrn_score": egrn_score,
                    "n_supporting_peaks": int(group["peak"].nunique()),
                    "supporting_peaks": supporting_peaks,
                    "tf_peak_score_mean": float(group["tf_peak_score"].mean()),
                    "peak_gene_score_mean": float(
                        group["peak_gene_score"].mean()
                    ),
                    "aggregation_mode": aggregation_mode,
                    # Backward-compatible aliases.
                    "gene": gene,
                    "score": egrn_score,
                    "n_candidate_peaks": int(group["peak"].nunique()),
                    "top_mediator_peaks": ";".join(
                        supporting_peaks.split(";")[:10]
                    ),
                }
            )
    tf_gene = pd.DataFrame(
        rows,
        columns=[
            "cluster",
            "tf",
            "target_gene",
            "egrn_score",
            "n_supporting_peaks",
            "supporting_peaks",
            "tf_peak_score_mean",
            "peak_gene_score_mean",
            "aggregation_mode",
            "gene",
            "score",
            "n_candidate_peaks",
            "top_mediator_peaks",
        ],
    )
    if "cell_type" in assignments and not tf_gene.empty:
        annotations = (
            assignments.groupby("cluster")["cell_type"]
            .agg(lambda values: values.astype(str).value_counts().index[0])
            .astype(str)
        )
        tf_gene["cell_type"] = tf_gene["cluster"].map(annotations)
    tf_gene = tf_gene.sort_values(["cluster", "score"], ascending=[True, False]).reset_index(drop=True)
    tf_gene.to_parquet(output_dir / "grn" / "cluster_specific_tf_gene_scores.parquet", index=False)

    top_edges = tf_gene.groupby("cluster", group_keys=False).head(int(inference_cfg.get("top_k_edges_per_cluster", 10000))).reset_index(drop=True)
    top_edges.to_parquet(output_dir / "grn" / "cluster_grn_top_edges.parquet", index=False)
    top_targets = int(inference_cfg.get("top_k_targets_per_tf", 100))
    key_tfs = (
        tf_gene.groupby(["cluster", "tf"], group_keys=False).head(top_targets)
        .groupby(["cluster", "tf"], as_index=False)["score"].sum()
        .rename(columns={"score": "key_tf_score"})
        .sort_values(["cluster", "key_tf_score"], ascending=[True, False])
    )
    key_tfs.to_parquet(output_dir / "grn" / "key_tfs_by_cluster.parquet", index=False)

    regulatory_cfg = dict(config.get("regulatory_state", {}))
    state_config_keys = set(RegulatoryStateConfig.__dataclass_fields__)
    state_settings = {
        key: value
        for key, value in regulatory_cfg.items()
        if key in state_config_keys
    }
    state_settings.setdefault("mediator_aggregation", aggregation_mode)
    state_settings.setdefault("aggregation_top_k", aggregation_top_k)
    state_config = RegulatoryStateConfig.from_mapping(state_settings)
    cell_names = list(map(str, build.state.names["cell"]))
    tf_names = list(dict.fromkeys(tp["tf"].astype(str)))
    peak_names = list(dict.fromkeys(paths["peak"].astype(str)))
    target_names = list(dict.fromkeys(paths["gene"].astype(str)))
    tf_activity = _selected_activity_frame(
        build.state.factors.rna_norm,
        list(map(str, build.state.names["gene"])),
        tf_names,
        cell_names,
    )
    peak_accessibility = _selected_activity_frame(
        build.state.factors.atac_tfidf,
        list(map(str, build.state.names["peak"])),
        peak_names,
        cell_names,
    )
    target_activity = (
        _selected_activity_frame(
            build.state.factors.rna_norm,
            list(map(str, build.state.names["gene"])),
            target_names,
            cell_names,
        )
        if state_config.use_target_gene_activity
        else None
    )
    regulatory_state = build_regulatory_state(
        tp,
        pg,
        tf_activity,
        peak_accessibility,
        target_gene_activity=target_activity,
        config=state_config,
    )
    regulatory_state.metadata.update(
        {
            "post_training_output": True,
            "tf_peak_score_source": "trained_TFPeakDecoder",
            "peak_gene_score_source": "trained_PeakGeneDecoder",
            "n_inferred_paths": int(len(paths)),
            "baseline_clustering_input": "z_cell",
            "primary_clustering_input": "regulatory_state",
        }
    )
    regulatory_clusters: pd.DataFrame | None = None
    if bool(regulatory_cfg.get("cluster", True)):
        if regulatory_state.matrix.shape[1]:
            annotation = None
            if "cell_type" in assignments:
                annotation_map = assignments.set_index(
                    assignments["cell_id"].astype(str)
                )["cell_type"]
                annotation = pd.Series(cell_names).map(annotation_map).to_numpy()
            regulatory_clusters = cluster_cells(
                regulatory_state,
                output_dir,
                config,
                cell_types=annotation,
                representation_mode="regulatory_state",
            )
            regulatory_state.metadata["clustering_status"] = "completed"
        else:
            regulatory_state.metadata["clustering_status"] = (
                "skipped_no_selected_tfs"
            )
    else:
        regulatory_state.metadata["clustering_status"] = "disabled"
    regulatory_state_paths = write_regulatory_state_outputs(
        regulatory_state,
        output_dir / "regulatory_state",
    )
    return {
        "tf_peak": tp,
        "peak_gene": pg,
        "paths": path_scores,
        "tf_gene": tf_gene,
        "top_edges": top_edges,
        "key_tfs": key_tfs,
        "regulatory_state": regulatory_state,
        "regulatory_state_paths": regulatory_state_paths,
        "regulatory_state_clusters": regulatory_clusters,
    }
