"""Chunked inference for the cell-conditioned HERTA Stage 2 Lite model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from herta.data.regulatory_graph import RegulatoryGraphBuildResult
from herta.model.cell_conditioned_decoders import Stage2CellConditionedModel


@dataclass(frozen=True)
class Stage2LiteInferenceResult:
    tf_peak_scores: pd.DataFrame
    peak_gene_scores: pd.DataFrame
    paths: pd.DataFrame
    cell_type_egrn: pd.DataFrame


def _complete_candidate_table(
    table: pd.DataFrame,
    maximum: int | None,
    relation: str,
) -> pd.DataFrame:
    """Return the full candidate universe without prior-ranked truncation."""

    if maximum is not None:
        raise ValueError(
            f"Global {relation} candidate truncation is disabled because it drops "
            "feature coverage. Set the corresponding max candidate option to None "
            "and control memory with cell_chunk_size/edge_chunk_size."
        )
    return table.reset_index(drop=True)


def _peak_stratified_candidate_table(
    table: pd.DataFrame,
    candidates_per_peak: int | None,
) -> pd.DataFrame:
    """Apply a local per-peak quota while preserving every candidate peak."""

    if candidates_per_peak is None:
        return table.reset_index(drop=True)
    if candidates_per_peak <= 0:
        raise ValueError("tp_candidates_per_peak must be positive or None.")
    score = "weight" if "weight" in table else "prior_score"
    return (
        table.sort_values(["peak", score], ascending=[True, False], kind="stable")
        .groupby("peak", sort=False, observed=True, group_keys=False)
        .head(candidates_per_peak)
        .reset_index(drop=True)
    )


def _score_relation(
    model: Stage2CellConditionedModel,
    embeddings: dict[str, torch.Tensor],
    names: dict[str, list[str]],
    candidates: pd.DataFrame,
    cell_ids: np.ndarray,
    relation: str,
    cell_chunk_size: int,
    edge_chunk_size: int,
) -> pd.DataFrame:
    gene_index = {name: i for i, name in enumerate(names["gene"])}
    peak_index = {name: i for i, name in enumerate(names["peak"])}
    cell_names = np.asarray(names["cell"], dtype=object)
    rows: list[pd.DataFrame] = []
    model.eval()
    with torch.no_grad():
        for c_start in range(0, len(cell_ids), cell_chunk_size):
            current_cells = cell_ids[c_start : c_start + cell_chunk_size]
            for e_start in range(0, len(candidates), edge_chunk_size):
                edge = candidates.iloc[e_start : e_start + edge_chunk_size]
                n_cells, n_edges = len(current_cells), len(edge)
                repeated_cells = torch.as_tensor(
                    np.repeat(current_cells, n_edges), dtype=torch.long
                )
                if relation == "tp":
                    source = torch.as_tensor(
                        np.tile(edge["tf"].astype(str).map(gene_index).to_numpy(), n_cells),
                        dtype=torch.long,
                    )
                    target = torch.as_tensor(
                        np.tile(edge["peak"].astype(str).map(peak_index).to_numpy(), n_cells),
                        dtype=torch.long,
                    )
                    logits, scores = model.score_tf_peak(
                        embeddings["cell"][repeated_cells],
                        embeddings["gene"][source],
                        embeddings["peak"][target],
                    )
                    frame = pd.DataFrame(
                        {
                            "cell_id": np.repeat(cell_names[current_cells], n_edges),
                            "tf": np.tile(edge["tf"].astype(str), n_cells),
                            "peak": np.tile(edge["peak"].astype(str), n_cells),
                            "tf_peak_logit": logits.numpy(),
                            "tf_peak_score": scores.numpy(),
                        }
                    )
                else:
                    source = torch.as_tensor(
                        np.tile(edge["peak"].astype(str).map(peak_index).to_numpy(), n_cells),
                        dtype=torch.long,
                    )
                    target = torch.as_tensor(
                        np.tile(edge["gene"].astype(str).map(gene_index).to_numpy(), n_cells),
                        dtype=torch.long,
                    )
                    logits, scores = model.score_peak_gene(
                        embeddings["cell"][repeated_cells],
                        embeddings["peak"][source],
                        embeddings["gene"][target],
                    )
                    frame = pd.DataFrame(
                        {
                            "cell_id": np.repeat(cell_names[current_cells], n_edges),
                            "peak": np.tile(edge["peak"].astype(str), n_cells),
                            "target_gene": np.tile(edge["gene"].astype(str), n_cells),
                            "peak_gene_logit": logits.numpy(),
                            "peak_gene_score": scores.numpy(),
                        }
                    )
                rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def infer_stage2_lite(
    model: Stage2CellConditionedModel,
    build: RegulatoryGraphBuildResult,
    state_embeddings: dict[str, torch.Tensor],
    cell_types: Sequence[str],
    config: dict | None = None,
    output_dir: str | Path | None = None,
) -> Stage2LiteInferenceResult:
    """Score cell-specific candidate relations and aggregate traceable eGRNs."""

    cfg = {} if config is None else config.get("stage2_lite_inference", config)
    embeddings = {name: value.detach().cpu().float() for name, value in state_embeddings.items()}
    labels = np.asarray(cell_types).astype(str)
    n_cells = len(build.state.names["cell"])
    if len(labels) != n_cells:
        raise ValueError("cell_types must align with Stage-1 cell embeddings.")
    cell_ids = np.arange(n_cells, dtype=int)
    tp = _complete_candidate_table(
        build.candidates["tp"], cfg.get("max_tp_candidates"), "TF-peak"
    )
    tp = _peak_stratified_candidate_table(
        tp, cfg.get("tp_candidates_per_peak", 1)
    )
    pg = _complete_candidate_table(
        build.candidates["pg"], cfg.get("max_pg_candidates"), "peak-gene"
    )
    tp_scores = _score_relation(
        model, embeddings, build.state.names, tp, cell_ids, "tp",
        int(cfg.get("cell_chunk_size", 32)), int(cfg.get("edge_chunk_size", 4096)),
    )
    pg_scores = _score_relation(
        model, embeddings, build.state.names, pg, cell_ids, "pg",
        int(cfg.get("cell_chunk_size", 32)), int(cfg.get("edge_chunk_size", 4096)),
    )
    label_map = dict(zip(build.state.names["cell"], labels, strict=True))
    tp_scores["cell_type"] = tp_scores["cell_id"].map(label_map)
    pg_scores["cell_type"] = pg_scores["cell_id"].map(label_map)
    paths = tp_scores.merge(pg_scores, on=["cell_id", "cell_type", "peak"], how="inner")
    if paths.empty:
        raise ValueError("Scored TF-peak and peak-gene candidates share no complete paths.")
    paths["path_score"] = paths["tf_peak_score"] * paths["peak_gene_score"]
    type_paths = (
        paths.groupby(["cell_type", "tf", "peak", "target_gene"], observed=True, as_index=False)
        .agg(
            path_score=("path_score", "mean"),
            tf_peak_score=("tf_peak_score", "mean"),
            peak_gene_score=("peak_gene_score", "mean"),
            n_cells=("cell_id", "nunique"),
        )
    )
    top_k = int(cfg.get("mediator_top_k", 10))
    selected = (
        type_paths.sort_values("path_score", ascending=False)
        .groupby(["cell_type", "tf", "target_gene"], observed=True, group_keys=False)
        .head(top_k)
    )
    egrn = (
        selected.groupby(["cell_type", "tf", "target_gene"], observed=True, as_index=False)
        .agg(
            egrn_score=("path_score", "mean"),
            n_supporting_peaks=("peak", "nunique"),
            supporting_peaks=("peak", lambda values: ";".join(map(str, values))),
            top_path_score=("path_score", "max"),
            mean_tf_peak_score=("tf_peak_score", "mean"),
            mean_peak_gene_score=("peak_gene_score", "mean"),
        )
        .sort_values(["cell_type", "egrn_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    result = Stage2LiteInferenceResult(tp_scores, pg_scores, paths, egrn)
    if output_dir is not None:
        target = Path(output_dir) / "stage2_lite"
        target.mkdir(parents=True, exist_ok=True)
        tp_scores.to_parquet(target / "cell_tf_peak_scores.parquet", index=False)
        pg_scores.to_parquet(target / "cell_peak_gene_scores.parquet", index=False)
        paths.to_parquet(target / "cell_tf_peak_gene_paths.parquet", index=False)
        egrn.to_csv(target / "cell_type_egrn.csv", index=False)
    return result
