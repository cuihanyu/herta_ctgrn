"""Path-based GRN inference and output writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from herta.data.graph_builder import FORWARD_RELATIONS, GraphBuildResult
from herta.model.herta_model import HertaModel


def _score_edge_frame(model: HertaModel, z: dict[str, torch.Tensor], rel: str, edge_type: tuple[str, str, str], data: HeteroData) -> pd.DataFrame:
    src_type, _, dst_type = edge_type
    edge_index = data[edge_type].edge_index
    with torch.no_grad():
        logits = model.decoders.score(rel, z[src_type], z[dst_type], edge_index)
        prob = torch.sigmoid(logits).cpu().numpy()
    return pd.DataFrame({"src": edge_index[0].cpu().numpy(), "dst": edge_index[1].cpu().numpy(), "score": prob})


def write_embeddings(z: dict[str, torch.Tensor], names: dict[str, list[str]], output_dir: str | Path) -> None:
    """Write node embeddings as parquet."""

    out = Path(output_dir) / "node_embeddings"
    out.mkdir(parents=True, exist_ok=True)
    for node_type, emb in z.items():
        df = pd.DataFrame(emb.detach().cpu().numpy())
        df.insert(0, "name", names[node_type])
        df.to_parquet(out / f"{node_type}_embeddings.parquet", index=False)


def infer_grn(model: HertaModel, build: GraphBuildResult, config: dict, output_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Infer cell-type-specific TF-gene GRNs through TF -> peak -> gene paths."""

    output_dir = Path(output_dir)
    (output_dir / "edge_scores").mkdir(parents=True, exist_ok=True)
    (output_dir / "grn").mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        z = model.encode(build.data)
    write_embeddings(z, build.names, output_dir)

    scores = {rel: _score_edge_frame(model, z, rel, et, build.data) for rel, et in FORWARD_RELATIONS.items()}
    file_names = {"ct": "cell_tf_scores", "cp": "cell_peak_scores", "tp": "tf_peak_scores", "pg": "peak_gene_scores", "cg": "cell_gene_scores"}
    for rel, df in scores.items():
        df.to_parquet(output_dir / "edge_scores" / f"{file_names[rel]}.parquet", index=False)

    tp_by_tf: dict[int, list[tuple[int, float]]] = {}
    pg_by_peak: dict[int, list[tuple[int, float]]] = {}
    for row in scores["tp"].itertuples(index=False):
        tp_by_tf.setdefault(int(row.src), []).append((int(row.dst), float(row.score)))
    for row in scores["pg"].itertuples(index=False):
        pg_by_peak.setdefault(int(row.src), []).append((int(row.dst), float(row.score)))

    cell_types = np.asarray(build.metadata["cell_types"]).astype(str)
    infer_cfg = config.get("inference", {})
    top_k = int(infer_cfg.get("top_k_edges_per_cell_type", 10000))
    score_scale = float(infer_cfg.get("score_scale", 6.0))
    mediator_peak_bonus = float(infer_cfg.get("mediator_peak_bonus", 0.25))
    tf_gene_rows: list[dict[str, object]] = []
    for z_name in sorted(set(cell_types)):
        for tf_id, peaks in tp_by_tf.items():
            for peak_id, tp_score in peaks:
                pg_edges = pg_by_peak.get(peak_id, [])
                if not pg_edges:
                    continue
                base = float(tp_score)
                if base <= 0:
                    continue
                for gene_id, pg_score in pg_edges:
                    tf_gene_rows.append(
                        {
                            "cell_type": z_name,
                            "tf": build.names["tf"][tf_id],
                            "gene": build.names["gene"][gene_id],
                            "raw_path_score": base * pg_score,
                            "mediator_peak": build.names["peak"][peak_id],
                        }
                    )
    tf_gene = pd.DataFrame(tf_gene_rows)
    if tf_gene.empty:
        tf_gene = pd.DataFrame(columns=["cell_type", "tf", "gene", "score", "raw_score", "n_mediator_peaks", "mediator_peaks"])
    else:
        tf_gene = (
            tf_gene.groupby(["cell_type", "tf", "gene"], as_index=False)
            .agg(
                raw_score=("raw_path_score", "sum"),
                n_mediator_peaks=("mediator_peak", "nunique"),
                mediator_peaks=("mediator_peak", lambda x: ",".join(sorted(set(map(str, x))))),
            )
        )
        support_bonus = 1.0 + mediator_peak_bonus * np.log1p(tf_gene["n_mediator_peaks"].to_numpy(dtype=float))
        tf_gene["score"] = (1.0 - np.exp(-score_scale * tf_gene["raw_score"].to_numpy(dtype=float))) * support_bonus
    tf_gene = tf_gene.sort_values("score", ascending=False)
    tf_gene.to_parquet(output_dir / "edge_scores" / "tf_gene_scores_by_cell_type.parquet", index=False)
    top_edges = tf_gene.groupby("cell_type", group_keys=False).head(top_k).reset_index(drop=True)
    top_edges.to_parquet(output_dir / "grn" / "cell_type_grn_top_edges.parquet", index=False)

    top_targets = int(config.get("inference", {}).get("top_k_targets_per_tf", 100))
    key_tf = (
        tf_gene.groupby(["cell_type", "tf"], group_keys=False)
        .head(top_targets)
        .groupby(["cell_type", "tf"], as_index=False)["score"]
        .sum()
        .rename(columns={"score": "key_tf_score"})
        .sort_values("key_tf_score", ascending=False)
    )
    key_tf.to_parquet(output_dir / "grn" / "key_tfs_by_cell_type.parquet", index=False)
    return {"tf_gene": tf_gene, "top_edges": top_edges, "key_tfs": key_tf, **scores}
