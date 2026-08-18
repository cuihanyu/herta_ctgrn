"""Full-graph toy training for HERTA-ctGRN."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from herta.data.graph_builder import FORWARD_RELATIONS
from herta.data.negative_sampling import sample_negative_targets
from herta.data.sampler import sample_positive_edges
from herta.model.herta_model import HertaModel
from herta.model.losses import weighted_infonce_loss
from herta.train.evaluate import binary_ranking_metrics


REL_CFG = {
    "ct": ("cell", "tf", "negative_ratio_ct", "tau_ct", "lambda_ct"),
    "cg": ("cell", "gene", "negative_ratio_cg", "tau_cg", "lambda_cg"),
    "cp": ("cell", "peak", "negative_ratio_cp", "tau_cp", "lambda_cp"),
    "tp": ("tf", "peak", "negative_ratio_tp", "tau_tp", "lambda_tp"),
    "pg": ("peak", "gene", "negative_ratio_pg", "tau_pg", "lambda_pg"),
}


def train_full_graph(data: HeteroData, config: dict, output_dir: str | Path) -> tuple[HertaModel, pd.DataFrame]:
    """Train HERTA on the full toy graph and save checkpoints/metrics."""

    output_dir = Path(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    model = HertaModel(
        data,
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    gen = torch.Generator().manual_seed(int(config.get("seed", 1)))
    batch_size = int(train_cfg.get("batch_size_edges_per_relation", train_cfg.get("batch_size_cells", 128)))
    rows: list[dict[str, float]] = []
    best = float("inf")
    best_state = None
    for epoch in range(1, int(train_cfg.get("epochs", 100)) + 1):
        model.train()
        opt.zero_grad()
        z = model.encode(data)
        total = torch.tensor(0.0)
        metrics: dict[str, float] = {"epoch": float(epoch)}
        for rel, edge_type in FORWARD_RELATIONS.items():
            src_type, dst_type, neg_key, tau_key, lambda_key = REL_CFG[rel]
            pos_edges, weights = sample_positive_edges(data, edge_type, batch_size, gen)
            neg_ids = sample_negative_targets(pos_edges, int(data[dst_type].num_nodes), int(train_cfg.get(neg_key, 5)), gen)
            pos_logits = model.decoders.score(rel, z[src_type], z[dst_type], pos_edges)
            src_rep = pos_edges[0].repeat_interleave(neg_ids.shape[1])
            neg_logits_flat = model.decoders.score_pairs(rel, z[src_type], z[dst_type], src_rep, neg_ids.reshape(-1))
            neg_logits = neg_logits_flat.reshape(pos_edges.shape[1], neg_ids.shape[1])
            loss = weighted_infonce_loss(pos_logits, neg_logits, weights, float(train_cfg.get(tau_key, 0.2)))
            total = total + float(train_cfg.get(lambda_key, 1.0)) * loss
            rel_metrics = binary_ranking_metrics(pos_logits, neg_logits)
            metrics[f"{rel}_loss"] = float(loss.detach())
            metrics[f"{rel}_sampled_auc"] = rel_metrics["sampled_auc"]
            metrics[f"{rel}_pairwise_accuracy"] = rel_metrics["pairwise_accuracy"]
        total.backward()
        opt.step()
        metrics["loss"] = float(total.detach())
        rows.append(metrics)
        if metrics["loss"] < best:
            best = metrics["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"model_state": model.state_dict(), "config": config}, output_dir / "checkpoints" / "last_model.pt")
    if best_state is not None:
        torch.save({"model_state": best_state, "config": config}, output_dir / "checkpoints" / "best_model.pt")
        model.load_state_dict(best_state)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "logs" / "train_metrics.csv", index=False)
    return model, df
