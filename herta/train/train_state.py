"""Stage-1 cell-state training and output writing."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
import torch

from herta.data.negative_sampling import sample_negative_targets
from herta.data.sampler import sample_positive_edges, sample_positive_edges_by_source
from herta.data.sampling import CellKNNSampler, SamplerConfig, SubgraphBatch
from herta.data.state_graph import STATE_RELATIONS, StateGraphBuildResult
from herta.model.losses import (
    Stage1LossConfig,
    weighted_infonce_loss,
)
from herta.model.state_model import StateModel
from herta.utils.checkpoint import save_checkpoint
from herta.utils.checkpoint import load_checkpoint
from herta.utils.seed import set_seed


STATE_REL_CFG = {
    "cg": ("cell", "gene", "negative_ratio_cg", "temperature_cg"),
    "cp": ("cell", "peak", "negative_ratio_cp", "temperature_cp"),
}


def _edge_checksum(edge_index: torch.Tensor) -> int:
    """Return a deterministic diagnostic checksum for a sampled edge tensor."""

    if edge_index.numel() == 0:
        return 0
    values = edge_index.detach().cpu().to(torch.int64)
    return int(((values[0] + 1) * 1_000_003 + values[1] + 1).sum())


def _seed_torch_dropout(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _subgraph_features(
    build: StateGraphBuildResult,
    batch: SubgraphBatch,
) -> dict[str, torch.Tensor]:
    """Select node-feature rows in the sampled local/global ID order."""

    return {
        node_type: build.features[node_type][batch.global_node_ids[node_type]]
        for node_type in batch.data.node_types
    }


def _global_edge_index(
    edge_index: torch.Tensor,
    source_type: str,
    target_type: str,
    global_node_ids: Mapping[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Map a local sampled edge tensor back to auditable global node IDs."""

    if global_node_ids is None:
        return edge_index
    return torch.stack(
        [
            global_node_ids[source_type][edge_index[0]],
            global_node_ids[target_type][edge_index[1]],
        ]
    )


def _inference_device(configured: object) -> torch.device:
    value = str(configured).lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference/training was requested but is unavailable.")
    return device


def _infer_wnn_subgraph_mean(
    model: StateModel,
    build: StateGraphBuildResult,
    sampler: CellKNNSampler,
    *,
    seed_batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float | str]]:
    """Cover all cells with local WNN graphs and average repeated node states."""

    if seed_batch_size < 1:
        raise ValueError("inference_seed_cells must be positive.")
    sums = {
        node_type: torch.zeros(
            (
                int(build.data[node_type].num_nodes),
                int(model.model_config["hidden_dim"]),
            ),
            dtype=torch.float32,
        )
        for node_type in build.data.node_types
    }
    counts = {
        node_type: torch.zeros(int(build.data[node_type].num_nodes), dtype=torch.long)
        for node_type in build.data.node_types
    }
    n_cells = int(build.data["cell"].num_nodes)
    batches = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, n_cells, seed_batch_size):
            seed_ids = torch.arange(start, min(start + seed_batch_size, n_cells))
            batch = sampler.sample(seed_ids)
            local_data = batch.data if device.type == "cpu" else batch.data.clone().to(device)
            local_features = {
                name: values.to(device)
                for name, values in _subgraph_features(build, batch).items()
            }
            local = model.encode(
                local_data,
                local_features,
                global_node_ids=dict(batch.global_node_ids),
            )
            for node_type, values in local.items():
                global_ids = batch.global_node_ids[node_type]
                sums[node_type].index_add_(0, global_ids, values.detach().cpu())
                counts[node_type].index_add_(
                    0, global_ids, torch.ones_like(global_ids, dtype=torch.long)
                )
            batches += 1
        if bool((counts["cell"] == 0).any()):
            raise RuntimeError("WNN inference failed to cover every cell node.")
        initialized = {
            name: value.detach().cpu()
            for name, value in model.initializer(
                {name: value.to(device) for name, value in build.features.items()}
            ).items()
        }
    embeddings: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, float | str] = {
        "final_inference_mode": "wnn_subgraph_mean",
        "final_inference_batches": float(batches),
    }
    for node_type in build.data.node_types:
        visited = counts[node_type] > 0
        values = sums[node_type] / counts[node_type].clamp_min(1)[:, None]
        values[~visited] = initialized[node_type][~visited]
        embeddings[node_type] = values
        diagnostics[f"final_unvisited_{node_type}_nodes"] = float((~visited).sum())
        diagnostics[f"final_mean_context_count_{node_type}"] = float(
            counts[node_type].float().mean()
        )
    for relation, edge_type in STATE_RELATIONS.items():
        target_type = edge_type[2]
        observed_targets = torch.unique(build.data[edge_type].edge_index[1])
        if bool((counts[target_type][observed_targets] == 0).any()):
            raise RuntimeError(
                f"WNN inference left observed {relation} target nodes unvisited."
            )
    return embeddings, diagnostics


def _write_embeddings(embeddings: dict[str, torch.Tensor], names: dict[str, list[str]], output_dir: Path) -> None:
    target = output_dir / "embeddings"
    target.mkdir(parents=True, exist_ok=True)
    for node_type, values in embeddings.items():
        frame = pd.DataFrame(values.detach().cpu().numpy())
        frame.insert(0, "name", names[node_type])
        frame.to_parquet(target / f"{node_type}.parquet", index=False)


def train_state(
    build: StateGraphBuildResult,
    config: dict,
    output_dir: str | Path,
    model: StateModel | None = None,
) -> tuple[StateModel, pd.DataFrame, dict[str, torch.Tensor]]:
    """Train Stage 1 on full or WNN-sampled observed graphs using graph loss only.

    ``model`` may be supplied by tutorial or experiment code that needs to
    inspect the exact initialized instance before training.  Omitting it keeps
    the historical behavior and constructs a model from ``config``.
    """

    output_dir = Path(output_dir)
    set_seed(int(config.get("seed", 1)))
    model_cfg = config.get("state_model", config.get("model", {}))
    train_cfg = config.get("state_training", config.get("training", {}))
    device = _inference_device(train_cfg.get("device", "cpu"))
    loss_mapping = dict(train_cfg)
    loss_mapping.update(train_cfg.get("losses", {}))
    Stage1LossConfig.from_mapping(loss_mapping)
    if model is None:
        model = StateModel(
            build.data,
            build.features,
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_heads=int(model_cfg.get("num_heads", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_id_residual=bool(model_cfg.get("use_id_residual", False)),
            atac_feature_dim=int(build.factors.atac_cell_scores.shape[1]),
            modality_fusion=str(model_cfg.get("modality_fusion", "joint")),
            atac_gate_init=float(model_cfg.get("atac_gate_init", 0.25)),
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    base_seed = int(config.get("seed", 1))
    rng_mode = str(train_cfg.get("rng_mode", "legacy"))
    if rng_mode not in {"legacy", "ablation_fixed"}:
        raise ValueError("state_training.rng_mode must be 'legacy' or 'ablation_fixed'.")
    generator = torch.Generator().manual_seed(base_seed)
    relation_generators = {
        relation: (
            torch.Generator().manual_seed(base_seed + 1_001 + offset)
            if rng_mode == "ablation_fixed"
            else generator
        )
        for offset, relation in enumerate(STATE_RELATIONS)
    }
    sampling_strategy = str(train_cfg.get("sampling_strategy", "edge_uniform"))
    if sampling_strategy not in {"edge_uniform", "cell_balanced", "wnn_subgraph"}:
        raise ValueError(
            "state_training.sampling_strategy must be 'edge_uniform', "
            "'cell_balanced', or 'wnn_subgraph'."
        )
    checkpoint_selection = str(
        train_cfg.get(
            "checkpoint_selection",
            "last" if sampling_strategy in {"cell_balanced", "wnn_subgraph"} else "loss",
        )
    )
    if checkpoint_selection not in {"last", "loss"}:
        raise ValueError("state_training.checkpoint_selection must be 'last' or 'loss'.")
    batch_size = int(train_cfg.get("batch_size_edges_per_relation", 256))
    batch_size_cells = int(train_cfg.get("batch_size_cells", 1024))
    subgraph_sampler: CellKNNSampler | None = None
    subgraph_generator = torch.Generator().manual_seed(base_seed + 3_001)
    subgraph_seed_cells = int(train_cfg.get("subgraph_seed_cells", 128))
    if sampling_strategy == "wnn_subgraph":
        n_cells = int(build.data["cell"].num_nodes)
        subgraph_neighbors = int(train_cfg.get("subgraph_neighbors", 20))
        subgraph_candidates = int(train_cfg.get("subgraph_candidate_neighbors", 100))
        if not 1 <= subgraph_seed_cells <= n_cells:
            raise ValueError(
                "state_training.subgraph_seed_cells must be between 1 and n_cells."
            )
        if not 1 <= subgraph_neighbors < n_cells:
            raise ValueError(
                "state_training.subgraph_neighbors must be between 1 and n_cells - 1."
            )
        subgraph_sampler = CellKNNSampler(
            build.data,
            SamplerConfig(
                seed=base_seed,
                n_neighbors=subgraph_neighbors,
                candidate_neighbors=subgraph_candidates,
                neighbor_temperature=float(
                    train_cfg.get("subgraph_neighbor_temperature", 0.2)
                ),
                include_reverse_edges=True,
                negative_ratio=max(
                    int(train_cfg.get("negative_ratio_cg", 5)),
                    int(train_cfg.get("negative_ratio_cp", 5)),
                ),
            ),
            rna_pca=build.factors.rna_cell_scores,
            atac_lsi=build.factors.atac_cell_scores,
        )
    sampled_sources: dict[str, set[int]] = {relation: set() for relation in STATE_RELATIONS}
    sampled_seed_cells: set[int] = set()
    rows: list[dict[str, float | str]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(train_cfg.get("epochs", 100)) + 1):
        if rng_mode == "ablation_fixed":
            _seed_torch_dropout(base_seed + 10_000 + epoch)
        model.train()
        optimizer.zero_grad()
        row: dict[str, float | str] = {
            "epoch": float(epoch),
            "rng_mode": rng_mode,
            "sampling_strategy": sampling_strategy,
        }
        training_data = build.data
        training_features = build.features
        global_node_ids: Mapping[str, torch.Tensor] | None = None
        if subgraph_sampler is not None:
            seed_ids = torch.randperm(
                int(build.data["cell"].num_nodes), generator=subgraph_generator
            )[:subgraph_seed_cells]
            sampled_seed_cells.update(map(int, seed_ids.tolist()))
            batch = subgraph_sampler.sample(seed_ids)
            training_data = batch.data
            training_features = _subgraph_features(build, batch)
            global_node_ids = batch.global_node_ids
            row.update(
                {
                    "subgraph_seed_cells": float(seed_ids.numel()),
                    "subgraph_seed_checksum": float(
                        _edge_checksum(torch.stack([seed_ids, seed_ids]))
                    ),
                    "subgraph_seed_cell_coverage": len(sampled_seed_cells)
                    / float(build.data["cell"].num_nodes),
                    "subgraph_context_cells": float(
                        batch.diagnostics["context_cell_count"]
                    ),
                    "subgraph_cell_fraction": float(
                        batch.data["cell"].num_nodes
                    )
                    / float(build.data["cell"].num_nodes),
                }
            )
            for node_type in ("cell", "gene", "peak"):
                row[f"subgraph_{node_type}_nodes"] = float(
                    batch.data[node_type].num_nodes
                )
            for relation, edge_type in STATE_RELATIONS.items():
                reverse = (edge_type[2], f"rev_{edge_type[1]}", edge_type[0])
                row[f"subgraph_{relation}_edges"] = float(
                    batch.data[edge_type].edge_index.shape[1]
                )
                row[f"subgraph_rev_{relation}_edges"] = float(
                    batch.data[reverse].edge_index.shape[1]
                )
        encoded_data = (
            training_data
            if device.type == "cpu"
            else training_data.clone().to(device)
        )
        encoded_features = {
            name: value.to(device) for name, value in training_features.items()
        }
        embeddings = model.encode(
            encoded_data,
            encoded_features,
            global_node_ids=(
                None if global_node_ids is None else dict(global_node_ids)
            ),
        )
        relation_losses: dict[str, torch.Tensor] = {}
        for relation, edge_type in STATE_RELATIONS.items():
            src_type, dst_type, neg_key, temp_key = STATE_REL_CFG[relation]
            relation_generator = relation_generators[relation]
            if sampling_strategy in {"cell_balanced", "wnn_subgraph"}:
                edges_per_cell = int(
                    train_cfg.get(
                        f"positive_edges_per_cell_{relation}",
                        train_cfg.get("positive_edges_per_cell", 4),
                    )
                )
                positive, weights = sample_positive_edges_by_source(
                    training_data,
                    edge_type,
                    batch_size_cells,
                    edges_per_cell,
                    relation_generator,
                )
            else:
                positive, weights = sample_positive_edges(
                    training_data, edge_type, batch_size, relation_generator
                )
            global_positive = _global_edge_index(
                positive, src_type, dst_type, global_node_ids
            )
            current_sources = set(map(int, global_positive[0].unique().tolist()))
            sampled_sources[relation].update(current_sources)
            row[f"{relation}_sampled_edges"] = float(positive.shape[1])
            row[f"{relation}_sampled_cells"] = float(len(current_sources))
            row[f"{relation}_cell_coverage"] = len(sampled_sources[relation]) / float(build.data["cell"].num_nodes)
            negatives = sample_negative_targets(
                positive,
                int(training_data[dst_type].num_nodes),
                int(train_cfg.get(neg_key, 5)),
                relation_generator,
                all_positive_edge_index=training_data[edge_type].edge_index,
            )
            row[f"{relation}_positive_checksum"] = float(
                _edge_checksum(global_positive)
            )
            negative_edges = torch.stack(
                [positive[0].repeat_interleave(negatives.shape[1]), negatives.reshape(-1)]
            )
            global_negative = _global_edge_index(
                negative_edges, src_type, dst_type, global_node_ids
            )
            row[f"{relation}_negative_checksum"] = float(
                _edge_checksum(global_negative)
            )
            positive_device = positive.to(device)
            negatives_device = negatives.to(device)
            pos_logits = model.decoders.score(
                relation,
                embeddings[src_type],
                embeddings[dst_type],
                positive_device,
            )
            src_ids = positive_device[0].repeat_interleave(negatives.shape[1])
            neg_logits = model.decoders.score_pairs(
                relation,
                embeddings[src_type],
                embeddings[dst_type],
                src_ids,
                negatives_device.reshape(-1),
            ).reshape(positive.shape[1], -1)
            loss = weighted_infonce_loss(
                pos_logits,
                neg_logits,
                weights.to(device),
                float(train_cfg.get(temp_key, 0.2)),
            )
            relation_losses[relation] = loss
            row[f"{relation}_loss"] = float(loss.detach())

        graph_loss = 0.5 * (
            relation_losses["cg"] + relation_losses["cp"]
        )
        graph_loss.backward()
        optimizer.step()
        graph_value = float(graph_loss.detach())
        row.update(
            {
                "graph_loss": graph_value,
                "L_graph": graph_value,
                "graph_status": "active",
                "loss": graph_value,
                "total_loss": graph_value,
                "active_weight_lambda_graph": 1.0,
            }
        )
        if model.initializer.atac_gate is not None:
            row["atac_gate"] = float(model.initializer.atac_gate.detach())
        rows.append(row)
        if checkpoint_selection == "last" or row["loss"] < best_loss:
            best_loss = row["loss"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Stage-1 training produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    final_inference_mode = str(
        train_cfg.get(
            "final_inference_mode",
            "wnn_subgraph_mean" if subgraph_sampler is not None else "full_graph",
        )
    )
    if final_inference_mode == "wnn_subgraph_mean":
        if subgraph_sampler is None:
            raise ValueError("wnn_subgraph_mean requires sampling_strategy='wnn_subgraph'.")
        embeddings, inference_diagnostics = _infer_wnn_subgraph_mean(
            model,
            build,
            subgraph_sampler,
            seed_batch_size=int(
                train_cfg.get("inference_seed_cells", subgraph_seed_cells)
            ),
            device=device,
        )
    elif final_inference_mode == "full_graph":
        with torch.no_grad():
            inference_data = (
                build.data if device.type == "cpu" else build.data.clone().to(device)
            )
            embeddings = {
                name: value.detach().cpu()
                for name, value in model.encode(
                    inference_data,
                    {name: value.to(device) for name, value in build.features.items()},
                ).items()
            }
        inference_diagnostics = {
            "final_inference_mode": "full_graph",
            "final_inference_batches": 1.0,
        }
    else:
        raise ValueError(
            "final_inference_mode must be 'full_graph' or 'wnn_subgraph_mean'."
        )
    metrics = pd.DataFrame(rows)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "logs" / "stage1_metrics.csv", index=False)
    pd.DataFrame([inference_diagnostics]).to_csv(
        output_dir / "logs" / "stage1_inference_diagnostics.csv", index=False
    )
    _write_embeddings(embeddings, build.names, output_dir)
    save_checkpoint(
        output_dir / "checkpoints" / "stage1_best.pt",
        stage="stage1",
        model_state=best_state,
        model_config=model.model_config,
        config=config,
        names=build.names,
        features={name: value.cpu() for name, value in build.features.items()},
        embeddings={name: value.cpu() for name, value in embeddings.items()},
        cell_types=build.metadata.get("cell_types"),
    )
    return model, metrics, embeddings


def load_state_model(
    build: StateGraphBuildResult,
    checkpoint: str | Path,
) -> tuple[StateModel, dict[str, torch.Tensor]]:
    """Restore a Stage-1 model and its frozen reference embeddings."""

    payload = load_checkpoint(checkpoint)
    if payload.get("stage") != "stage1":
        raise ValueError("Expected a Stage-1 checkpoint.")
    if payload.get("names") != build.names:
        raise ValueError("Stage-1 checkpoint node identifiers do not match the current data.")
    model = StateModel(build.data, build.features, **payload["model_config"])
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    legacy_atac_decoder = {
        "atac_reconstruction.weight",
        "atac_reconstruction.bias",
    }
    if incompatible.missing_keys or not set(incompatible.unexpected_keys).issubset(
        legacy_atac_decoder
    ):
        raise RuntimeError(
            "Stage-1 checkpoint parameters are incompatible with the current model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )
    embeddings = {name: value.detach().cpu() for name, value in payload["embeddings"].items()}
    return model, embeddings
