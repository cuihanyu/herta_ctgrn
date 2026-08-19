"""Stage-1 cell-state training and output writing."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Mapping

import pandas as pd
import torch

from herta.data.negative_sampling import sample_negative_targets
from herta.data.sampler import (
    sample_cells_with_all_relations,
    sample_positive_edges_for_cells,
)
from herta.data.sampling import CellKNNSampler, SamplerConfig, SubgraphBatch
from herta.data.state_graph import STATE_RELATIONS, StateGraphBuildResult
from herta.model.losses import (
    Stage1LossConfig,
    stage1_objective,
    weighted_infonce_loss,
    wnn_neighborhood_infonce_loss,
)
from herta.model.state_model import StateModel
from herta.utils.checkpoint import save_checkpoint
from herta.utils.checkpoint import load_checkpoint
from herta.utils.seed import set_seed


STATE_REL_CFG = {
    "cg": ("cell", "gene", "negative_ratio_cg", "temperature_cg"),
    "cp": ("cell", "peak", "negative_ratio_cp", "temperature_cp"),
}

LOGGER = logging.getLogger(__name__)


def _split_fixed_validation_edges(
    data,
    *,
    validation_fraction: float,
    validation_cells: int,
    max_edges_per_cell: int,
    generator: torch.Generator,
) -> tuple[object, dict[str, dict[str, torch.Tensor]]]:
    """Hold out deterministic observed edges while retaining train degree per cell."""

    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if validation_cells < 1 or max_edges_per_cell < 1:
        raise ValueError(
            "validation_cells and validation_positive_edges_per_cell must be positive."
        )
    training_data = data.clone()
    if validation_fraction == 0:
        return training_data, {}
    n_cells = int(data["cell"].num_nodes)
    selected_sources = torch.randperm(n_cells, generator=generator)[
        : min(validation_cells, n_cells)
    ]
    validation: dict[str, dict[str, torch.Tensor]] = {}
    for relation, edge_type in STATE_RELATIONS.items():
        edge_index = data[edge_type].edge_index
        edge_weight = data[edge_type].edge_weight
        sources = edge_index[0]
        counts = torch.bincount(sources, minlength=n_cells)
        starts = torch.cumsum(counts, dim=0) - counts
        held_out: list[torch.Tensor] = []
        for source in selected_sources.tolist():
            degree = int(counts[source])
            if degree < 2:
                continue
            requested = max(1, int(round(degree * validation_fraction)))
            count = min(requested, max_edges_per_cell, degree - 1)
            local = torch.randperm(degree, generator=generator)[:count]
            held_out.append(starts[source] + local)
        if not held_out:
            continue
        validation_ids = torch.sort(torch.cat(held_out)).values
        train_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        train_mask[validation_ids] = False
        training_data[edge_type].edge_index = edge_index[:, train_mask]
        for key, value in data[edge_type].items():
            if key == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == edge_index.shape[1]:
                training_data[edge_type][key] = value[train_mask]
        reverse = (edge_type[2], f"rev_{edge_type[1]}", edge_type[0])
        if reverse in training_data.edge_types:
            training_data[reverse].edge_index = edge_index[:, train_mask].flip(0)
            for key, value in data[reverse].items():
                if key == "edge_index":
                    continue
                if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == edge_index.shape[1]:
                    training_data[reverse][key] = value[train_mask]
        validation[relation] = {
            "positive": edge_index[:, validation_ids],
            "weight": edge_weight[validation_ids],
            "full_positive": edge_index,
        }
    return training_data, validation


def _map_global_edges_to_local_if_present(
    edge_index: torch.Tensor,
    global_node_ids: Mapping[str, torch.Tensor],
    source_type: str,
    target_type: str,
) -> torch.Tensor:
    """Map global edges whose endpoints occur in a sampled batch to local IDs."""

    source_ids = global_node_ids[source_type]
    target_ids = global_node_ids[target_type]
    source_local = torch.searchsorted(source_ids, edge_index[0])
    target_local = torch.searchsorted(target_ids, edge_index[1])
    source_ok = source_local < len(source_ids)
    target_ok = target_local < len(target_ids)
    source_match = torch.zeros_like(source_ok)
    target_match = torch.zeros_like(target_ok)
    source_match[source_ok] = source_ids[source_local[source_ok]] == edge_index[0, source_ok]
    target_match[target_ok] = target_ids[target_local[target_ok]] == edge_index[1, target_ok]
    keep = source_match & target_match
    return torch.stack([source_local[keep], target_local[keep]])


def _fixed_validation_payload(
    validation: dict[str, dict[str, torch.Tensor]],
    data,
    train_cfg: Mapping[str, object],
    generator: torch.Generator,
) -> dict[str, dict[str, torch.Tensor]]:
    """Attach one deterministic negative matrix to every held-out relation."""

    payload: dict[str, dict[str, torch.Tensor]] = {}
    for relation, values in validation.items():
        _, dst_type, neg_key, _ = STATE_REL_CFG[relation]
        positive = values["positive"]
        payload[relation] = dict(values)
        payload[relation]["negative"] = sample_negative_targets(
            positive,
            int(data[dst_type].num_nodes),
            int(train_cfg.get(neg_key, 5)),
            generator,
            all_positive_edge_index=values["full_positive"],
        )
    return payload


def _evaluate_fixed_validation(
    model: StateModel,
    build: StateGraphBuildResult,
    training_data,
    validation: dict[str, dict[str, torch.Tensor]],
    loss_config: Stage1LossConfig,
    train_cfg: Mapping[str, object],
    device: torch.device,
    subgraph_sampler: CellKNNSampler | None,
) -> dict[str, float]:
    """Evaluate fixed held-out positives and negatives without resampling."""

    if set(validation) != set(STATE_RELATIONS):
        raise ValueError("Fixed validation requires held-out CG and CP edges.")
    model.eval()
    with torch.no_grad():
        if subgraph_sampler is None:
            encoded_data = (
                training_data
                if device.type == "cpu"
                else training_data.clone().to(device)
            )
            embeddings = model.encode(
                encoded_data,
                {name: value.to(device) for name, value in build.features.items()},
            )
        else:
            validation_sources = torch.unique(
                torch.cat(
                    [values["positive"][0] for values in validation.values()]
                ),
                sorted=True,
            )
            batch = subgraph_sampler.sample(validation_sources)
            local_data = (
                batch.data if device.type == "cpu" else batch.data.clone().to(device)
            )
            local_embeddings = model.encode(
                local_data,
                {
                    name: value.to(device)
                    for name, value in _subgraph_features(build, batch).items()
                },
                global_node_ids=dict(batch.global_node_ids),
            )
            embeddings = model.initializer(
                {name: value.to(device) for name, value in build.features.items()}
            )
            embeddings = {name: value.clone() for name, value in embeddings.items()}
            for node_type, values in local_embeddings.items():
                embeddings[node_type].index_copy_(
                    0,
                    batch.global_node_ids[node_type].to(device),
                    values,
                )
        losses: dict[str, torch.Tensor] = {}
        for relation, values in validation.items():
            src_type, dst_type, _, temp_key = STATE_REL_CFG[relation]
            positive = values["positive"].to(device)
            negative = values["negative"].to(device)
            pos_logits = model.decoders.score(
                relation, embeddings[src_type], embeddings[dst_type], positive
            )
            src_ids = positive[0].repeat_interleave(negative.shape[1])
            neg_logits = model.decoders.score_pairs(
                relation,
                embeddings[src_type],
                embeddings[dst_type],
                src_ids,
                negative.reshape(-1),
            ).reshape(positive.shape[1], -1)
            losses[relation] = weighted_infonce_loss(
                pos_logits,
                neg_logits,
                values["weight"].to(device),
                float(train_cfg.get(temp_key, 0.2)),
            )
        total = stage1_objective(
            losses["cg"],
            losses["cp"],
            lambda_graph=loss_config.weights.lambda_graph,
            lambda_cg=loss_config.lambda_cg,
            lambda_cp=loss_config.lambda_cp,
            normalize_relation_weights=False,
        )
    return {
        "validation_loss_cg": float(losses["cg"]),
        "validation_loss_cp": float(losses["cp"]),
        "validation_loss": float(total),
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
        observed_targets = torch.unique(sampler.data[edge_type].edge_index[1])
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
    loss_config = Stage1LossConfig.from_mapping(loss_mapping)
    if (
        loss_config.weights.lambda_graph != 1.0
        or loss_config.lambda_cg != 0.5
        or loss_config.lambda_cp != 0.5
        or loss_config.lambda_wnn != 0.0
    ):
        raise ValueError(
            "The formal Stage-1 objective is fixed to 0.5*L_CG + 0.5*L_CP."
        )
    modality_fusion = str(model_cfg.get("modality_fusion", "joint"))
    if modality_fusion not in {"joint", "concat"}:
        raise ValueError(
            "The formal Stage-1 baseline fixes modality_fusion='joint' (concat)."
        )
    if model is None:
        model = StateModel(
            build.data,
            build.features,
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_heads=int(model_cfg.get("num_heads", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_id_residual=bool(model_cfg.get("use_id_residual", False)),
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
    sampling_strategy = str(train_cfg.get("sampling_strategy", "cell_balanced"))
    if sampling_strategy != "cell_balanced":
        raise ValueError(
            "The formal Stage-1 baseline requires sampling_strategy='cell_balanced'."
        )
    checkpoint_selection = str(
        train_cfg.get(
            "checkpoint_selection",
            "loss",
        )
    )
    if checkpoint_selection != "loss":
        raise ValueError(
            "The formal Stage-1 baseline requires checkpoint_selection='loss'."
        )
    epochs = int(train_cfg.get("epochs", 100))
    # Omitted values retain the historical one-update epoch; maintained configs
    # now set this explicitly to 10 so existing callers are not silently expanded.
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 1))
    max_steps = int(train_cfg.get("max_steps", 1000))
    if epochs < 1 or steps_per_epoch < 1 or max_steps < 1:
        raise ValueError("epochs, steps_per_epoch, and max_steps must be positive.")
    total_planned_steps = min(epochs * steps_per_epoch, max_steps)
    validation_interval = int(train_cfg.get("validation_interval_steps", 50))
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 10))
    if validation_interval < 1 or early_stopping_patience < 1:
        raise ValueError(
            "validation_interval_steps and early_stopping_patience must be positive."
        )
    training_graph, validation = build.data, {}
    batch_size_cells = int(train_cfg.get("batch_size_cells", 1024))
    subgraph_sampler: CellKNNSampler | None = None
    subgraph_generator = torch.Generator().manual_seed(base_seed + 3_001)
    wnn_generator = torch.Generator().manual_seed(base_seed + 5_001)
    subgraph_seed_cells = int(train_cfg.get("subgraph_seed_cells", 128))
    if sampling_strategy == "wnn_subgraph":
        n_cells = int(training_graph["cell"].num_nodes)
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
            training_graph,
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
    best_optimizer_state: dict | None = None
    best_global_step = 0
    validation_count = 0
    validations_without_improvement = 0
    global_step = 0
    stop_training = False
    log_interval = int(train_cfg.get("log_interval_steps", 25))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    if log_interval < 1 or grad_clip_norm <= 0:
        raise ValueError("log_interval_steps and grad_clip_norm must be positive.")
    for epoch in range(1, epochs + 1):
        for step_in_epoch in range(1, steps_per_epoch + 1):
            if global_step >= max_steps:
                stop_training = True
                break
            global_step += 1
            if rng_mode == "ablation_fixed":
                _seed_torch_dropout(base_seed + 10_000 + global_step)
            model.train()
            optimizer.zero_grad()
            row: dict[str, float | str] = {
                "epoch": float(epoch),
                "step_in_epoch": float(step_in_epoch),
                "global_step": float(global_step),
                "rng_mode": rng_mode,
                "sampling_strategy": sampling_strategy,
            }
            training_data = training_graph
            training_features = build.features
            global_node_ids: Mapping[str, torch.Tensor] | None = None
            if subgraph_sampler is not None:
                seed_ids = torch.randperm(
                    int(training_graph["cell"].num_nodes),
                    generator=subgraph_generator,
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
            sampled_cells = sample_cells_with_all_relations(
                training_data,
                tuple(STATE_RELATIONS.values()),
                batch_size_cells,
                generator,
            )
            row["sampled_cell_count"] = float(sampled_cells.numel())
            for relation, edge_type in STATE_RELATIONS.items():
                src_type, dst_type, neg_key, temp_key = STATE_REL_CFG[relation]
                relation_generator = relation_generators[relation]
                edges_per_cell = int(
                    train_cfg.get(
                        f"positive_edges_per_cell_{relation}",
                        train_cfg.get("positive_edges_per_cell", 4),
                    )
                )
                positive, weights = sample_positive_edges_for_cells(
                    training_data,
                    edge_type,
                    sampled_cells,
                    edges_per_cell,
                    relation_generator,
                )
                global_positive = _global_edge_index(
                    positive, src_type, dst_type, global_node_ids
                )
                current_sources = set(map(int, global_positive[0].unique().tolist()))
                sampled_sources[relation].update(current_sources)
                row[f"{relation}_sampled_edges"] = float(positive.shape[1])
                row[f"{relation}_positive_count"] = float(positive.shape[1])
                row[f"{relation}_sampled_cells"] = float(len(current_sources))
                row[f"{relation}_cell_coverage"] = len(sampled_sources[relation]) / float(
                    build.data["cell"].num_nodes
                )
                all_local_positive = training_data[edge_type].edge_index
                if relation in validation:
                    if global_node_ids is None:
                        all_local_positive = validation[relation]["full_positive"]
                    else:
                        mapped_validation = _map_global_edges_to_local_if_present(
                            validation[relation]["positive"],
                            global_node_ids,
                            src_type,
                            dst_type,
                        )
                        all_local_positive = torch.cat(
                            [all_local_positive, mapped_validation], dim=1
                        )
                negatives = sample_negative_targets(
                    positive,
                    int(training_data[dst_type].num_nodes),
                    int(train_cfg.get(neg_key, 5)),
                    relation_generator,
                    all_positive_edge_index=all_local_positive,
                )
                row[f"{relation}_negative_count"] = float(negatives.numel())
                row[f"{relation}_positive_checksum"] = float(
                    _edge_checksum(global_positive)
                )
                negative_edges = torch.stack(
                    [
                        positive[0].repeat_interleave(negatives.shape[1]),
                        negatives.reshape(-1),
                    ]
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
                relation_loss = weighted_infonce_loss(
                    pos_logits,
                    neg_logits,
                    weights.to(device),
                    float(train_cfg.get(temp_key, 0.2)),
                )
                relation_losses[relation] = relation_loss
                row[f"{relation}_loss"] = float(relation_loss.detach())
                row[f"loss_{relation}"] = float(relation_loss.detach())

            graph_loss = (
                loss_config.lambda_cg * relation_losses["cg"]
                + loss_config.lambda_cp * relation_losses["cp"]
            )
            wnn_loss: torch.Tensor | None = None
            if loss_config.lambda_wnn > 0:
                if subgraph_sampler is None or global_node_ids is None:
                    raise RuntimeError("WNN loss requires a sampled WNN batch.")
                members = batch.diagnostics["context_member_ids"]
                local_members = torch.searchsorted(
                    global_node_ids["cell"], members
                )
                if not torch.equal(
                    global_node_ids["cell"][local_members], members
                ):
                    raise RuntimeError("WNN members are absent from the sampled cell graph.")
                wnn_loss = wnn_neighborhood_infonce_loss(
                    embeddings["cell"],
                    local_members[:, 0],
                    local_members[:, 1:],
                    batch.diagnostics["neighbor_weights"],
                    num_negatives=int(train_cfg.get("negative_ratio_wnn", 5)),
                    temperature=float(train_cfg.get("temperature_wnn", 0.2)),
                    generator=wnn_generator,
                )
                row["wnn_loss"] = float(wnn_loss.detach())
                row["loss_wnn"] = float(wnn_loss.detach())
                row["wnn_negative_count"] = float(
                    int(train_cfg.get("negative_ratio_wnn", 5))
                )
            total_loss = stage1_objective(
                relation_losses["cg"],
                relation_losses["cp"],
                lambda_graph=loss_config.weights.lambda_graph,
                lambda_cg=loss_config.lambda_cg,
                lambda_cp=loss_config.lambda_cp,
                wnn_loss=wnn_loss,
                lambda_wnn=loss_config.lambda_wnn,
                normalize_relation_weights=False,
            )
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_norm
            )
            optimizer.step()
            graph_value = float(graph_loss.detach())
            total_value = float(total_loss.detach())
            row.update(
                {
                    "graph_loss": graph_value,
                    "L_graph": graph_value,
                    "graph_status": "active",
                    "loss": total_value,
                    "loss_total": total_value,
                    "total_loss": total_value,
                    "gradient_norm": float(gradient_norm),
                    "active_weight_lambda_graph": loss_config.weights.lambda_graph,
                    "active_weight_lambda_cg": loss_config.lambda_cg,
                    "active_weight_lambda_cp": loss_config.lambda_cp,
                    "active_weight_lambda_wnn": loss_config.lambda_wnn,
                }
            )

            should_validate = bool(validation) and (
                global_step % validation_interval == 0
                or global_step == total_planned_steps
            )
            if should_validate:
                validation_count += 1
                validation_metrics = _evaluate_fixed_validation(
                    model,
                    build,
                    training_graph,
                    validation,
                    loss_config,
                    train_cfg,
                    device,
                    subgraph_sampler,
                )
                row.update(validation_metrics)
            selection_value: float | None = None
            if checkpoint_selection == "last":
                selection_value = total_value
            elif checkpoint_selection == "loss":
                if total_value < best_loss:
                    selection_value = total_value
            elif should_validate:
                current_validation = float(row["validation_loss"])
                if current_validation < best_loss:
                    selection_value = current_validation
                    validations_without_improvement = 0
                else:
                    validations_without_improvement += 1
            if selection_value is not None:
                best_loss = selection_value
                best_global_step = global_step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            row["best_global_step"] = float(best_global_step)
            row["validation_count"] = float(validation_count)
            rows.append(row)
            if global_step == 1 or global_step % log_interval == 0:
                LOGGER.info(
                    "Stage1 epoch=%d step=%d global_step=%d loss_total=%.6f "
                    "loss_cg=%.6f loss_cp=%.6f negatives_cg=%d negatives_cp=%d",
                    epoch,
                    step_in_epoch,
                    global_step,
                    total_value,
                    row["loss_cg"],
                    row["loss_cp"],
                    int(row["cg_negative_count"]),
                    int(row["cp_negative_count"]),
                )
            if (
                checkpoint_selection == "validation"
                and should_validate
                and validations_without_improvement >= early_stopping_patience
            ):
                row["early_stopping_triggered"] = 1.0
                stop_training = True
                break
        if stop_training:
            break

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
        optimizer_state=best_optimizer_state,
        model_config=model.model_config,
        config=config,
        global_step=best_global_step,
        random_seed=base_seed,
        checkpoint_selection=checkpoint_selection,
        best_selection_loss=best_loss,
        validation_positive_edges={
            relation: values["positive"].cpu()
            for relation, values in validation.items()
        },
        validation_negative_targets={
            relation: values["negative"].cpu()
            for relation, values in validation.items()
        },
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
