"""Shared Stage-2 regulatory training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from herta.data.negative_sampling import sample_negative_targets
from herta.data.regulatory_graph import REGULATORY_RELATIONS, EdgeSplit, RegulatoryGraphBuildResult
from herta.model.grn_model import GRNModel
from herta.model.losses import (
    MultiTaskLossManager,
    Stage2LossConfig,
    embedding_anchor_loss,
    regulatory_path_ranking_loss,
    tf_gene_bce_loss,
    tf_peak_gene_consistency_loss,
    weighted_infonce_loss,
)
from herta.model.state_model import StateModel
from herta.utils.checkpoint import save_checkpoint
from herta.utils.seed import set_seed


@dataclass(frozen=True)
class _FixedPathQueries:
    paths: torch.Tensor
    tp_ids: torch.Tensor
    pg_ids: torch.Tensor
    negatives: torch.Tensor


@dataclass(frozen=True)
class _FixedValidationProtocol:
    data: object
    features: dict[str, torch.Tensor]
    global_node_ids: object
    is_tf: torch.Tensor
    batch: SubgraphBatch | None
    splits: dict[str, EdgeSplit]
    all_positive: dict[str, torch.Tensor]
    negative_edges: dict[str, torch.Tensor]
    paths: _FixedPathQueries | None
    diagnostics: dict[str, float | str]


def _all_positive_edges(build: RegulatoryGraphBuildResult, relation: str) -> torch.Tensor:
    return torch.cat([split.edge_index for split in build.splits[relation].values()], dim=1)


def _build_context_path_sampler(
    build: RegulatoryGraphBuildResult,
    train_cfg: dict,
    seed: int,
) -> None:
    """Reject regulatory path/message sampling in the minimal workflow."""

    raw = train_cfg.get("sampler")
    if raw is None:
        return None
    sampler_cfg = dict(raw)
    name = str(sampler_cfg.pop("name", "full_graph")).lower()
    if name in {"none", "full", "fullgraphsampler", "full_graph"}:
        return None
    raise ValueError(
        "Regulatory path/message sampling is deferred; Stage 2 requires full_graph "
        "observed CG/CP messages and independently batched TP/PG queries."
    )


def _sample_seed_cells(
    sampler: HERTAContextPathSampler,
    generator: torch.Generator,
) -> torch.Tensor:
    n_cells = int(sampler.data["cell"].num_nodes)
    return torch.randperm(n_cells, generator=generator)[
        : min(sampler.config.seed_cell_count, n_cells)
    ]


def _batch_features(
    build: RegulatoryGraphBuildResult,
    batch: SubgraphBatch,
) -> dict[str, torch.Tensor]:
    return {
        node_type: build.state.features[node_type][batch.global_node_ids[node_type]]
        for node_type in batch.data.node_types
    }


def _localize_split(
    split: EdgeSplit,
    edge_type: tuple[str, str, str],
    batch: SubgraphBatch,
) -> EdgeSplit:
    source = batch.global_to_local[edge_type[0]][split.edge_index[0]]
    target = batch.global_to_local[edge_type[2]][split.edge_index[1]]
    keep = (source >= 0) & (target >= 0)
    features = split.edge_features[keep] if split.edge_features is not None else None
    return EdgeSplit(
        torch.stack([source[keep], target[keep]]),
        split.edge_weight[keep],
        features,
    )


def _local_all_positive(
    build: RegulatoryGraphBuildResult,
    relation: str,
    batch: SubgraphBatch,
) -> torch.Tensor:
    edge_type = (
        REGULATORY_RELATIONS[relation]
        if relation in REGULATORY_RELATIONS
        else ("gene", "targets", "gene")
    )
    full = EdgeSplit(
        _all_positive_edges(build, relation),
        torch.ones(_all_positive_edges(build, relation).shape[1]),
    )
    return _localize_split(full, edge_type, batch).edge_index


def _tensor_checksum(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _fixed_negative_edges(
    positive: torch.Tensor,
    all_positive: torch.Tensor,
    num_targets: int,
    negative_ratio: int,
    generator: torch.Generator,
) -> torch.Tensor:
    targets = sample_negative_targets(
        positive,
        num_targets,
        negative_ratio,
        generator,
        all_positive_edge_index=all_positive,
    )
    return torch.stack(
        [positive[0].repeat_interleave(negative_ratio), targets.reshape(-1)]
    )


def _fixed_relation_negative_edges(
    build: RegulatoryGraphBuildResult,
    relation: str,
    positive: torch.Tensor,
    all_positive: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create one fixed typed negative for every TP or PG positive query."""

    if relation == "tp":
        return _fixed_negative_edges(
            positive,
            all_positive,
            int(build.data["peak"].num_nodes),
            1,
            generator,
        )
    if relation != "pg":
        raise ValueError(f"Unsupported minimal negative relation: {relation}")
    genes = build.state.metadata["genes"].reset_index(drop=True)
    peaks = build.state.metadata["peaks"].reset_index(drop=True)
    required_gene = {"chrom", "chromStart", "chromEnd", "strand"}
    required_peak = {"chrom", "chromStart", "chromEnd"}
    if required_gene.difference(genes.columns) or required_peak.difference(peaks.columns):
        return _fixed_negative_edges(
            positive,
            all_positive,
            int(build.data["gene"].num_nodes),
            1,
            generator,
        )
    gene_start = pd.to_numeric(genes["chromStart"], errors="coerce")
    gene_end = pd.to_numeric(genes["chromEnd"], errors="coerce")
    resolved_tss = gene_start.where(
        genes["strand"].astype(str).eq("+"), gene_end - 1
    )
    valid_gene_coordinates = torch.as_tensor(
        resolved_tss.notna().to_numpy(), dtype=torch.bool
    )
    gene_tss = torch.as_tensor(
        resolved_tss.fillna(0).to_numpy(dtype="int64"), dtype=torch.long
    )
    gene_chrom = genes["chrom"].astype(str).to_numpy()
    positive_by_peak: dict[int, set[int]] = {}
    for peak_id, gene_id in all_positive.t().tolist():
        positive_by_peak.setdefault(int(peak_id), set()).add(int(gene_id))
    targets: list[int] = []
    all_gene_ids = torch.arange(len(genes), dtype=torch.long)
    pools_by_peak: dict[int, torch.Tensor] = {}
    for peak_id in positive[0].tolist():
        resolved_peak_id = int(peak_id)
        pool = pools_by_peak.get(resolved_peak_id)
        if pool is None:
            peak = peaks.iloc[resolved_peak_id]
            start, end = int(peak["chromStart"]), int(peak["chromEnd"])
            distance = torch.where(
                gene_tss < start,
                start - gene_tss,
                torch.where(gene_tss >= end, gene_tss - (end - 1), 0),
            )
            same_chrom = torch.as_tensor(
                gene_chrom == str(peak["chrom"]), dtype=torch.bool
            )
            excluded = positive_by_peak.get(resolved_peak_id, set())
            allowed = torch.ones(len(genes), dtype=torch.bool)
            if excluded:
                allowed[torch.as_tensor(sorted(excluded), dtype=torch.long)] = False
            preferred = all_gene_ids[
                allowed & valid_gene_coordinates & same_chrom & (distance > 250_000)
            ]
            pool = preferred if preferred.numel() else all_gene_ids[allowed]
            pools_by_peak[resolved_peak_id] = pool
        if not pool.numel():
            raise ValueError(f"Peak node {peak_id} has no eligible PG negative target.")
        selected = torch.randint(pool.numel(), (1,), generator=generator)
        targets.append(int(pool[selected]))
    return torch.stack([positive[0], torch.as_tensor(targets, dtype=torch.long)])


def _sampler_metrics(diagnostics: dict[str, object]) -> dict[str, float | str]:
    result: dict[str, float | str] = {
        "sampler_name": str(diagnostics.get("sampler", "HERTAContextPathSampler"))
    }
    scalar_keys = (
        "seed_cell_count",
        "context_cell_count",
        "active_gene_count",
        "active_peak_count",
        "state_gene_candidate_count",
        "state_peak_candidate_count",
        "selected_complete_path_count",
        "path_retention_ratio",
        "cell_type_coverage",
        "edge_type_balance",
        "estimated_memory_mib",
        "sampling_time_ms",
        "negative_positive_collisions",
        "held_out_message_overlap",
    )
    for key in scalar_keys:
        value = diagnostics.get(key)
        if value is not None:
            result[f"sampler_{key}"] = float(value)
    edge_counts = diagnostics.get("edge_counts", {})
    for relation, edge_type in REGULATORY_RELATIONS.items():
        if relation in {"cg", "cp", "tp", "pg"} and edge_type in edge_counts:
            result[f"sampler_{relation}_edges"] = float(edge_counts[edge_type])
    return result


def _relation_loss(
    model: GRNModel,
    embeddings: dict[str, torch.Tensor],
    relation: str,
    positive: torch.Tensor,
    weights: torch.Tensor,
    positive_edge_features: torch.Tensor | None,
    all_positive: torch.Tensor,
    num_targets: int,
    negative_ratio: int,
    temperature: float,
    generator: torch.Generator,
    *,
    is_tf: torch.Tensor | None = None,
    negative_edge_index: torch.Tensor | None = None,
    confidence_power: float = 1.0,
    confidence_floor: float = 0.0,
    hard_negative_pool_size: int = 0,
    hard_negative_fraction: float = 0.0,
) -> torch.Tensor:
    if confidence_power <= 0:
        raise ValueError("confidence_power must be positive.")
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be in [0, 1].")
    if hard_negative_pool_size < 0:
        raise ValueError("hard_negative_pool_size must be non-negative.")
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be in [0, 1].")
    src_type, _, dst_type = REGULATORY_RELATIONS[relation]
    if negative_edge_index is None:
        negatives = sample_negative_targets(
            positive,
            num_targets,
            negative_ratio,
            generator,
            all_positive_edge_index=all_positive,
        )
    else:
        if negative_edge_index.shape[1] % positive.shape[1] != 0:
            raise ValueError("Sampler negatives must be grouped evenly by positive edge.")
        resolved_ratio = negative_edge_index.shape[1] // positive.shape[1]
        expected_sources = positive[0].repeat_interleave(resolved_ratio)
        if not torch.equal(negative_edge_index[0], expected_sources):
            raise ValueError("Sampler negative sources do not align with positive edges.")
        negatives = negative_edge_index[1].reshape(positive.shape[1], resolved_ratio)
    model_is_tf = (model.is_tf if is_tf is None else is_tf) if relation == "tp" else None
    pos_logits = model.decoders.score(
        relation,
        embeddings[src_type],
        embeddings[dst_type],
        positive,
        positive_edge_features,
        is_tf=model_is_tf,
    )
    hard_count = int(round(negatives.shape[1] * hard_negative_fraction))
    if hard_count and hard_negative_pool_size:
        pool_size = max(hard_negative_pool_size, hard_count)
        candidates = sample_negative_targets(
            positive,
            num_targets,
            pool_size,
            generator,
            all_positive_edge_index=all_positive,
        )
        candidate_sources = positive[0].repeat_interleave(pool_size)
        with torch.no_grad():
            candidate_logits = model.decoders.score_pairs(
                relation,
                embeddings[src_type],
                embeddings[dst_type],
                candidate_sources,
                candidates.reshape(-1),
                is_tf=model_is_tf,
            ).reshape(positive.shape[1], pool_size)
        hard_ids = torch.topk(candidate_logits, hard_count, dim=1).indices
        hard_targets = torch.gather(candidates, 1, hard_ids)
        negatives = torch.cat(
            [hard_targets, negatives[:, : negatives.shape[1] - hard_count]], dim=1
        )
    src_ids = positive[0].repeat_interleave(negatives.shape[1])
    neg_logits = model.decoders.score_pairs(
        relation,
        embeddings[src_type],
        embeddings[dst_type],
        src_ids,
        negatives.reshape(-1),
        is_tf=model_is_tf,
    ).reshape(positive.shape[1], -1)
    if confidence_power != 1.0 or confidence_floor != 0.0:
        max_weight = weights.max().clamp_min(1e-8)
        weights = (weights / max_weight).clamp(0.0, 1.0).pow(confidence_power)
        weights = confidence_floor + (1.0 - confidence_floor) * weights
        weights = weights / weights.mean().clamp_min(1e-8)
    return weighted_infonce_loss(pos_logits, neg_logits, weights, temperature)


def _tf_gene_loss(
    model: GRNModel,
    embeddings: dict[str, torch.Tensor],
    positive: torch.Tensor,
    positive_edge_features: torch.Tensor | None,
    all_positive: torch.Tensor,
    num_genes: int,
    negative_ratio: int,
    generator: torch.Generator,
    *,
    is_tf: torch.Tensor | None = None,
    negative_edge_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score known TF-target pairs against type-constrained sampled negatives."""

    if negative_edge_index is None:
        negatives = sample_negative_targets(
            positive,
            num_genes,
            negative_ratio,
            generator,
            all_positive_edge_index=all_positive,
        )
    else:
        if negative_edge_index.shape[1] % positive.shape[1] != 0:
            raise ValueError("Fixed TF-gene negatives must align with positive edges.")
        resolved_ratio = negative_edge_index.shape[1] // positive.shape[1]
        expected_sources = positive[0].repeat_interleave(resolved_ratio)
        if not torch.equal(negative_edge_index[0], expected_sources):
            raise ValueError("Fixed TF-gene negative sources do not align with positives.")
        negatives = negative_edge_index[1].reshape(positive.shape[1], resolved_ratio)
    gene_embeddings = embeddings["gene"]
    model_is_tf = model.is_tf if is_tf is None else is_tf
    pos_logits = model.decoders.score(
        "tg",
        gene_embeddings,
        gene_embeddings,
        positive,
        positive_edge_features,
        is_tf=model_is_tf,
    )
    src_ids = positive[0].repeat_interleave(negatives.shape[1])
    neg_logits = model.decoders.score_pairs(
        "tg",
        gene_embeddings,
        gene_embeddings,
        src_ids,
        negatives.reshape(-1),
        is_tf=model_is_tf,
    )
    return tf_gene_bce_loss(pos_logits, neg_logits)


def _sampled_consistency_loss(
    model: GRNModel,
    embeddings: dict[str, torch.Tensor],
    sampled_edges: dict[str, torch.Tensor],
    sampled_features: dict[str, torch.Tensor | None],
    mode: str,
    is_tf: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compare direct TF-gene scores with sampled two-hop path aggregation."""

    if set(sampled_edges) != {"tp", "pg"}:
        return None
    tp_score = model.decoders.probabilities(
        "tp",
        embeddings["gene"],
        embeddings["peak"],
        sampled_edges["tp"],
        sampled_features["tp"],
        is_tf=model.is_tf if is_tf is None else is_tf,
    )
    pg_score = model.decoders.probabilities(
        "pg",
        embeddings["peak"],
        embeddings["gene"],
        sampled_edges["pg"],
        sampled_features["pg"],
    )
    aggregation = model.tf_peak_gene_aggregator(
        sampled_edges["tp"],
        tp_score,
        sampled_edges["pg"],
        pg_score,
    )
    if aggregation.edge_index.shape[1] == 0:
        return None
    direct = model.decoders.probabilities(
        "tg",
        embeddings["gene"],
        embeddings["gene"],
        aggregation.edge_index,
        is_tf=model.is_tf if is_tf is None else is_tf,
    )
    return tf_peak_gene_consistency_loss(
        direct,
        aggregation.score,
        mode=mode,
    )


def _join_complete_paths(
    tp_edges: torch.Tensor,
    pg_edges: torch.Tensor,
    max_paths: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Join split-matched TP and PG queries without consulting other splits."""

    pg_by_peak: dict[int, list[int]] = {}
    for pg_id, peak in enumerate(pg_edges[0].tolist()):
        pg_by_peak.setdefault(int(peak), []).append(pg_id)
    paths: list[tuple[int, int, int]] = []
    tp_ids: list[int] = []
    pg_ids: list[int] = []
    for tp_id, (tf, peak) in enumerate(tp_edges.t().tolist()):
        for pg_id in pg_by_peak.get(int(peak), []):
            paths.append((int(tf), int(peak), int(pg_edges[1, pg_id])))
            tp_ids.append(tp_id)
            pg_ids.append(pg_id)
    if not paths:
        empty = torch.empty(0, dtype=torch.long)
        return torch.empty((3, 0), dtype=torch.long), empty, empty
    order = torch.randperm(len(paths), generator=generator)[: min(max_paths, len(paths))]
    path_tensor = torch.as_tensor(paths, dtype=torch.long).t()[:, order]
    return (
        path_tensor,
        torch.as_tensor(tp_ids, dtype=torch.long)[order],
        torch.as_tensor(pg_ids, dtype=torch.long)[order],
    )


def _sample_corrupted_paths(
    paths: torch.Tensor,
    all_tp: torch.Tensor,
    all_pg: torch.Tensor,
    is_tf: torch.Tensor,
    num_peaks: int,
    num_genes: int,
    negative_ratio: int,
    generator: torch.Generator,
) -> torch.Tensor | None:
    """Create type-correct TF/peak/gene corruptions excluding all known edges."""

    known_tp = set(map(tuple, all_tp.t().tolist()))
    known_pg = set(map(tuple, all_pg.t().tolist()))
    tf_ids = torch.nonzero(is_tf, as_tuple=False).flatten().tolist()
    candidates = {"tf": tf_ids, "peak": list(range(num_peaks)), "gene": list(range(num_genes))}
    modes = ("gene", "tf", "peak")
    rows: list[list[tuple[int, int, int]]] = []
    for tf, peak, gene in paths.t().tolist():
        negatives: list[tuple[int, int, int]] = []
        for negative_id in range(negative_ratio):
            chosen = None
            for mode_offset in range(3):
                mode = modes[(negative_id + mode_offset) % 3]
                values = candidates[mode]
                order = torch.randperm(len(values), generator=generator).tolist()
                for index in order:
                    value = int(values[index])
                    if mode == "gene" and value != gene and (peak, value) not in known_pg:
                        chosen = (tf, peak, value)
                    elif mode == "tf" and value != tf and (value, peak) not in known_tp:
                        chosen = (value, peak, gene)
                    elif (
                        mode == "peak"
                        and value != peak
                        and (tf, value) not in known_tp
                        and (value, gene) not in known_pg
                    ):
                        chosen = (tf, value, gene)
                    if chosen is not None:
                        break
                if chosen is not None:
                    break
            if chosen is None:
                return None
            negatives.append(chosen)
        rows.append(negatives)
    return torch.as_tensor(rows, dtype=torch.long)


def _split_path_ranking_loss(
    model: GRNModel,
    embeddings: dict[str, torch.Tensor],
    tp_split: EdgeSplit,
    pg_split: EdgeSplit,
    all_tp: torch.Tensor,
    all_pg: torch.Tensor,
    is_tf: torch.Tensor,
    generator: torch.Generator,
    *,
    max_paths: int,
    negative_ratio: int,
    temperature: float,
) -> tuple[torch.Tensor | None, int]:
    paths, tp_ids, pg_ids = _join_complete_paths(
        tp_split.edge_index, pg_split.edge_index, max_paths, generator
    )
    if paths.shape[1] == 0:
        return None, 0
    negatives = _sample_corrupted_paths(
        paths,
        all_tp,
        all_pg,
        is_tf,
        embeddings["peak"].shape[0],
        embeddings["gene"].shape[0],
        negative_ratio,
        generator,
    )
    if negatives is None:
        return None, int(paths.shape[1])
    return (
        _fixed_path_ranking_loss(
            model,
            embeddings,
            tp_split,
            pg_split,
            is_tf,
            _FixedPathQueries(paths, tp_ids, pg_ids, negatives),
            temperature=temperature,
        ),
        int(paths.shape[1]),
    )


def _fixed_path_ranking_loss(
    model: GRNModel,
    embeddings: dict[str, torch.Tensor],
    tp_split: EdgeSplit,
    pg_split: EdgeSplit,
    is_tf: torch.Tensor,
    queries: _FixedPathQueries,
    *,
    temperature: float,
) -> torch.Tensor:
    paths = queries.paths
    tp_ids = queries.tp_ids
    pg_ids = queries.pg_ids
    negatives = queries.negatives
    tp_features = tp_split.edge_features[tp_ids] if tp_split.edge_features is not None else None
    pg_features = pg_split.edge_features[pg_ids] if pg_split.edge_features is not None else None
    tp_positive = model.decoders.score(
        "tp", embeddings["gene"], embeddings["peak"], paths[:2], tp_features, is_tf=is_tf
    )
    pg_positive = model.decoders.score(
        "pg", embeddings["peak"], embeddings["gene"], paths[1:], pg_features
    )
    flat = negatives.reshape(-1, 3)
    tp_negative = model.decoders.score_pairs(
        "tp", embeddings["gene"], embeddings["peak"], flat[:, 0], flat[:, 1], is_tf=is_tf
    )
    pg_negative = model.decoders.score_pairs(
        "pg", embeddings["peak"], embeddings["gene"], flat[:, 1], flat[:, 2]
    )
    positive_path_log_probability = F.logsigmoid(tp_positive) + F.logsigmoid(pg_positive)
    negative_path_log_probability = (
        F.logsigmoid(tp_negative) + F.logsigmoid(pg_negative)
    ).reshape(paths.shape[1], negatives.shape[1])
    return regulatory_path_ranking_loss(
        positive_path_log_probability,
        negative_path_log_probability,
        temperature=temperature,
    )


def _build_fixed_validation_protocol(
    build: RegulatoryGraphBuildResult,
    train_cfg: dict,
    all_positive: dict[str, torch.Tensor],
    sampler: HERTAContextPathSampler | None,
    *,
    seed: int,
) -> _FixedValidationProtocol:
    """Materialize one validation context and all stochastic queries once."""

    generator = torch.Generator().manual_seed(seed)
    batch = None
    if sampler is None:
        data = build.data
        features = build.state.features
        global_node_ids = None
        is_tf = build.is_tf
        splits = {
            relation: build.splits[relation]["validation"]
            for relation in ("tp", "pg")
        }
        localized_positive = {
            relation: all_positive[relation] for relation in ("tp", "pg")
        }
    else:
        attempts = max(int(train_cfg.get("validation_context_attempts", 8)), 1)
        selected = None
        selected_score = (-1, -1, -1)
        for _ in range(attempts):
            candidate = sampler.sample(_sample_seed_cells(sampler, generator))
            candidate_splits = {
                relation: _localize_split(
                    build.splits[relation]["validation"],
                    REGULATORY_RELATIONS[relation],
                    candidate,
                )
                for relation in ("tp", "pg")
            }
            counts = tuple(
                int(candidate_splits[relation].edge_index.shape[1])
                for relation in ("tp", "pg")
            )
            tp_peaks = candidate_splits["tp"].edge_index[1]
            pg_peaks = candidate_splits["pg"].edge_index[0]
            shared_peak_count = int(torch.isin(tp_peaks, pg_peaks).sum())
            score = (
                int(shared_peak_count > 0),
                int(min(counts) > 0),
                min(counts),
            )
            if score > selected_score:
                selected = (candidate, candidate_splits)
                selected_score = score
            if score[0] == 1:
                break
        if selected is None or selected_score[1] == 0:
            raise RuntimeError(
                "Could not construct a fixed validation context containing both TP and PG queries."
            )
        batch, splits = selected
        data = batch.data
        features = _batch_features(build, batch)
        global_node_ids = batch.global_node_ids
        is_tf = batch.data["gene"].is_tf.bool()
        localized_positive = {
            relation: _local_all_positive(build, relation, batch)
            for relation in ("tp", "pg")
        }

    if "tg" in build.splits and build.splits["tg"]["validation"].edge_index.shape[1] > 0:
        splits["tg"] = (
            build.splits["tg"]["validation"]
            if batch is None
            else _localize_split(
                build.splits["tg"]["validation"],
                ("gene", "targets", "gene"),
                batch,
            )
        )
        localized_positive["tg"] = (
            all_positive["tg"]
            if batch is None
            else _local_all_positive(build, "tg", batch)
        )

    negative_edges = {}
    diagnostics: dict[str, float | str] = {
        "validation_protocol_fixed": 1.0,
        "validation_context_checksum": _tensor_checksum(
            torch.cat(
                [
                    (
                        torch.arange(int(data[node_type].num_nodes))
                        if global_node_ids is None
                        else global_node_ids[node_type]
                    ).long()
                    for node_type in data.node_types
                ]
            )
        ),
        "validation_held_out_message_overlap": float(
            0.0 if batch is None else batch.diagnostics.get("held_out_message_overlap", 0.0)
        ),
    }
    for relation in ("tp", "pg"):
        split = splits[relation]
        ratio = int(train_cfg.get(f"negative_ratio_{relation}", 1))
        if ratio != 1:
            raise ValueError("The minimal Stage-2 negative ratio is fixed at 1.")
        negative_edges[relation] = _fixed_relation_negative_edges(
            build,
            relation,
            split.edge_index,
            localized_positive[relation],
            generator,
        )
        diagnostics[f"validation_{relation}_positive_count"] = float(
            split.edge_index.shape[1]
        )
        diagnostics[f"validation_{relation}_negative_count"] = float(
            negative_edges[relation].shape[1]
        )
        diagnostics[f"validation_{relation}_positive_checksum"] = _tensor_checksum(
            split.edge_index
        )
        diagnostics[f"validation_{relation}_negative_checksum"] = _tensor_checksum(
            negative_edges[relation]
        )

    if "tg" in splits and splits["tg"].edge_index.shape[1] > 0:
        ratio = int(train_cfg.get("negative_ratio_tg", 5))
        negative_edges["tg"] = _fixed_negative_edges(
            splits["tg"].edge_index,
            localized_positive["tg"],
            int(data["gene"].num_nodes),
            ratio,
            generator,
        )
        diagnostics["validation_tg_positive_checksum"] = _tensor_checksum(
            splits["tg"].edge_index
        )
        diagnostics["validation_tg_negative_checksum"] = _tensor_checksum(
            negative_edges["tg"]
        )

    fixed_paths = None
    paths, tp_ids, pg_ids = _join_complete_paths(
        splits["tp"].edge_index,
        splits["pg"].edge_index,
        int(train_cfg.get("path_validation_size", 256)),
        generator,
    )
    if paths.shape[1] > 0:
        corruptions = _sample_corrupted_paths(
            paths,
            localized_positive["tp"],
            localized_positive["pg"],
            is_tf,
            int(data["peak"].num_nodes),
            int(data["gene"].num_nodes),
            int(train_cfg.get("negative_ratio_path", 3)),
            generator,
        )
        if corruptions is not None:
            fixed_paths = _FixedPathQueries(paths, tp_ids, pg_ids, corruptions)
            diagnostics["validation_path_positive_count"] = float(paths.shape[1])
            diagnostics["validation_path_positive_checksum"] = _tensor_checksum(paths)
            diagnostics["validation_path_negative_checksum"] = _tensor_checksum(corruptions)
    diagnostics.setdefault("validation_path_positive_count", 0.0)
    return _FixedValidationProtocol(
        data=data,
        features=features,
        global_node_ids=global_node_ids,
        is_tf=is_tf,
        batch=batch,
        splits=splits,
        all_positive=localized_positive,
        negative_edges=negative_edges,
        paths=fixed_paths,
        diagnostics=diagnostics,
    )


def train_grn(
    build: RegulatoryGraphBuildResult,
    state_model: StateModel,
    state_embeddings: dict[str, torch.Tensor],
    config: dict,
    output_dir: str | Path,
    model: GRNModel | None = None,
) -> tuple[GRNModel, pd.DataFrame]:
    """Fine-tune one shared regulatory model from Stage-1 representations.

    ``model`` is optional so tutorials can initialize and inspect the exact
    Stage-2 instance that is subsequently optimized.
    """

    output_dir = Path(output_dir)
    set_seed(int(config.get("seed", 1)))
    train_cfg = config.get("regulatory_training", config.get("training", {}))
    model_cfg = config.get("grn_model", {})
    loss_mapping = dict(train_cfg)
    loss_mapping.update(train_cfg.get("losses", {}))
    loss_config = Stage2LossConfig.from_mapping(loss_mapping)
    expected_weights = {
        "lambda_graph": 0.0,
        "lambda_contrast": 0.0,
        "lambda_tf_peak": 1.0,
        "lambda_peak_gene": 1.0,
        "lambda_tf_gene": 0.0,
        "lambda_consistency": 0.0,
        "lambda_path": 0.0,
        "lambda_cluster": 0.0,
        "lambda_anchor": 0.0,
    }
    if loss_config.weights.as_dict() != expected_weights:
        raise ValueError(
            "shared_backbone_minimal_v1 uses only equal-weight TP and PG InfoNCE."
        )
    loss_manager = MultiTaskLossManager(loss_config.weights)
    decoder_mode = str(model_cfg.get("decoder_mode", "projected_dot"))
    use_edge_features = bool(model_cfg.get("use_edge_features", False))
    if decoder_mode != "projected_dot" or use_edge_features:
        raise ValueError(
            "The minimal Stage-2 model requires projected_dot decoders without edge features."
        )
    if model is None:
        model = GRNModel(
            build.data,
            state_model,
            num_layers=model_cfg.get("num_layers"),
            num_heads=model_cfg.get("num_heads"),
            dropout=model_cfg.get("dropout"),
            state_embeddings=state_embeddings,
            features=build.state.features,
            decoder_mode=decoder_mode,
            use_edge_features=False,
            tf_peak_edge_feature_dim=int(
                model_cfg.get("tf_peak_edge_feature_dim", 3)
            ),
            peak_gene_edge_feature_dim=int(
                model_cfg.get("peak_gene_edge_feature_dim", 2)
            ),
            tf_gene_edge_feature_dim=int(
                model_cfg.get("tf_gene_edge_feature_dim", 1)
            ),
            aggregator_mode=model_cfg.get(
                "aggregator_mode", "mean"
            ),
            aggregator_top_k=int(model_cfg.get("aggregator_top_k", 10)),
        )
    freeze_state_backbone = bool(train_cfg.get("freeze_state_backbone", True))
    if not freeze_state_backbone:
        raise ValueError("The minimal Stage-2 initializer and encoder must remain frozen.")
    frozen_modules = [model.initializer, model.encoder]
    if freeze_state_backbone:
        state_model.requires_grad_(False)
        state_model.eval()
        for module in frozen_modules:
            module.requires_grad_(False)
            module.eval()
        for relation in ("cg", "cp"):
            model.decoders.decoders[relation].requires_grad_(False)
            model.decoders.decoders[relation].eval()
    encoder_parameters = [
        parameter
        for module in frozen_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    decoder_parameters = [
        parameter for parameter in model.decoders.parameters() if parameter.requires_grad
    ]
    parameter_groups = []
    if encoder_parameters:
        parameter_groups.append(
            {"params": encoder_parameters, "lr": float(train_cfg.get("encoder_lr", 1e-4))}
        )
    if decoder_parameters:
        parameter_groups.append(
            {"params": decoder_parameters, "lr": float(train_cfg.get("decoder_lr", 1e-3))}
        )
    if not parameter_groups:
        raise ValueError("Stage-2 training has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    generator = torch.Generator().manual_seed(int(config.get("seed", 1)))
    batch_size = int(train_cfg.get("batch_size_edges_per_relation", 256))
    for relation in ("tp", "pg"):
        if int(train_cfg.get(f"negative_ratio_{relation}", 1)) != 1:
            raise ValueError("The minimal Stage-2 negative ratio is fixed at 1.")
    if int(train_cfg.get("pg_hard_negative_pool_size", 0)) != 0 or float(
        train_cfg.get("pg_hard_negative_fraction", 0.0)
    ) != 0.0:
        raise ValueError("Hard-negative mining is disabled in shared_backbone_minimal_v1.")
    all_positive = {relation: _all_positive_edges(build, relation) for relation in ("tp", "pg")}
    if "tg" in build.splits:
        all_positive["tg"] = _all_positive_edges(build, "tg")
    stage2_sampler = _build_context_path_sampler(
        build, train_cfg, int(config.get("seed", 1))
    )
    validation_sampler = _build_context_path_sampler(
        build, train_cfg, int(config.get("seed", 1)) + 1_000_003
    )
    validation_protocol = _build_fixed_validation_protocol(
        build,
        train_cfg,
        all_positive,
        validation_sampler,
        seed=int(train_cfg.get("validation_seed", int(config.get("seed", 1)) + 20_003)),
    )
    fixed_train_negatives = {
        relation: _fixed_relation_negative_edges(
            build,
            relation,
            build.splits[relation]["train"].edge_index,
            all_positive[relation],
            generator,
        )
        for relation in ("tp", "pg")
    }
    rows: list[dict[str, float | str]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    early_stopping_warmup = int(train_cfg.get("early_stopping_warmup", 0))
    if early_stopping_patience < 0 or early_stopping_warmup < 0:
        raise ValueError("Early-stopping patience and warmup must be non-negative.")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative.")
    balance_tp_pg_queries = bool(train_cfg.get("balance_tp_pg_queries", False))
    pg_confidence_power = 1.0
    pg_confidence_floor = 0.0
    pg_hard_negative_pool_size = int(train_cfg.get("pg_hard_negative_pool_size", 0))
    pg_hard_negative_fraction = float(train_cfg.get("pg_hard_negative_fraction", 0.0))
    for epoch in range(1, int(train_cfg.get("epochs", 50)) + 1):
        model.train()
        if freeze_state_backbone:
            for module in frozen_modules:
                module.eval()
            for relation in ("cg", "cp"):
                model.decoders.decoders[relation].eval()
        optimizer.zero_grad()
        sampled_batch = None
        if stage2_sampler is None:
            training_data = build.data
            training_features = build.state.features
            training_global_ids = None
            local_is_tf = model.is_tf
        else:
            sampled_batch = stage2_sampler.sample(
                _sample_seed_cells(stage2_sampler, generator)
            )
            training_data = sampled_batch.data
            training_features = _batch_features(build, sampled_batch)
            training_global_ids = sampled_batch.global_node_ids
            local_is_tf = sampled_batch.data["gene"].is_tf.bool()
        embeddings = model.encode(
            training_data,
            training_features,
            global_node_ids=training_global_ids,
        )
        row: dict[str, float | str] = {
            "epoch": float(epoch),
            "stage1_backbone_frozen": float(freeze_state_backbone),
            "decoder_lr": float(train_cfg.get("decoder_lr", 1e-3)),
            "pg_confidence_power": pg_confidence_power,
            "pg_confidence_floor": pg_confidence_floor,
            "pg_hard_negative_pool_size": float(pg_hard_negative_pool_size),
            "pg_hard_negative_fraction": pg_hard_negative_fraction,
            "stage2_trainable_parameter_count": float(
                sum(parameter.numel() for group in parameter_groups for parameter in group["params"])
            ),
        }
        row.update(validation_protocol.diagnostics)
        if sampled_batch is None:
            row["sampler_name"] = "full_graph"
        else:
            row.update(_sampler_metrics(dict(sampled_batch.diagnostics)))
        relation_losses: dict[str, torch.Tensor] = {}
        sampled_edges: dict[str, torch.Tensor] = {}
        sampled_features: dict[str, torch.Tensor | None] = {}
        relation_batches = {}
        for relation in ("tp", "pg"):
            edge_type = REGULATORY_RELATIONS[relation]
            sampler_negatives = None
            if sampled_batch is None:
                train_split = build.splits[relation]["train"]
                n_train = int(train_split.edge_index.shape[1])
                if n_train < 1:
                    raise ValueError(f"No training {relation} candidates are available.")
                sampled_indices = torch.randperm(n_train, generator=generator)[
                    : min(batch_size, n_train)
                ]
                positive = train_split.edge_index[:, sampled_indices]
                weights = torch.ones(len(sampled_indices), dtype=torch.float32)
                positive_edge_features = None
                sampler_negatives = fixed_train_negatives[relation][:, sampled_indices]
                relation_all_positive = all_positive[relation]
            else:
                positive = sampled_batch.data[edge_type].edge_index
                weights = getattr(
                    sampled_batch.data[edge_type],
                    "edge_weight",
                    torch.ones(positive.shape[1]),
                )
                positive_edge_features = getattr(
                    sampled_batch.data[edge_type], "edge_features", None
                )
                sampler_negatives = sampled_batch.negative_edge_index[edge_type]
                relation_all_positive = _local_all_positive(
                    build, relation, sampled_batch
                )
            relation_batches[relation] = [
                positive,
                weights,
                positive_edge_features,
                relation_all_positive,
                sampler_negatives,
            ]

        if balance_tp_pg_queries:
            relation_budget = min(
                batch_size,
                *(int(relation_batches[relation][0].shape[1]) for relation in ("tp", "pg")),
            )
            if relation_budget < 1:
                raise ValueError("Balanced TP/PG sampling requires both relation types.")
            for relation in ("tp", "pg"):
                positive, weights, positive_edge_features, relation_all_positive, sampler_negatives = relation_batches[relation]
                selected = torch.randperm(positive.shape[1], generator=generator)[:relation_budget]
                positive = positive[:, selected]
                weights = weights[selected]
                if positive_edge_features is not None:
                    positive_edge_features = positive_edge_features[selected]
                if sampler_negatives is not None:
                    ratio = sampler_negatives.shape[1] // relation_batches[relation][0].shape[1]
                    sampler_negatives = sampler_negatives.reshape(2, -1, ratio)[:, selected, :].reshape(2, -1)
                relation_batches[relation] = [
                    positive,
                    weights,
                    positive_edge_features,
                    relation_all_positive,
                    sampler_negatives,
                ]
        row["tp_pg_queries_balanced"] = float(balance_tp_pg_queries)

        for relation in ("tp", "pg"):
            edge_type = REGULATORY_RELATIONS[relation]
            positive, weights, positive_edge_features, relation_all_positive, sampler_negatives = relation_batches[relation]
            sampled_edges[relation] = positive
            sampled_features[relation] = positive_edge_features
            row[f"{relation}_positive_count"] = float(positive.shape[1])
            if relation == "pg":
                resolved_negative_ratio = (
                    int(train_cfg.get("negative_ratio_pg", 1))
                    if sampler_negatives is None
                    else sampler_negatives.shape[1] // positive.shape[1]
                )
                row["pg_hard_negative_count"] = float(
                    positive.shape[1]
                    * int(round(resolved_negative_ratio * pg_hard_negative_fraction))
                    * int(pg_hard_negative_pool_size > 0)
                )
            loss = _relation_loss(
                model,
                embeddings,
                relation,
                positive,
                weights,
                positive_edge_features,
                relation_all_positive,
                int(training_data[edge_type[2]].num_nodes),
                1,
                float(train_cfg.get(f"temperature_{relation}", 0.1)),
                generator,
                is_tf=local_is_tf,
                negative_edge_index=sampler_negatives,
                confidence_power=pg_confidence_power if relation == "pg" else 1.0,
                confidence_floor=pg_confidence_floor if relation == "pg" else 0.0,
                hard_negative_pool_size=(
                    pg_hard_negative_pool_size if relation == "pg" else 0
                ),
                hard_negative_fraction=(
                    pg_hard_negative_fraction if relation == "pg" else 0.0
                ),
            )
            relation_losses[relation] = loss
            row[f"{relation}_loss"] = float(loss.detach())

        tf_gene_loss = None
        if (
            loss_config.weights.lambda_tf_gene > 0
            and "tg" in build.splits
            and build.splits["tg"]["train"].edge_index.shape[1] > 0
        ):
            tg_split = build.splits["tg"]["train"]
            if sampled_batch is not None:
                tg_split = _localize_split(
                    tg_split,
                    ("gene", "targets", "gene"),
                    sampled_batch,
                )
            tg_train = tg_split.edge_index
            tg_features = tg_split.edge_features
            if tg_train.shape[1] > batch_size:
                selected = torch.randperm(
                    tg_train.shape[1], generator=generator
                )[:batch_size]
                tg_train = tg_train[:, selected]
                if tg_features is not None:
                    tg_features = tg_features[selected]
            if tg_train.shape[1] > 0:
                tg_all_positive = (
                    all_positive["tg"]
                    if sampled_batch is None
                    else _local_all_positive(build, "tg", sampled_batch)
                )
                tf_gene_loss = _tf_gene_loss(
                    model,
                    embeddings,
                    tg_train,
                    tg_features,
                    tg_all_positive,
                    int(training_data["gene"].num_nodes),
                    int(train_cfg.get("negative_ratio_tg", 5)),
                    generator,
                    is_tf=local_is_tf,
                )
        consistency = None
        if loss_config.weights.lambda_consistency > 0:
            consistency = _sampled_consistency_loss(
                model,
                embeddings,
                sampled_edges,
                sampled_features,
                loss_config.consistency_mode,
                is_tf=local_is_tf,
            )
        path_loss = None
        row["path_positive_count"] = 0.0
        if loss_config.weights.lambda_path > 0:
            train_tp = build.splits["tp"]["train"]
            train_pg = build.splits["pg"]["train"]
            path_all_tp = all_positive["tp"]
            path_all_pg = all_positive["pg"]
            if sampled_batch is not None:
                train_tp = _localize_split(train_tp, REGULATORY_RELATIONS["tp"], sampled_batch)
                train_pg = _localize_split(train_pg, REGULATORY_RELATIONS["pg"], sampled_batch)
                path_all_tp = _local_all_positive(build, "tp", sampled_batch)
                path_all_pg = _local_all_positive(build, "pg", sampled_batch)
            path_loss, path_count = _split_path_ranking_loss(
                model,
                embeddings,
                train_tp,
                train_pg,
                path_all_tp,
                path_all_pg,
                local_is_tf,
                generator,
                max_paths=int(train_cfg.get("path_batch_size", 256)),
                negative_ratio=int(train_cfg.get("negative_ratio_path", 3)),
                temperature=loss_config.path_temperature,
            )
            row["path_positive_count"] = float(path_count)
        anchor = (
            embedding_anchor_loss(
                embeddings,
                state_embeddings
                if sampled_batch is None
                else {
                    node_type: state_embeddings[node_type][
                        sampled_batch.global_node_ids[node_type]
                    ]
                    for node_type in embeddings
                },
            )
            if loss_config.weights.lambda_anchor > 0
            else None
        )
        loss_output = loss_manager(
            tf_peak_loss=relation_losses["tp"],
            peak_gene_loss=relation_losses["pg"],
            tf_gene_loss=tf_gene_loss,
            consistency_loss=consistency,
            path_loss=path_loss,
            clustering_loss=None,
            anchor_loss=anchor,
            allow_missing={"tf_gene", "consistency", "path", "cluster"},
        )
        loss_output.total.backward()
        optimizer.step()
        row.update(loss_output.detached_metrics())
        row["graph_loss"] = float(
            (0.5 * (relation_losses["tp"] + relation_losses["pg"])).detach()
        )
        row["grn_loss"] = row["tf_gene_loss"]
        rows.append(row)

        validation_losses: dict[str, torch.Tensor] = {}
        row.update(
            {
                "validation_tf_peak_loss": float("nan"),
                "validation_peak_gene_loss": float("nan"),
                "validation_tf_gene_loss": float("nan"),
                "validation_path_loss": float("nan"),
                "validation_anchor_loss": float("nan"),
                "validation_path_status": (
                    "disabled"
                    if loss_config.weights.lambda_path <= 0
                    else (
                        "active" if validation_protocol.paths is not None else "skipped"
                    )
                ),
            }
        )
        model.eval()
        with torch.no_grad():
            validation_embeddings = model.encode(
                validation_protocol.data,
                validation_protocol.features,
                global_node_ids=validation_protocol.global_node_ids,
            )
            for relation in ("tp", "pg"):
                split = validation_protocol.splits[relation]
                validation_losses[relation] = _relation_loss(
                    model,
                    validation_embeddings,
                    relation,
                    split.edge_index,
                    split.edge_weight,
                    split.edge_features,
                    validation_protocol.all_positive[relation],
                    int(
                        validation_protocol.data[
                            REGULATORY_RELATIONS[relation][2]
                        ].num_nodes
                    ),
                    int(train_cfg.get(f"negative_ratio_{relation}", 1)),
                    float(train_cfg.get(f"temperature_{relation}", 0.1)),
                    generator,
                    is_tf=validation_protocol.is_tf,
                    negative_edge_index=validation_protocol.negative_edges[relation],
                    confidence_power=pg_confidence_power if relation == "pg" else 1.0,
                    confidence_floor=pg_confidence_floor if relation == "pg" else 0.0,
                )
                row[f"validation_{'tf_peak' if relation == 'tp' else 'peak_gene'}_loss"] = float(
                    validation_losses[relation]
                )
            validation_tf_gene = None
            if (
                loss_config.weights.lambda_tf_gene > 0
                and "tg" in validation_protocol.splits
            ):
                tg_validation = validation_protocol.splits["tg"]
                if tg_validation.edge_index.shape[1] > 0:
                    validation_tf_gene = _tf_gene_loss(
                        model,
                        validation_embeddings,
                        tg_validation.edge_index,
                        tg_validation.edge_features,
                        validation_protocol.all_positive["tg"],
                        int(validation_protocol.data["gene"].num_nodes),
                        int(train_cfg.get("negative_ratio_tg", 5)),
                        generator,
                        is_tf=validation_protocol.is_tf,
                        negative_edge_index=validation_protocol.negative_edges["tg"],
                    )
                    row["validation_tf_gene_loss"] = float(validation_tf_gene)
            validation_path = None
            if loss_config.weights.lambda_path > 0 and validation_protocol.paths is not None:
                validation_path = _fixed_path_ranking_loss(
                    model,
                    validation_embeddings,
                    validation_protocol.splits["tp"],
                    validation_protocol.splits["pg"],
                    validation_protocol.is_tf,
                    validation_protocol.paths,
                    temperature=loss_config.path_temperature,
                )
                row["validation_path_loss"] = float(validation_path)
            if set(validation_losses) == {"tp", "pg"}:
                validation_anchor = (
                    embedding_anchor_loss(
                        validation_embeddings,
                        state_embeddings
                        if validation_protocol.batch is None
                        else {
                            node_type: state_embeddings[node_type][
                                validation_protocol.batch.global_node_ids[node_type]
                            ]
                            for node_type in validation_embeddings
                        },
                    )
                    if loss_config.weights.lambda_anchor > 0
                    else None
                )
                if validation_anchor is not None:
                    row["validation_anchor_loss"] = float(validation_anchor)
                validation_output = loss_manager(
                    tf_peak_loss=validation_losses["tp"],
                    peak_gene_loss=validation_losses["pg"],
                    tf_gene_loss=validation_tf_gene,
                    path_loss=validation_path,
                    anchor_loss=validation_anchor,
                    allow_missing={"tf_gene", "consistency", "path", "cluster"},
                )
                selection_loss = float(validation_output.total)
            else:
                selection_loss = float(row["total_loss"])
        row["validation_loss"] = selection_loss
        improved = selection_loss < best_loss - early_stopping_min_delta
        if improved:
            best_loss = selection_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        should_stop = (
            early_stopping_patience > 0
            and epoch >= early_stopping_warmup
            and epochs_without_improvement >= early_stopping_patience
        )
        row["validation_improved"] = float(improved)
        row["best_validation_loss"] = best_loss
        row["best_epoch"] = float(best_epoch)
        row["epochs_without_improvement"] = float(epochs_without_improvement)
        row["early_stopped"] = float(should_stop)
        if should_stop:
            break

    if best_state is None:
        raise RuntimeError("Stage-2 training produced no checkpoint.")
    model.load_state_dict(best_state)
    metrics = pd.DataFrame(rows)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "logs" / "stage2_metrics.csv", index=False)
    sampler_columns = [
        column
        for column in metrics.columns
        if column == "epoch" or column == "sampler_name" or column.startswith("sampler_")
    ]
    metrics.loc[:, sampler_columns].to_csv(
        output_dir / "logs" / "stage2_sampler_diagnostics.csv", index=False
    )
    validation_columns = [
        column
        for column in metrics.columns
        if column == "epoch"
        or column == "stage1_backbone_frozen"
        or column.startswith("validation_")
    ]
    metrics.loc[:, validation_columns].to_csv(
        output_dir / "logs" / "stage2_validation_diagnostics.csv", index=False
    )
    save_checkpoint(
        output_dir / "checkpoints" / "stage2_best.pt",
        stage="stage2",
        model_state=best_state,
        model_config=model.model_config,
        config=config,
        names=build.state.names,
        tf_gene_indices=build.tf_gene_indices,
    )
    return model, metrics
