"""Configurable, independently testable objectives for redesigned HERTA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Collection, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def weighted_infonce_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    edge_weight: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Compute confidence-weighted InfoNCE, normalized by total edge weight."""

    if tau <= 0:
        raise ValueError("tau must be positive.")
    if pos_logits.ndim != 1:
        raise ValueError("pos_logits must be one-dimensional.")
    if neg_logits.ndim != 2 or neg_logits.shape[0] != len(pos_logits):
        raise ValueError("neg_logits must have shape [n_positive, n_negative].")
    if edge_weight.shape != pos_logits.shape:
        raise ValueError("edge_weight must align with pos_logits.")
    if not bool(
        torch.isfinite(pos_logits).all()
        and torch.isfinite(neg_logits).all()
        and torch.isfinite(edge_weight).all()
    ):
        raise ValueError("InfoNCE inputs must contain finite values.")
    if bool((edge_weight < 0).any()):
        raise ValueError("edge_weight must be non-negative.")
    logits = torch.cat([pos_logits[:, None], neg_logits], dim=1) / tau
    log_prob = logits[:, 0] - torch.logsumexp(logits, dim=1)
    return -(edge_weight * log_prob).sum() / edge_weight.sum().clamp_min(1e-8)


def wnn_neighborhood_infonce_loss(
    cell_embeddings: torch.Tensor,
    anchor_ids: torch.Tensor,
    neighbor_ids: torch.Tensor,
    neighbor_weights: torch.Tensor,
    *,
    num_negatives: int = 5,
    temperature: float = 0.2,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Contrast WNN neighbors against fixed-count non-neighbor cells."""

    if cell_embeddings.ndim != 2 or cell_embeddings.shape[0] < 2:
        raise ValueError("cell_embeddings must have shape [n_cells, dim].")
    if anchor_ids.ndim != 1 or neighbor_ids.ndim != 2:
        raise ValueError("anchor_ids and neighbor_ids must be 1D and 2D tensors.")
    if neighbor_ids.shape != neighbor_weights.shape or neighbor_ids.shape[0] != len(anchor_ids):
        raise ValueError("WNN neighbors and weights must align with anchor_ids.")
    if num_negatives < 1 or temperature <= 0:
        raise ValueError("num_negatives and temperature must be positive.")
    if not bool(torch.isfinite(cell_embeddings).all() and torch.isfinite(neighbor_weights).all()):
        raise ValueError("WNN loss inputs must contain finite values.")
    if bool((neighbor_weights < 0).any()) or float(neighbor_weights.sum()) <= 0:
        raise ValueError("WNN neighbor weights must be non-negative with positive sum.")
    n_cells = int(cell_embeddings.shape[0])
    if bool((anchor_ids < 0).any()) or bool((anchor_ids >= n_cells).any()):
        raise IndexError("WNN anchor IDs are out of range.")
    if bool((neighbor_ids < 0).any()) or bool((neighbor_ids >= n_cells).any()):
        raise IndexError("WNN neighbor IDs are out of range.")
    negatives = torch.empty(
        (*neighbor_ids.shape, num_negatives), dtype=torch.long
    )
    for row, anchor in enumerate(anchor_ids.tolist()):
        forbidden = torch.unique(
            torch.cat([anchor_ids.new_tensor([anchor]), neighbor_ids[row]])
        )
        if len(forbidden) >= n_cells:
            raise ValueError("A WNN anchor has no eligible negative cells.")
        candidates = torch.randint(
            n_cells,
            (neighbor_ids.shape[1], num_negatives),
            generator=generator,
        )
        invalid = torch.isin(candidates, forbidden)
        while bool(invalid.any()):
            candidates[invalid] = torch.randint(
                n_cells, (int(invalid.sum()),), generator=generator
            )
            invalid = torch.isin(candidates, forbidden)
        negatives[row] = candidates
    unit = F.normalize(cell_embeddings, dim=1)
    anchor_device = anchor_ids.to(cell_embeddings.device)
    neighbor_device = neighbor_ids.to(cell_embeddings.device)
    anchors = unit[anchor_device][:, None, :].expand(-1, neighbor_ids.shape[1], -1)
    positive_logits = (anchors * unit[neighbor_device]).sum(dim=-1).reshape(-1)
    negative_logits = (
        anchors[:, :, None, :] * unit[negatives.to(cell_embeddings.device)]
    ).sum(dim=-1).reshape(-1, num_negatives)
    return weighted_infonce_loss(
        positive_logits,
        negative_logits,
        neighbor_weights.to(cell_embeddings.device).reshape(-1),
        temperature,
    )


def link_prediction_bce_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    *,
    positive_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary link-prediction loss for observed, TF-peak, or peak-gene edges."""

    positive = pos_logits.reshape(-1)
    negative = neg_logits.reshape(-1)
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("Link prediction BCE requires positive and negative logits.")
    if not bool(torch.isfinite(positive).all() and torch.isfinite(negative).all()):
        raise ValueError("Link prediction logits must contain finite values.")
    logits = torch.cat([positive, negative])
    labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)])
    if positive_weight is None:
        pos_weight = torch.tensor(
            max(float(negative.numel()) / float(positive.numel()), 1.0),
            dtype=logits.dtype,
            device=logits.device,
        )
    else:
        pos_weight = torch.as_tensor(
            positive_weight,
            dtype=logits.dtype,
            device=logits.device,
        )
        if pos_weight.numel() != 1 or not bool(torch.isfinite(pos_weight)) or float(
            pos_weight
        ) <= 0:
            raise ValueError("positive_weight must be one finite positive value.")
    return F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
    )


def contrastive_info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor | None = None,
    *,
    temperature: float = 0.2,
    normalize: bool = True,
) -> torch.Tensor:
    """InfoNCE for paired representations with in-batch or explicit negatives."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if (
        anchor.ndim != 2
        or positive.shape != anchor.shape
        or anchor.shape[0] == 0
    ):
        raise ValueError("anchor and positive must share non-empty shape [batch, dim].")
    if not bool(torch.isfinite(anchor).all() and torch.isfinite(positive).all()):
        raise ValueError("Contrastive representations must contain finite values.")
    query = F.normalize(anchor, dim=-1) if normalize else anchor
    paired = F.normalize(positive, dim=-1) if normalize else positive
    if negatives is None:
        logits = query @ paired.t() / temperature
        labels = torch.arange(len(query), device=query.device)
        return F.cross_entropy(logits, labels)
    if (
        negatives.ndim != 3
        or negatives.shape[0] != anchor.shape[0]
        or negatives.shape[2] != anchor.shape[1]
        or negatives.shape[1] == 0
    ):
        raise ValueError("negatives must have shape [batch, n_negative, dim].")
    if not bool(torch.isfinite(negatives).all()):
        raise ValueError("Contrastive negatives must contain finite values.")
    candidate_negatives = (
        F.normalize(negatives, dim=-1) if normalize else negatives
    )
    positive_logits = (query * paired).sum(dim=-1, keepdim=True)
    negative_logits = torch.einsum("bd,bnd->bn", query, candidate_negatives)
    logits = torch.cat([positive_logits, negative_logits], dim=1) / temperature
    labels = torch.zeros(len(query), dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


def weighted_multi_positive_contrastive_loss(
    anchor: torch.Tensor,
    candidate: torch.Tensor,
    positive_indices: torch.Tensor,
    positive_weights: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Weighted multi-positive InfoNCE for label-free cell neighborhoods."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if anchor.ndim != 2 or candidate.shape != anchor.shape or anchor.shape[0] == 0:
        raise ValueError("anchor and candidate must share non-empty shape [cells, dim].")
    if positive_indices.ndim != 2 or positive_indices.shape != positive_weights.shape:
        raise ValueError("positive indices and weights must share shape [cells, positives].")
    if positive_indices.shape[0] != anchor.shape[0] or positive_indices.shape[1] == 0:
        raise ValueError("positive rows must align with cells and contain a positive.")
    if bool((positive_indices < 0).any()) or bool((positive_indices >= len(candidate)).any()):
        raise ValueError("positive indices are outside the candidate cell range.")
    if not bool(
        torch.isfinite(anchor).all()
        and torch.isfinite(candidate).all()
        and torch.isfinite(positive_weights).all()
    ):
        raise ValueError("multi-positive contrastive inputs must be finite.")
    if bool((positive_weights < 0).any()) or bool((positive_weights.sum(dim=1) <= 0).any()):
        raise ValueError("every cell must have positive non-negative WNN weight.")

    logits = F.normalize(anchor, dim=-1) @ F.normalize(candidate, dim=-1).t()
    logits = logits / temperature
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    selected = log_probability.gather(1, positive_indices.long())
    normalized_weight = positive_weights / positive_weights.sum(dim=1, keepdim=True)
    return -(normalized_weight * selected).sum(dim=1).mean()


def tf_gene_bce_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    """Return class-balanced BCE over known and sampled TF-target pairs."""

    return link_prediction_bce_loss(pos_logits, neg_logits)


def tf_peak_gene_consistency_loss(
    direct_tf_gene_score: torch.Tensor,
    aggregated_tf_gene_score: torch.Tensor,
    *,
    mode: str = "mse",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Align direct TF-gene probabilities with two-hop TF→peak→gene scores."""

    if (
        direct_tf_gene_score.shape != aggregated_tf_gene_score.shape
        or direct_tf_gene_score.numel() == 0
    ):
        raise ValueError("Direct and aggregated TF-gene scores must share a non-empty shape.")
    if mode not in {"mse", "smooth_l1", "symmetric_bce"}:
        raise ValueError("mode must be 'mse', 'smooth_l1', or 'symmetric_bce'.")
    for name, score in (
        ("direct", direct_tf_gene_score),
        ("aggregated", aggregated_tf_gene_score),
    ):
        if not bool(torch.isfinite(score).all()) or not bool(
            ((score >= 0) & (score <= 1)).all()
        ):
            raise ValueError(f"{name} TF-gene scores must be finite probabilities.")
    if mode == "mse":
        return F.mse_loss(direct_tf_gene_score, aggregated_tf_gene_score)
    if mode == "smooth_l1":
        return F.smooth_l1_loss(
            direct_tf_gene_score,
            aggregated_tf_gene_score,
        )
    direct = direct_tf_gene_score.clamp(eps, 1 - eps)
    aggregate = aggregated_tf_gene_score.clamp(eps, 1 - eps)
    return 0.5 * (
        F.binary_cross_entropy(direct, aggregate.detach())
        + F.binary_cross_entropy(aggregate, direct.detach())
    )


def regulatory_path_ranking_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Rank split-matched complete TF-peak-gene paths above corruptions."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if positive_logits.ndim != 1 or positive_logits.numel() == 0:
        raise ValueError("positive path logits must be a non-empty vector.")
    if negative_logits.ndim != 2 or negative_logits.shape[0] != len(positive_logits):
        raise ValueError("negative path logits must have shape [paths, negatives].")
    if negative_logits.shape[1] == 0 or not bool(
        torch.isfinite(positive_logits).all() and torch.isfinite(negative_logits).all()
    ):
        raise ValueError("path ranking logits must be non-empty and finite.")
    logits = torch.cat([positive_logits[:, None], negative_logits], dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


def prototype_clustering_loss(
    embeddings: torch.Tensor,
    assignments: torch.Tensor,
    prototypes: torch.Tensor | None = None,
    *,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Optional regulatory-state compactness loss around cluster prototypes."""

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must have non-empty shape [n_cells, dim].")
    if assignments.ndim != 1 or len(assignments) != len(embeddings):
        raise ValueError("assignments must align with embeddings.")
    if assignments.dtype != torch.long:
        raise ValueError("assignments must use torch.long cluster indices.")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings must contain finite values.")
    keep = assignments != ignore_index
    if not bool(keep.any()):
        return embeddings.sum() * 0.0
    selected_embeddings = embeddings[keep]
    selected_assignments = assignments[keep]
    if bool((selected_assignments < 0).any()):
        raise ValueError("Cluster assignments must be non-negative or ignore_index.")
    if prototypes is None:
        prototype_rows: list[torch.Tensor] = []
        max_cluster = int(selected_assignments.max())
        for cluster in range(max_cluster + 1):
            members = selected_embeddings[selected_assignments == cluster]
            if members.numel() == 0:
                prototype_rows.append(
                    torch.zeros(
                        embeddings.shape[1],
                        dtype=embeddings.dtype,
                        device=embeddings.device,
                    )
                )
            else:
                prototype_rows.append(members.mean(dim=0))
        prototypes = torch.stack(prototype_rows)
    if (
        prototypes.ndim != 2
        or prototypes.shape[1] != embeddings.shape[1]
        or int(selected_assignments.max()) >= len(prototypes)
    ):
        raise ValueError("prototypes must cover every assigned cluster and embedding dimension.")
    if not bool(torch.isfinite(prototypes).all()):
        raise ValueError("prototypes must contain finite values.")
    targets = prototypes.to(
        device=embeddings.device,
        dtype=embeddings.dtype,
    )[selected_assignments]
    return (selected_embeddings - targets).pow(2).sum(dim=1).mean()


@dataclass(frozen=True)
class LossWeights:
    """All configurable lambda weights in the HERTA redesign objective."""

    lambda_graph: float = 1.0
    lambda_contrast: float = 0.0
    lambda_tf_peak: float = 1.0
    lambda_peak_gene: float = 1.0
    lambda_tf_gene: float = 0.0
    lambda_consistency: float = 0.1
    lambda_path: float = 0.0
    lambda_cluster: float = 0.0
    lambda_anchor: float = 0.0

    def __post_init__(self) -> None:
        for field_info in fields(self):
            value = float(getattr(self, field_info.name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{field_info.name} must be finite and non-negative.")
            object.__setattr__(self, field_info.name, value)

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, float],
        *,
        defaults: "LossWeights | None" = None,
    ) -> "LossWeights":
        """Resolve canonical lambda names plus concise redesign aliases."""

        aliases = {
            "lambda_tp": "lambda_tf_peak",
            "lambda_pg": "lambda_peak_gene",
            "lambda_tg": "lambda_tf_gene",
            "lambda_cons": "lambda_consistency",
        }
        canonical = {field.name for field in fields(cls)}
        resolved: dict[str, float] = (
            {} if defaults is None else defaults.as_dict()
        )
        provided: set[str] = set()
        for key, value in config.items():
            target = aliases.get(key, key)
            if target in canonical:
                if target in provided and float(resolved[target]) != float(value):
                    raise ValueError(f"Conflicting values were provided for {target}.")
                resolved[target] = float(value)
                provided.add(target)
            elif str(key).startswith("lambda_"):
                raise ValueError(f"Unknown HERTA loss weight: {key}")
        return cls(**resolved)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1LossConfig:
    """Stage-1 defaults and non-weight loss settings."""

    weights: LossWeights = field(
        default_factory=lambda: LossWeights(
            lambda_graph=1.0,
            lambda_contrast=0.0,
            lambda_tf_peak=0.0,
            lambda_peak_gene=0.0,
            lambda_tf_gene=0.0,
            lambda_consistency=0.0,
            lambda_path=0.0,
            lambda_cluster=0.0,
            lambda_anchor=0.0,
        )
    )
    lambda_cg: float = 0.6
    lambda_cp: float = 0.4
    lambda_wnn: float = 0.0
    normalize_relation_weights: bool = True

    def __post_init__(self) -> None:
        lambda_cg = float(self.lambda_cg)
        lambda_cp = float(self.lambda_cp)
        if not np.isfinite(lambda_cg) or not np.isfinite(lambda_cp):
            raise ValueError("lambda_cg and lambda_cp must be finite.")
        if lambda_cg < 0 or lambda_cp < 0 or lambda_cg + lambda_cp <= 0:
            raise ValueError(
                "lambda_cg and lambda_cp must be non-negative with a positive sum."
            )
        if self.normalize_relation_weights:
            total = lambda_cg + lambda_cp
            lambda_cg /= total
            lambda_cp /= total
        object.__setattr__(self, "lambda_cg", lambda_cg)
        object.__setattr__(self, "lambda_cp", lambda_cp)
        lambda_wnn = float(self.lambda_wnn)
        if not np.isfinite(lambda_wnn) or lambda_wnn < 0:
            raise ValueError("lambda_wnn must be finite and non-negative.")
        object.__setattr__(self, "lambda_wnn", lambda_wnn)
    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> "Stage1LossConfig":
        legacy_keys = {
            "contrastive_mode",
            "contrastive_temperature",
            "temperature_contrastive",
            "wnn_neighbors",
            "wnn_self_weight",
            "mask_fraction",
        }.intersection(config)
        if legacy_keys:
            raise ValueError(
                "Stage 1 no longer accepts masked or contrastive settings; "
                "use WNN subgraph sampling with graph loss only."
            )
        defaults = cls()
        weights = LossWeights.from_mapping(
            {
                key: float(value)
                for key, value in config.items()
                if str(key).startswith("lambda_")
                and key not in {"lambda_cg", "lambda_cp", "lambda_wnn"}
            },
            defaults=defaults.weights,
        )
        active_auxiliaries = {
            name: value
            for name, value in weights.as_dict().items()
            if name != "lambda_graph" and value != 0.0
        }
        if active_auxiliaries:
            raise ValueError(
                "Stage 1 supports only lambda_graph; disable auxiliary weights: "
                f"{sorted(active_auxiliaries)}."
            )
        return cls(
            weights=weights,
            lambda_cg=float(config.get("lambda_cg", defaults.lambda_cg)),
            lambda_cp=float(config.get("lambda_cp", defaults.lambda_cp)),
            lambda_wnn=float(config.get("lambda_wnn", defaults.lambda_wnn)),
            normalize_relation_weights=bool(
                config.get(
                    "normalize_relation_weights",
                    defaults.normalize_relation_weights,
                )
            ),
        )


@dataclass(frozen=True)
class Stage2LossConfig:
    """Minimal Stage-2 TP/PG-only objective settings."""

    weights: LossWeights = field(
        default_factory=lambda: LossWeights(
            lambda_graph=0.0,
            lambda_contrast=0.0,
            lambda_tf_peak=1.0,
            lambda_peak_gene=1.0,
            lambda_tf_gene=0.0,
            lambda_consistency=0.0,
            lambda_path=0.0,
            lambda_cluster=0.0,
            lambda_anchor=0.0,
        )
    )
    consistency_mode: str = "mse"
    path_temperature: float = 0.2

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> "Stage2LossConfig":
        defaults = cls()
        weight_mapping = {
            key: float(value)
            for key, value in config.items()
            if str(key).startswith("lambda_") and key != "lambda_grn"
        }
        if "lambda_grn" in config and "lambda_tf_gene" not in weight_mapping:
            weight_mapping["lambda_tf_gene"] = float(config["lambda_grn"])
        weights = LossWeights.from_mapping(
            weight_mapping,
            defaults=defaults.weights,
        )
        consistency_mode = str(config.get("consistency_mode", "mse"))
        if consistency_mode not in {"mse", "smooth_l1", "symmetric_bce"}:
            raise ValueError(
                "consistency_mode must be 'mse', 'smooth_l1', or 'symmetric_bce'."
            )
        path_temperature = float(config.get("path_temperature", 0.2))
        if not np.isfinite(path_temperature) or path_temperature <= 0:
            raise ValueError("path_temperature must be finite and positive.")
        return cls(
            weights=weights,
            consistency_mode=consistency_mode,
            path_temperature=path_temperature,
        )


@dataclass(frozen=True)
class MultiTaskLossOutput:
    """Total loss together with raw and weighted components for logging."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]
    weighted_components: dict[str, torch.Tensor]
    missing_zero_weight: tuple[str, ...]
    skipped_active: tuple[str, ...]
    statuses: dict[str, str]
    weights: dict[str, float]

    def detached_metrics(self) -> dict[str, float | str]:
        metrics: dict[str, float | str] = {}
        for name, value in self.components.items():
            status = self.statuses.get(name, "active")
            numeric = (
                float(value.detach()) if status == "active" else float("nan")
            )
            metrics[f"{name}_loss"] = numeric
            metrics[f"L_{name}"] = numeric
            metrics[f"{name}_status"] = status
            if name == "contrastive":
                metrics["L_contrast"] = numeric
                metrics["contrast_status"] = status
        metrics["loss"] = float(self.total.detach())
        metrics["total_loss"] = float(self.total.detach())
        for weight_name, value in self.weights.items():
            metrics[f"active_weight_{weight_name}"] = float(value)
        return metrics


class MultiTaskLossManager(nn.Module):
    """Combine separately computed HERTA objectives with explicit lambdas."""

    _COMPONENT_TO_WEIGHT = {
        "graph": "lambda_graph",
        "contrastive": "lambda_contrast",
        "tf_peak": "lambda_tf_peak",
        "peak_gene": "lambda_peak_gene",
        "tf_gene": "lambda_tf_gene",
        "consistency": "lambda_consistency",
        "path": "lambda_path",
        "cluster": "lambda_cluster",
        "anchor": "lambda_anchor",
    }

    def __init__(
        self,
        weights: LossWeights | Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if weights is None:
            self.weights = LossWeights()
        elif isinstance(weights, LossWeights):
            self.weights = weights
        else:
            self.weights = LossWeights.from_mapping(weights)
        self.register_buffer("_zero", torch.zeros((), dtype=torch.float32))

    @staticmethod
    def _scalar(name: str, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise ValueError(f"{name}_loss must be a scalar torch.Tensor.")
        if not bool(torch.isfinite(value)) or bool(value < 0):
            raise ValueError(f"{name}_loss must be finite and non-negative.")
        return value

    def forward(
        self,
        *,
        graph_loss: torch.Tensor | None = None,
        contrastive_loss: torch.Tensor | None = None,
        tf_peak_loss: torch.Tensor | None = None,
        peak_gene_loss: torch.Tensor | None = None,
        tf_gene_loss: torch.Tensor | None = None,
        consistency_loss: torch.Tensor | None = None,
        path_loss: torch.Tensor | None = None,
        clustering_loss: torch.Tensor | None = None,
        anchor_loss: torch.Tensor | None = None,
        allow_missing: Collection[str] = (),
    ) -> MultiTaskLossOutput:
        """Return the weighted total without starting or owning a training loop."""

        supplied = {
            "graph": self._scalar("graph", graph_loss),
            "contrastive": self._scalar("contrastive", contrastive_loss),
            "tf_peak": self._scalar("tf_peak", tf_peak_loss),
            "peak_gene": self._scalar("peak_gene", peak_gene_loss),
            "tf_gene": self._scalar("tf_gene", tf_gene_loss),
            "consistency": self._scalar("consistency", consistency_loss),
            "path": self._scalar("path", path_loss),
            "cluster": self._scalar("cluster", clustering_loss),
            "anchor": self._scalar("anchor", anchor_loss),
        }
        devices = {
            value.device
            for value in supplied.values()
            if value is not None
        }
        if len(devices) > 1:
            raise ValueError("All loss components must be on the same device.")
        reference = next(
            (
                value
                for value in supplied.values()
                if value is not None
            ),
            self._zero,
        )
        zero = reference * 0.0
        components: dict[str, torch.Tensor] = {}
        weighted: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        skipped: list[str] = []
        statuses: dict[str, str] = {}
        allowed_missing = set(allow_missing)
        unknown_allowed = allowed_missing.difference(self._COMPONENT_TO_WEIGHT)
        if unknown_allowed:
            raise ValueError(
                f"allow_missing contains unknown loss components: {sorted(unknown_allowed)}"
            )
        for name, weight_name in self._COMPONENT_TO_WEIGHT.items():
            value = supplied[name]
            weight = getattr(self.weights, weight_name)
            if value is None:
                if weight > 0:
                    if name not in allowed_missing:
                        raise ValueError(
                            f"{name}_loss is required because {weight_name}={weight}."
                        )
                    skipped.append(name)
                    statuses[name] = "skipped"
                else:
                    missing.append(name)
                    statuses[name] = "disabled"
                value = zero
            elif weight == 0:
                statuses[name] = "disabled"
            else:
                statuses[name] = "active"
            components[name] = value
            weighted[name] = value * weight
        total = torch.stack(list(weighted.values())).sum()
        return MultiTaskLossOutput(
            total=total,
            components=components,
            weighted_components=weighted,
            missing_zero_weight=tuple(missing),
            skipped_active=tuple(skipped),
            statuses=statuses,
            weights=self.weights.as_dict(),
        )


def stage1_objective(
    cg_loss: torch.Tensor,
    cp_loss: torch.Tensor,
    lambda_graph: float = 1.0,
    lambda_cg: float = 0.6,
    lambda_cp: float = 0.4,
    wnn_loss: torch.Tensor | None = None,
    lambda_wnn: float = 0.0,
    normalize_relation_weights: bool = True,
) -> torch.Tensor:
    """Return the configured observed-graph Stage-1 objective."""

    config = Stage1LossConfig.from_mapping(
        {
            "lambda_graph": lambda_graph,
            "lambda_cg": lambda_cg,
            "lambda_cp": lambda_cp,
            "lambda_wnn": lambda_wnn,
            "normalize_relation_weights": normalize_relation_weights,
        }
    )
    graph = config.weights.lambda_graph * (
        config.lambda_cg * cg_loss + config.lambda_cp * cp_loss
    )
    if config.lambda_wnn == 0:
        return graph
    if wnn_loss is None:
        raise ValueError("wnn_loss is required when lambda_wnn is positive.")
    return graph + config.lambda_wnn * wnn_loss


def stage2_objective(
    tp_loss: torch.Tensor,
    pg_loss: torch.Tensor,
    tf_gene_loss: torch.Tensor,
    consistency_loss: torch.Tensor,
    path_loss: torch.Tensor,
    anchor_loss: torch.Tensor,
    lambda_tf_peak: float = 1.0,
    lambda_peak_gene: float = 1.0,
    lambda_tf_gene: float = 0.5,
    lambda_consistency: float = 0.0,
    lambda_path: float = 0.5,
    lambda_anchor: float = 1e-3,
) -> torch.Tensor:
    """Combine the canonical Stage-2 regulatory objectives."""

    return (
        lambda_tf_peak * tp_loss
        + lambda_peak_gene * pg_loss
        + lambda_tf_gene * tf_gene_loss
        + lambda_consistency * consistency_loss
        + lambda_path * path_loss
        + lambda_anchor * anchor_loss
    )


def embedding_anchor_loss(
    current: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Mean squared drift from frozen Stage-1 embeddings."""

    losses = [(current[name] - reference[name].to(current[name].device)).pow(2).mean() for name in current]
    return torch.stack(losses).mean()
