"""Relation-specific link decoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from herta.model.regulatory_scoring import (
    DEFAULT_AGGREGATION_MODE,
    aggregate_tensor,
    canonical_aggregation_mode,
    path_score_tensor,
)


class MLPDecoder(nn.Module):
    """MLP edge scorer over concatenated source, target, and product embeddings."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([src, dst, src * dst], dim=-1)).squeeze(-1)


class RelationDecoders(nn.Module):
    """Five relation-specific decoders."""

    def __init__(self, hidden_dim: int, relation_names: list[str]) -> None:
        super().__init__()
        self.decoders = nn.ModuleDict({name: MLPDecoder(hidden_dim) for name in relation_names})

    def score(self, relation: str, src_emb: torch.Tensor, dst_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Score positive or candidate edges for a relation."""

        return self.decoders[relation](src_emb[edge_index[0]], dst_emb[edge_index[1]])

    def score_pairs(self, relation: str, src_emb: torch.Tensor, dst_emb: torch.Tensor, src_ids: torch.Tensor, dst_ids: torch.Tensor) -> torch.Tensor:
        """Score explicit source and destination id arrays."""

        return self.decoders[relation](src_emb[src_ids], dst_emb[dst_ids])


class ProjectedDotDecoder(nn.Module):
    """Relation-specific projected dot-product scorer."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim**-0.5

    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return (self.src(src) * self.dst(dst)).sum(dim=-1) * self.scale


class ProjectedRelationDecoders(nn.Module):
    """Projected dot-product decoders with the legacy scoring interface."""

    def __init__(self, hidden_dim: int, relation_names: list[str]) -> None:
        super().__init__()
        self.decoders = nn.ModuleDict({name: ProjectedDotDecoder(hidden_dim) for name in relation_names})

    def score(self, relation: str, src_emb: torch.Tensor, dst_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoders[relation](src_emb[edge_index[0]], dst_emb[edge_index[1]])

    def score_pairs(self, relation: str, src_emb: torch.Tensor, dst_emb: torch.Tensor, src_ids: torch.Tensor, dst_ids: torch.Tensor) -> torch.Tensor:
        return self.decoders[relation](src_emb[src_ids], dst_emb[dst_ids])


class _RegulatoryMLPDecoder(nn.Module):
    """Shared edge-feature-aware implementation for regulatory probabilities."""

    def __init__(
        self,
        hidden_dim: int,
        edge_feature_dim: int = 0,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if edge_feature_dim < 0:
            raise ValueError("edge_feature_dim must be non-negative.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.hidden_dim = int(hidden_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        inner_dim = int(mlp_hidden_dim or hidden_dim)
        if inner_dim <= 0:
            raise ValueError("mlp_hidden_dim must be positive.")
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3 + edge_feature_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, 1),
        )

    def _edge_inputs(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            source_embeddings.ndim != 2
            or target_embeddings.ndim != 2
            or source_embeddings.shape[1] != self.hidden_dim
            or target_embeddings.shape[1] != self.hidden_dim
        ):
            raise ValueError(
                f"Source and target embeddings must be two-dimensional with "
                f"hidden_dim={self.hidden_dim}."
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges].")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must use torch.long indices.")
        if edge_index.numel():
            if int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= len(
                source_embeddings
            ):
                raise ValueError("edge_index contains an out-of-range source index.")
            if int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= len(
                target_embeddings
            ):
                raise ValueError("edge_index contains an out-of-range target index.")
        if self.edge_feature_dim:
            if edge_features is None:
                edge_features = torch.zeros(
                    (edge_index.shape[1], self.edge_feature_dim),
                    dtype=source_embeddings.dtype,
                    device=source_embeddings.device,
                )
            if edge_features.shape != (edge_index.shape[1], self.edge_feature_dim):
                raise ValueError(
                    "edge_features must have shape "
                    f"({edge_index.shape[1]}, {self.edge_feature_dim})."
                )
            if not bool(torch.isfinite(edge_features).all()):
                raise ValueError("edge_features contains non-finite values.")
        elif edge_features is not None and edge_features.shape != (
            edge_index.shape[1],
            0,
        ):
            raise ValueError("This decoder was configured without edge features.")
        source = source_embeddings[edge_index[0]]
        target = target_embeddings[edge_index[1]]
        parts = [source, target, source * target]
        if self.edge_feature_dim:
            parts.append(
                edge_features.to(device=source.device, dtype=source.dtype)
            )
        return torch.cat(parts, dim=-1)

    def logits(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return unconstrained logits for later loss implementations."""

        inputs = self._edge_inputs(
            source_embeddings,
            target_embeddings,
            edge_index,
            edge_features,
        )
        return self.net(inputs).squeeze(-1)

    def forward(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a continuous learned score in ``[0, 1]``."""

        return torch.sigmoid(
            self.logits(
                source_embeddings,
                target_embeddings,
                edge_index,
                edge_features,
            )
        )


class TFPeakDecoder(_RegulatoryMLPDecoder):
    """Score TF-gene→peak candidates with optional motif features."""

    def forward(
        self,
        gene_embeddings: torch.Tensor,
        peak_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        *,
        is_tf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if is_tf is not None:
            if (
                is_tf.ndim != 1
                or len(is_tf) != len(gene_embeddings)
                or is_tf.dtype != torch.bool
            ):
                raise ValueError("is_tf must align with gene_embeddings.")
            source_ids = edge_index[0].to(is_tf.device)
            if edge_index.shape[1] and not bool(is_tf[source_ids].all()):
                raise ValueError("TF-peak edges must originate from TF gene nodes.")
        return super().forward(
            gene_embeddings,
            peak_embeddings,
            edge_index,
            edge_features,
        )


class PeakGeneDecoder(_RegulatoryMLPDecoder):
    """Score peak→gene candidates with optional distance/prior features."""


class TFGeneDecoder(_RegulatoryMLPDecoder):
    """Optional direct TF-gene baseline or weak-supervision decoder."""

    def forward(
        self,
        gene_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        *,
        is_tf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if is_tf is not None:
            if (
                is_tf.ndim != 1
                or len(is_tf) != len(gene_embeddings)
                or is_tf.dtype != torch.bool
            ):
                raise ValueError("is_tf must align with gene_embeddings.")
            source_ids = edge_index[0].to(is_tf.device)
            if edge_index.shape[1] and not bool(is_tf[source_ids].all()):
                raise ValueError("TF-gene edges must originate from TF gene nodes.")
        return super().forward(
            gene_embeddings,
            gene_embeddings,
            edge_index,
            edge_features,
        )


class GRNDecoderSuite(nn.Module):
    """Compatibility suite with feature-aware regulatory decoders by default."""

    VALID_MODES = {"edge_feature_aware", "projected_dot"}

    def __init__(
        self,
        hidden_dim: int,
        *,
        mode: str = "edge_feature_aware",
        use_edge_features: bool = True,
        tf_peak_edge_feature_dim: int = 3,
        peak_gene_edge_feature_dim: int = 2,
        tf_gene_edge_feature_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}.")
        self.mode = mode
        self.use_edge_features = bool(use_edge_features)
        self.decoders = nn.ModuleDict(
            {
                name: ProjectedDotDecoder(hidden_dim)
                for name in ("cg", "cp", "tp", "pg", "tg")
            }
        )
        dimensions = {
            "tp": tf_peak_edge_feature_dim,
            "pg": peak_gene_edge_feature_dim,
            "tg": tf_gene_edge_feature_dim,
        }
        if not self.use_edge_features:
            dimensions = {name: 0 for name in dimensions}
        self.edge_feature_dims = dimensions
        self.tf_peak = TFPeakDecoder(
            hidden_dim,
            edge_feature_dim=dimensions["tp"],
            dropout=dropout,
        )
        self.peak_gene = PeakGeneDecoder(
            hidden_dim,
            edge_feature_dim=dimensions["pg"],
            dropout=dropout,
        )
        self.tf_gene = TFGeneDecoder(
            hidden_dim,
            edge_feature_dim=dimensions["tg"],
            dropout=dropout,
        )

    @staticmethod
    def _validate_tf_sources(
        edge_index: torch.Tensor,
        is_tf: torch.Tensor | None,
        n_genes: int,
        relation: str,
    ) -> None:
        if is_tf is None:
            return
        if is_tf.ndim != 1 or len(is_tf) != n_genes or is_tf.dtype != torch.bool:
            raise ValueError("is_tf must align with gene embeddings.")
        source_ids = edge_index[0].to(is_tf.device)
        if edge_index.shape[1] and not bool(is_tf[source_ids].all()):
            raise ValueError(f"{relation} edges must originate from TF gene nodes.")

    def score(
        self,
        relation: str,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        *,
        is_tf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits using the configured decoder path."""

        if relation not in self.decoders:
            raise KeyError(f"Unknown decoder relation: {relation}")
        if self.mode == "projected_dot" or relation in {"cg", "cp"}:
            return self.decoders[relation](
                src_emb[edge_index[0]],
                dst_emb[edge_index[1]],
            )
        if not self.use_edge_features:
            edge_features = None
        if relation == "tp":
            self._validate_tf_sources(
                edge_index, is_tf, len(src_emb), "TF-peak"
            )
            return self.tf_peak.logits(
                src_emb, dst_emb, edge_index, edge_features
            )
        if relation == "pg":
            return self.peak_gene.logits(
                src_emb, dst_emb, edge_index, edge_features
            )
        self._validate_tf_sources(edge_index, is_tf, len(src_emb), "TF-gene")
        return self.tf_gene.logits(
            src_emb, dst_emb, edge_index, edge_features
        )

    def score_pairs(
        self,
        relation: str,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        src_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        *,
        is_tf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits for explicit source/destination id arrays."""

        edge_index = torch.stack([src_ids, dst_ids], dim=0)
        return self.score(
            relation,
            src_emb,
            dst_emb,
            edge_index,
            edge_features,
            is_tf=is_tf,
        )

    def probabilities(
        self,
        relation: str,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        *,
        is_tf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return continuous scores in ``[0, 1]``."""

        return torch.sigmoid(
            self.score(
                relation,
                src_emb,
                dst_emb,
                edge_index,
                edge_features,
                is_tf=is_tf,
            )
        )


@dataclass(frozen=True)
class TFGeneAggregation:
    """Sparse TF→gene scores and their retained mediator paths."""

    edge_index: torch.Tensor
    score: torch.Tensor
    n_mediator_peaks: torch.Tensor
    path_tf_index: torch.Tensor
    path_peak_index: torch.Tensor
    path_gene_index: torch.Tensor
    path_score: torch.Tensor


class TFPeakGeneAggregator(nn.Module):
    """Compose TF→peak and peak→gene scores without a dense Cartesian tensor."""

    def __init__(
        self,
        mode: str = DEFAULT_AGGREGATION_MODE,
        eps: float = 1e-8,
        top_k: int = 10,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.mode = canonical_aggregation_mode(mode)
        self.eps = float(eps)
        self.top_k = int(top_k)

    @staticmethod
    def _validate_relation(
        edge_index: torch.Tensor,
        score: torch.Tensor,
        name: str,
    ) -> None:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"{name}_edge_index must have shape [2, n_edges].")
        if edge_index.dtype != torch.long:
            raise ValueError(f"{name}_edge_index must use torch.long indices.")
        if score.ndim != 1 or len(score) != edge_index.shape[1]:
            raise ValueError(f"{name}_score must align with {name}_edge_index.")
        if not bool(torch.isfinite(score).all()) or not bool(
            ((score >= 0) & (score <= 1)).all()
        ):
            raise ValueError(f"{name}_score must contain finite values in [0, 1].")

    def forward(
        self,
        tf_peak_edge_index: torch.Tensor,
        tf_peak_score: torch.Tensor,
        peak_gene_edge_index: torch.Tensor,
        peak_gene_score: torch.Tensor,
    ) -> TFGeneAggregation:
        self._validate_relation(tf_peak_edge_index, tf_peak_score, "tf_peak")
        self._validate_relation(peak_gene_edge_index, peak_gene_score, "peak_gene")
        devices = {
            tf_peak_edge_index.device,
            peak_gene_edge_index.device,
            tf_peak_score.device,
            peak_gene_score.device,
        }
        if len(devices) != 1:
            raise ValueError("All edge indices and scores must share a device.")

        peak_to_pg: dict[int, list[int]] = {}
        for pg_index, peak in enumerate(peak_gene_edge_index[0].tolist()):
            peak_to_pg.setdefault(int(peak), []).append(pg_index)
        path_tf: list[int] = []
        path_peak: list[int] = []
        path_gene: list[int] = []
        path_scores: list[torch.Tensor] = []
        for tp_index, (tf, peak) in enumerate(tf_peak_edge_index.t().tolist()):
            for pg_index in peak_to_pg.get(int(peak), []):
                path_tf.append(int(tf))
                path_peak.append(int(peak))
                path_gene.append(int(peak_gene_edge_index[1, pg_index]))
                path_scores.append(
                    path_score_tensor(
                        tf_peak_score[tp_index],
                        peak_gene_score[pg_index],
                    )
                )

        device = tf_peak_edge_index.device
        score_device = tf_peak_score.device
        if not path_scores:
            empty_index = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            empty_score = torch.empty(
                0, dtype=tf_peak_score.dtype, device=score_device
            )
            return TFGeneAggregation(
                edge_index=empty_index,
                score=empty_score,
                n_mediator_peaks=empty_long,
                path_tf_index=empty_long,
                path_peak_index=empty_long,
                path_gene_index=empty_long,
                path_score=empty_score,
            )

        path_tf_tensor = torch.tensor(path_tf, dtype=torch.long, device=device)
        path_peak_tensor = torch.tensor(path_peak, dtype=torch.long, device=device)
        path_gene_tensor = torch.tensor(path_gene, dtype=torch.long, device=device)
        path_score = torch.stack(path_scores)
        pairs = torch.stack([path_tf_tensor, path_gene_tensor], dim=1)
        unique_pairs, inverse = torch.unique(
            pairs,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        aggregate_scores: list[torch.Tensor] = []
        mediator_counts: list[int] = []
        for pair_index in range(len(unique_pairs)):
            selected = inverse == pair_index
            values = path_score[selected]
            aggregate_scores.append(
                aggregate_tensor(
                    values,
                    self.mode,
                    dim=0,
                    top_k=self.top_k,
                )
            )
            mediator_counts.append(
                int(torch.unique(path_peak_tensor[selected]).numel())
            )
        return TFGeneAggregation(
            edge_index=unique_pairs.t().contiguous(),
            score=torch.stack(aggregate_scores),
            n_mediator_peaks=torch.tensor(
                mediator_counts, dtype=torch.long, device=device
            ),
            path_tf_index=path_tf_tensor,
            path_peak_index=path_peak_tensor,
            path_gene_index=path_gene_tensor,
            path_score=path_score,
        )
