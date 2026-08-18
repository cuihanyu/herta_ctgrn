"""Shared Stage-2 regulatory model initialized from Stage 1."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch_geometric.data import HeteroData

from herta.model.decoders import (
    GRNDecoderSuite,
    TFGeneAggregation,
    TFPeakGeneAggregator,
)
from herta.model.hgt import HGTEncoder
from herta.model.regulatory_scoring import DEFAULT_AGGREGATION_MODE
from herta.model.state_model import StateModel


def _load_shape_compatible(module: nn.Module, source: nn.Module) -> None:
    target_state = module.state_dict()
    compatible = {
        name: value
        for name, value in source.state_dict().items()
        if name in target_state and target_state[name].shape == value.shape
    }
    module.load_state_dict(compatible, strict=False)


@dataclass(frozen=True)
class GRNModelOutput:
    """Explicit Stage-2 embeddings and regulatory predictions."""

    z_cell: torch.Tensor
    z_gene: torch.Tensor
    z_peak: torch.Tensor
    tf_peak_logits: torch.Tensor
    tf_peak_scores: torch.Tensor
    peak_gene_logits: torch.Tensor
    peak_gene_scores: torch.Tensor
    tf_gene_logits: torch.Tensor | None = None
    tf_gene_scores: torch.Tensor | None = None
    tf_gene_edge_index: torch.Tensor | None = None
    tf_gene_aggregation: TFGeneAggregation | None = None

    @property
    def embeddings(self) -> dict[str, torch.Tensor]:
        return {
            "cell": self.z_cell,
            "gene": self.z_gene,
            "peak": self.z_peak,
        }


class GRNModel(nn.Module):
    """One shared regulatory encoder and TP/PG decoders for all clusters."""

    def __init__(
        self,
        data: HeteroData,
        state_model: StateModel,
        num_layers: int | None = None,
        num_heads: int | None = None,
        dropout: float | None = None,
        state_embeddings: dict[str, torch.Tensor] | None = None,
        features: dict[str, torch.Tensor] | None = None,
        decoder_mode: str = "edge_feature_aware",
        use_edge_features: bool = True,
        tf_peak_edge_feature_dim: int = 3,
        peak_gene_edge_feature_dim: int = 2,
        tf_gene_edge_feature_dim: int = 1,
        aggregator_mode: str = DEFAULT_AGGREGATION_MODE,
        aggregator_top_k: int = 10,
    ) -> None:
        super().__init__()
        cfg = state_model.model_config
        hidden_dim = int(cfg["hidden_dim"])
        graph_is_tf = getattr(data["gene"], "is_tf", None)
        if graph_is_tf is None:
            graph_is_tf = torch.zeros(
                int(data["gene"].num_nodes), dtype=torch.bool
            )
        self.register_buffer("is_tf", graph_is_tf.detach().clone().bool())
        self.initializer = copy.deepcopy(state_model.initializer)
        if state_embeddings is not None and features is None:
            raise ValueError("features are required when state_embeddings are provided.")
        with torch.no_grad():
            projected = self.initializer(features) if features is not None else {}
        for node_type in data.node_types:
            if state_embeddings is None:
                offset = torch.zeros((int(data[node_type].num_nodes), hidden_dim), dtype=torch.float32)
            else:
                offset = state_embeddings[node_type].detach().cpu() - projected[node_type].detach().cpu()
            self.register_buffer(f"stage1_offset_{node_type}", offset)
        self.encoder = HGTEncoder(
            hidden_dim,
            int(num_layers if num_layers is not None else cfg["num_layers"]),
            int(num_heads if num_heads is not None else cfg["num_heads"]),
            data.metadata(),
            float(dropout if dropout is not None else cfg["dropout"]),
        )
        _load_shape_compatible(self.encoder, state_model.encoder)
        decoder_dropout = float(
            dropout if dropout is not None else cfg["dropout"]
        )
        self.decoders = GRNDecoderSuite(
            hidden_dim,
            mode=decoder_mode,
            use_edge_features=use_edge_features,
            tf_peak_edge_feature_dim=tf_peak_edge_feature_dim,
            peak_gene_edge_feature_dim=peak_gene_edge_feature_dim,
            tf_gene_edge_feature_dim=tf_gene_edge_feature_dim,
            dropout=decoder_dropout,
        )
        self.tf_peak_gene_aggregator = TFPeakGeneAggregator(
            aggregator_mode,
            top_k=aggregator_top_k,
        )
        for relation in ("cg", "cp"):
            _load_shape_compatible(self.decoders.decoders[relation], state_model.decoders.decoders[relation])
        self.model_config = {
            "hidden_dim": hidden_dim,
            "num_layers": len(self.encoder.layers),
            "num_heads": int(num_heads if num_heads is not None else cfg["num_heads"]),
            "dropout": decoder_dropout,
            "decoder_mode": decoder_mode,
            "use_edge_features": bool(use_edge_features),
            "tf_peak_edge_feature_dim": int(tf_peak_edge_feature_dim),
            "peak_gene_edge_feature_dim": int(peak_gene_edge_feature_dim),
            "tf_gene_edge_feature_dim": int(tf_gene_edge_feature_dim),
            "aggregator_mode": self.tf_peak_gene_aggregator.mode,
            "aggregator_top_k": int(aggregator_top_k),
        }

    def initialize(
        self,
        features: dict[str, torch.Tensor],
        global_node_ids: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return node states initialized exactly at the Stage-1 embeddings."""

        projected = self.initializer(features)
        return {
            node_type: values
            + (
                getattr(self, f"stage1_offset_{node_type}")
                if global_node_ids is None
                else getattr(self, f"stage1_offset_{node_type}")[global_node_ids[node_type]]
            ).to(values.device)
            for node_type, values in projected.items()
        }

    def encode(
        self,
        data: HeteroData,
        features: dict[str, torch.Tensor],
        global_node_ids: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a full graph or a sampler batch with global Stage-1 offsets."""

        return self.encoder(
            self.initialize(features, global_node_ids=global_node_ids),
            data.edge_index_dict,
        )

    @staticmethod
    def _empty_edge_index(embedding: torch.Tensor) -> torch.Tensor:
        return torch.empty((2, 0), dtype=torch.long, device=embedding.device)

    @staticmethod
    def _query_value(
        query_tensors: Mapping[str, object] | None,
        key: str,
        attribute: str,
    ) -> torch.Tensor | None:
        if query_tensors is None or key not in query_tensors:
            return None
        value = getattr(query_tensors[key], attribute, None)
        return value if isinstance(value, torch.Tensor) else None

    def decode(
        self,
        embeddings: Mapping[str, torch.Tensor],
        *,
        query_tensors: Mapping[str, object] | None = None,
        tf_peak_edge_index: torch.Tensor | None = None,
        peak_gene_edge_index: torch.Tensor | None = None,
        tf_gene_edge_index: torch.Tensor | None = None,
        tf_peak_edge_features: torch.Tensor | None = None,
        peak_gene_edge_features: torch.Tensor | None = None,
        tf_gene_edge_features: torch.Tensor | None = None,
        is_tf: torch.Tensor | None = None,
        aggregate_tf_gene: bool = True,
    ) -> GRNModelOutput:
        """Decode candidate regulatory edges from precomputed embeddings."""

        z_cell = embeddings["cell"]
        z_gene = embeddings["gene"]
        z_peak = embeddings["peak"]
        tf_peak_edge_index = (
            tf_peak_edge_index
            if tf_peak_edge_index is not None
            else self._query_value(query_tensors, "tp", "edge_index")
        )
        peak_gene_edge_index = (
            peak_gene_edge_index
            if peak_gene_edge_index is not None
            else self._query_value(query_tensors, "pg", "edge_index")
        )
        tf_gene_edge_index = (
            tf_gene_edge_index
            if tf_gene_edge_index is not None
            else self._query_value(query_tensors, "tg", "edge_index")
        )
        tf_peak_edge_features = (
            tf_peak_edge_features
            if tf_peak_edge_features is not None
            else self._query_value(query_tensors, "tp", "edge_features")
        )
        peak_gene_edge_features = (
            peak_gene_edge_features
            if peak_gene_edge_features is not None
            else self._query_value(query_tensors, "pg", "edge_features")
        )
        tf_gene_edge_features = (
            tf_gene_edge_features
            if tf_gene_edge_features is not None
            else self._query_value(query_tensors, "tg", "edge_features")
        )
        if tf_peak_edge_index is None:
            tf_peak_edge_index = self._empty_edge_index(z_gene)
        if peak_gene_edge_index is None:
            peak_gene_edge_index = self._empty_edge_index(z_peak)

        tf_peak_logits = self.decoders.score(
            "tp",
            z_gene,
            z_peak,
            tf_peak_edge_index,
            tf_peak_edge_features,
            is_tf=is_tf,
        )
        peak_gene_logits = self.decoders.score(
            "pg",
            z_peak,
            z_gene,
            peak_gene_edge_index,
            peak_gene_edge_features,
        )
        tf_peak_scores = torch.sigmoid(tf_peak_logits)
        peak_gene_scores = torch.sigmoid(peak_gene_logits)

        aggregation = None
        tf_gene_logits = None
        tf_gene_scores = None
        output_tf_gene_edge_index = tf_gene_edge_index
        if tf_gene_edge_index is not None:
            tf_gene_logits = self.decoders.score(
                "tg",
                z_gene,
                z_gene,
                tf_gene_edge_index,
                tf_gene_edge_features,
                is_tf=is_tf,
            )
            tf_gene_scores = torch.sigmoid(tf_gene_logits)
        elif aggregate_tf_gene:
            aggregation = self.tf_peak_gene_aggregator(
                tf_peak_edge_index,
                tf_peak_scores,
                peak_gene_edge_index,
                peak_gene_scores,
            )
            output_tf_gene_edge_index = aggregation.edge_index
            tf_gene_scores = aggregation.score
            tf_gene_logits = torch.logit(
                tf_gene_scores.clamp(min=1e-6, max=1.0 - 1e-6)
            )
        return GRNModelOutput(
            z_cell=z_cell,
            z_gene=z_gene,
            z_peak=z_peak,
            tf_peak_logits=tf_peak_logits,
            tf_peak_scores=tf_peak_scores,
            peak_gene_logits=peak_gene_logits,
            peak_gene_scores=peak_gene_scores,
            tf_gene_logits=tf_gene_logits,
            tf_gene_scores=tf_gene_scores,
            tf_gene_edge_index=output_tf_gene_edge_index,
            tf_gene_aggregation=aggregation,
        )

    def forward(
        self,
        data: HeteroData,
        features: dict[str, torch.Tensor],
        *,
        query_tensors: Mapping[str, object] | None = None,
        tf_peak_edge_index: torch.Tensor | None = None,
        peak_gene_edge_index: torch.Tensor | None = None,
        tf_gene_edge_index: torch.Tensor | None = None,
        tf_peak_edge_features: torch.Tensor | None = None,
        peak_gene_edge_features: torch.Tensor | None = None,
        tf_gene_edge_features: torch.Tensor | None = None,
        aggregate_tf_gene: bool = True,
    ) -> GRNModelOutput:
        """Encode the graph and return explicit Stage-2 predictions."""

        embeddings = self.encode(data, features)
        return self.decode(
            embeddings,
            query_tensors=query_tensors,
            tf_peak_edge_index=tf_peak_edge_index,
            peak_gene_edge_index=peak_gene_edge_index,
            tf_gene_edge_index=tf_gene_edge_index,
            tf_peak_edge_features=tf_peak_edge_features,
            peak_gene_edge_features=peak_gene_edge_features,
            tf_gene_edge_features=tf_gene_edge_features,
            is_tf=self.is_tf,
            aggregate_tf_gene=aggregate_tf_gene,
        )
