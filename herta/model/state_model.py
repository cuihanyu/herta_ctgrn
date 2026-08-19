"""Stage-1 cell-state representation model."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import HeteroData

from herta.model.data_initialization import DataDrivenNodeInitializer
from herta.model.decoders import ProjectedRelationDecoders
from herta.model.hgt import HGTEncoder


class StateModel(nn.Module):
    """Data initialization, HGT, and observed-relation decoders."""

    def __init__(
        self,
        data: HeteroData,
        features: dict[str, torch.Tensor],
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        use_id_residual: bool = False,
    ) -> None:
        super().__init__()
        input_dims = {name: int(value.shape[1]) for name, value in features.items()}
        num_nodes = {name: int(data[name].num_nodes) for name in data.node_types}
        for name, values in features.items():
            if values.ndim != 2 or values.shape[0] != num_nodes[name]:
                raise ValueError(f"{name} features must align with graph nodes.")
            if not torch.isfinite(values).all():
                raise ValueError(f"{name} features contain NaN or infinite values.")
        self.initializer = DataDrivenNodeInitializer(
            input_dims,
            hidden_dim,
            num_nodes,
            use_id_residual,
        )
        self.encoder = HGTEncoder(hidden_dim, num_layers, num_heads, data.metadata(), dropout)
        self.decoders = ProjectedRelationDecoders(hidden_dim, ["cg", "cp"])
        self.model_config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "use_id_residual": use_id_residual,
        }
        self.initialization_audit = {
            "cell_concatenated_shape": list(features["cell"].shape),
            "gene_feature_shape": list(features["gene"].shape),
            "peak_feature_shape": list(features["peak"].shape),
            "projected_hidden_dimension": int(hidden_dim),
            "all_features_finite": True,
            "fusion": "rna_pca_concat_atac_lsi",
        }

    def encode(
        self,
        data: HeteroData,
        features: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] | None = None,
        global_node_ids: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.encoder(
            self.initializer(features, node_ids=global_node_ids),
            edge_index_dict or data.edge_index_dict,
        )
