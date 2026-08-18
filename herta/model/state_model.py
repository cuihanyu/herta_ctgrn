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
        atac_feature_dim: int = 0,
        modality_fusion: str = "joint",
        atac_gate_init: float = 0.25,
    ) -> None:
        super().__init__()
        if modality_fusion not in {"joint", "gated"}:
            raise ValueError("modality_fusion must be 'joint' or 'gated'.")
        input_dims = {name: int(value.shape[1]) for name, value in features.items()}
        num_nodes = {name: int(data[name].num_nodes) for name in data.node_types}
        cell_modality_dims = None
        if modality_fusion == "gated":
            rna_feature_dim = input_dims["cell"] - atac_feature_dim
            if rna_feature_dim < 1 or atac_feature_dim < 1:
                raise ValueError("Gated fusion requires positive RNA and ATAC cell feature dimensions.")
            cell_modality_dims = (rna_feature_dim, atac_feature_dim)
        self.initializer = DataDrivenNodeInitializer(
            input_dims,
            hidden_dim,
            num_nodes,
            use_id_residual,
            cell_modality_dims=cell_modality_dims,
            atac_gate_init=atac_gate_init,
        )
        self.encoder = HGTEncoder(hidden_dim, num_layers, num_heads, data.metadata(), dropout)
        self.decoders = ProjectedRelationDecoders(hidden_dim, ["cg", "cp"])
        self.model_config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "use_id_residual": use_id_residual,
            "atac_feature_dim": atac_feature_dim,
            "modality_fusion": modality_fusion,
            "atac_gate_init": atac_gate_init,
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
