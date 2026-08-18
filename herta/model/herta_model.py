"""HERTA model wrapper."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import HeteroData

from herta.data.graph_builder import FORWARD_RELATIONS
from herta.model.decoders import RelationDecoders
from herta.model.hgt import HGTEncoder
from herta.model.init_embeddings import RandomNodeEmbeddings


class HertaModel(nn.Module):
    """Random embeddings + minimal HGT + relation decoders."""

    def __init__(self, data: HeteroData, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        num_nodes = {node_type: int(data[node_type].num_nodes) for node_type in data.node_types}
        self.embeddings = RandomNodeEmbeddings(num_nodes, hidden_dim)
        self.encoder = HGTEncoder(hidden_dim, num_layers, num_heads, data.metadata(), dropout=dropout)
        self.decoders = RelationDecoders(hidden_dim, list(FORWARD_RELATIONS.keys()))

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Return contextual node embeddings."""

        return self.encoder(self.embeddings(), data.edge_index_dict)
