"""SIMBA-style random trainable embeddings."""

from __future__ import annotations

import torch
from torch import nn


class RandomNodeEmbeddings(nn.Module):
    """Trainable lookup tables for each node type."""

    def __init__(self, num_nodes: dict[str, int], hidden_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict()
        for node_type, count in num_nodes.items():
            emb = nn.Embedding(count, hidden_dim)
            nn.init.normal_(emb.weight, mean=0.0, std=hidden_dim**-0.5)
            with torch.no_grad():
                emb.weight.div_(emb.weight.norm(dim=1, keepdim=True) + eps)
            self.embeddings[node_type] = emb

    def forward(self) -> dict[str, torch.Tensor]:
        """Return all node embeddings."""

        return {node_type: emb.weight for node_type, emb in self.embeddings.items()}
