"""Data-driven node initialization for two-stage HERTA."""

from __future__ import annotations

import torch
from torch import nn


class DataDrivenNodeInitializer(nn.Module):
    """Project modality factors into a shared hidden space."""

    def __init__(
        self,
        input_dims: dict[str, int],
        hidden_dim: int,
        num_nodes: dict[str, int],
        use_id_residual: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if set(input_dims) != {"cell", "gene", "peak"}:
            raise ValueError("Initializer requires cell, gene, and peak input dimensions.")
        if any(dimension < 1 for dimension in input_dims.values()):
            raise ValueError("Every node type must have a positive input dimension.")
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(dim, hidden_dim)
                for name, dim in input_dims.items()
            }
        )
        self.norms = nn.ModuleDict({name: nn.LayerNorm(hidden_dim) for name in input_dims})
        self.id_embeddings = nn.ModuleDict()
        if use_id_residual:
            for name, count in num_nodes.items():
                embedding = nn.Embedding(count, hidden_dim)
                nn.init.zeros_(embedding.weight)
                self.id_embeddings[name] = embedding

    def forward(
        self,
        features: dict[str, torch.Tensor],
        node_ids: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Project node features, optionally selecting global ID residual rows."""

        output: dict[str, torch.Tensor] = {}
        for name, values in features.items():
            if values.ndim != 2 or not torch.isfinite(values).all():
                raise ValueError(f"Features for {name} must be a finite rank-2 tensor.")
            projected = self.projections[name](values)
            if name in self.id_embeddings:
                if node_ids is None:
                    identity = self.id_embeddings[name].weight
                else:
                    if name not in node_ids:
                        raise ValueError(f"Missing global node IDs for {name}.")
                    ids = torch.as_tensor(
                        node_ids[name], device=values.device, dtype=torch.long
                    )
                    if ids.ndim != 1 or ids.numel() != values.shape[0]:
                        raise ValueError(
                            f"Global node IDs for {name} must align with feature rows."
                        )
                    identity = self.id_embeddings[name](ids)
                if identity.shape != projected.shape:
                    raise ValueError(
                        f"ID residual rows for {name} do not match projected features."
                    )
                projected = projected + identity
            output[name] = self.norms[name](projected)
        return output
