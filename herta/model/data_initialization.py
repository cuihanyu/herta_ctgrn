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
        cell_modality_dims: tuple[int, int] | None = None,
        atac_gate_init: float = 0.25,
    ) -> None:
        super().__init__()
        if cell_modality_dims is not None:
            if sum(cell_modality_dims) != input_dims.get("cell"):
                raise ValueError("cell_modality_dims must sum to the cell feature dimension.")
            if not 0 < atac_gate_init < 1:
                raise ValueError("atac_gate_init must be between 0 and 1.")
        self.cell_modality_dims = cell_modality_dims
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(dim, hidden_dim)
                for name, dim in input_dims.items()
                if name != "cell" or cell_modality_dims is None
            }
        )
        if cell_modality_dims is not None:
            self.rna_projection = nn.Linear(cell_modality_dims[0], hidden_dim)
            self.atac_projection = nn.Linear(cell_modality_dims[1], hidden_dim)
            self.rna_norm = nn.LayerNorm(hidden_dim)
            self.atac_norm = nn.LayerNorm(hidden_dim)
            self.atac_gate_logit = nn.Parameter(torch.logit(torch.tensor(float(atac_gate_init))))
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
            if name == "cell" and self.cell_modality_dims is not None:
                rna_dim, atac_dim = self.cell_modality_dims
                rna = self.rna_norm(self.rna_projection(values[:, :rna_dim]))
                atac = self.atac_norm(self.atac_projection(values[:, rna_dim : rna_dim + atac_dim]))
                gate = torch.sigmoid(self.atac_gate_logit)
                projected = (rna + gate * atac) / (1.0 + gate)
            else:
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

    @property
    def atac_gate(self) -> torch.Tensor | None:
        """Return the learned ATAC contribution for gated cell fusion."""

        return torch.sigmoid(self.atac_gate_logit) if self.cell_modality_dims is not None else None
