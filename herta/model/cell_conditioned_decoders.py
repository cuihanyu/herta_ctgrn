"""Minimal cell-conditioned Stage-2 regulatory decoders."""

from __future__ import annotations

import torch
from torch import nn


class _CellConditionedDecoder(nn.Module):
    """Score triples by concatenating one cell and two endpoint embeddings."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_cell: torch.Tensor,
        z_source: torch.Tensor,
        z_target: torch.Tensor,
    ) -> torch.Tensor:
        expected = z_cell.shape
        if (
            z_cell.ndim != 2
            or z_source.shape != expected
            or z_target.shape != expected
            or expected[1] != self.hidden_dim
        ):
            raise ValueError(
                "Cell, source, and target embeddings must share shape "
                f"[batch, {self.hidden_dim}]."
            )
        if not bool(
            torch.isfinite(z_cell).all()
            and torch.isfinite(z_source).all()
            and torch.isfinite(z_target).all()
        ):
            raise ValueError("Decoder embeddings must be finite.")
        return self.net(torch.cat([z_cell, z_source, z_target], dim=-1)).squeeze(-1)


class CellConditionedTFPeakDecoder(_CellConditionedDecoder):
    """Return logits for P(TF -> peak | cell)."""


class CellConditionedPeakGeneDecoder(_CellConditionedDecoder):
    """Return logits for P(peak -> gene | cell)."""


class Stage2CellConditionedModel(nn.Module):
    """Two shared decoders operating directly on frozen Stage-1 embeddings."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.tf_peak_decoder = CellConditionedTFPeakDecoder(hidden_dim, dropout)
        self.peak_gene_decoder = CellConditionedPeakGeneDecoder(hidden_dim, dropout)

    def score_tf_peak(
        self,
        z_cell: torch.Tensor,
        z_tf: torch.Tensor,
        z_peak: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.tf_peak_decoder(z_cell, z_tf, z_peak)
        return logits, torch.sigmoid(logits)

    def score_peak_gene(
        self,
        z_cell: torch.Tensor,
        z_peak: torch.Tensor,
        z_gene: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.peak_gene_decoder(z_cell, z_peak, z_gene)
        return logits, torch.sigmoid(logits)

    @property
    def model_config(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "stage2_mode": "cell_conditioned_lite",
            "stage2_hgt": False,
        }
