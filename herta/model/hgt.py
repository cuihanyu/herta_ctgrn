"""Relation-aware HGT encoder with an explicit three-node output contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import HGTConv


@dataclass(frozen=True)
class HGTEncoderOutput:
    """Contextual embeddings for the redesigned HERTA node schema."""

    z_cell: torch.Tensor
    z_gene: torch.Tensor
    z_peak: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "cell": self.z_cell,
            "gene": self.z_gene,
            "peak": self.z_peak,
        }


class HGTEncoder(nn.Module):
    """Stack PyG ``HGTConv`` layers with residual, normalization, and dropout."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("num_heads must be positive and divide hidden_dim.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        node_types, edge_types = metadata
        if not node_types:
            raise ValueError("HGT metadata must contain at least one node type.")
        if len(set(node_types)) != len(node_types):
            raise ValueError("HGT metadata contains duplicate node types.")
        self.hidden_dim = int(hidden_dim)
        self.node_types = tuple(node_types)
        self.edge_types = tuple(edge_types)
        self.layers = nn.ModuleList(
            [
                HGTConv(
                    hidden_dim,
                    hidden_dim,
                    metadata,
                    heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(hidden_dim)
                        for node_type in node_types
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def _validate_inputs(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> None:
        missing = set(self.node_types).difference(x_dict)
        if missing:
            raise ValueError(f"Missing HGT node features: {sorted(missing)}")
        for node_type in self.node_types:
            values = x_dict[node_type]
            if values.ndim != 2 or values.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"{node_type} HGT input must have shape "
                    f"[n_nodes, {self.hidden_dim}], got {tuple(values.shape)}."
                )
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"{node_type} HGT input contains non-finite values.")
        unknown_relations = set(edge_index_dict).difference(self.edge_types)
        if unknown_relations:
            raise ValueError(
                f"edge_index_dict contains relations absent from metadata: "
                f"{sorted(unknown_relations)}"
            )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Encode node embeddings contextually."""

        self._validate_inputs(x_dict, edge_index_dict)
        out = {node_type: x_dict[node_type] for node_type in self.node_types}
        for conv, norms in zip(self.layers, self.norms):
            updated = conv(out, edge_index_dict)
            next_out: dict[str, torch.Tensor] = {}
            for node_type in self.node_types:
                message = updated.get(node_type)
                if message is None:
                    message = torch.zeros_like(out[node_type])
                next_out[node_type] = norms[node_type](
                    out[node_type] + self.dropout(message)
                )
            out = next_out
        return out

    def encode_three_node(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> HGTEncoderOutput:
        """Return explicit ``z_cell``, ``z_gene``, and ``z_peak`` embeddings."""

        required = {"cell", "gene", "peak"}
        if set(self.node_types) != required:
            raise ValueError(
                "encode_three_node requires metadata with exactly cell, gene, and peak nodes."
            )
        encoded = self.forward(x_dict, edge_index_dict)
        return HGTEncoderOutput(
            z_cell=encoded["cell"],
            z_gene=encoded["gene"],
            z_peak=encoded["peak"],
        )
