"""Training for the minimal cell-conditioned HERTA Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse

from herta.data.regulatory_graph import RegulatoryGraphBuildResult
from herta.model.cell_conditioned_decoders import Stage2CellConditionedModel
from herta.utils.seed import set_seed


def _frozen_embeddings(
    embeddings: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    required = {"cell", "gene", "peak"}
    if set(embeddings) != required:
        raise ValueError(f"state_embeddings must contain exactly {sorted(required)}.")
    values = {name: value.detach().cpu().float().contiguous() for name, value in embeddings.items()}
    dimensions = {value.shape[1] for value in values.values() if value.ndim == 2}
    if len(dimensions) != 1 or any(value.ndim != 2 for value in values.values()):
        raise ValueError("Stage-1 embeddings must be 2D and share one hidden dimension.")
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError("Stage-1 embeddings must be finite.")
    return values


def _row_max_normalize(matrix: object) -> sparse.csr_matrix:
    value = sparse.csr_matrix(matrix, dtype=np.float32)
    if value.nnz and (not np.isfinite(value.data).all() or np.any(value.data < 0)):
        raise ValueError("Activity matrices must be finite and non-negative.")
    maximum = np.asarray(value.max(axis=1).toarray()).ravel()
    scale = np.divide(1.0, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    return (sparse.diags(scale) @ value).tocsr()


@dataclass(frozen=True)
class CoupledPathCandidates:
    peaks: torch.Tensor
    tf_flat: torch.Tensor
    tf_offsets: torch.Tensor
    tf_counts: torch.Tensor
    gene_flat: torch.Tensor
    gene_offsets: torch.Tensor
    gene_counts: torch.Tensor


def _train_rows(table: pd.DataFrame) -> pd.DataFrame:
    if "split" not in table:
        return table
    return table.loc[table["split"].astype(str).eq("train")]


def _coupled_path_candidates(
    build: RegulatoryGraphBuildResult,
) -> CoupledPathCandidates:
    gene_index = {name: index for index, name in enumerate(build.state.names["gene"])}
    peak_index = {name: index for index, name in enumerate(build.state.names["peak"])}
    tp = _train_rows(build.candidates["tp"]).drop_duplicates(["tf", "peak"])
    pg = _train_rows(build.candidates["pg"]).drop_duplicates(["peak", "gene"])
    if tp.empty or pg.empty:
        raise ValueError("Stage 2 Lite requires non-empty train-split TP and PG candidates.")
    shared = sorted(set(tp["peak"].astype(str)) & set(pg["peak"].astype(str)))
    if not shared:
        raise ValueError("Train-split TP and PG candidates share no peaks.")

    tf_groups = tp.groupby(tp["peak"].astype(str), sort=False)["tf"].agg(list)
    gene_groups = pg.groupby(pg["peak"].astype(str), sort=False)["gene"].agg(list)

    def pack(groups: pd.Series) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = [
            torch.as_tensor(
                [gene_index[str(name)] for name in groups.loc[peak]], dtype=torch.long
            )
            for peak in shared
        ]
        counts = torch.as_tensor([len(value) for value in values], dtype=torch.long)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
        return torch.cat(values), offsets, counts

    tf_flat, tf_offsets, tf_counts = pack(tf_groups)
    gene_flat, gene_offsets, gene_counts = pack(gene_groups)
    return CoupledPathCandidates(
        peaks=torch.as_tensor([peak_index[peak] for peak in shared], dtype=torch.long),
        tf_flat=tf_flat,
        tf_offsets=tf_offsets,
        tf_counts=tf_counts,
        gene_flat=gene_flat,
        gene_offsets=gene_offsets,
        gene_counts=gene_counts,
    )


def _sample_coupled_paths(
    n_cells: int,
    candidates: CoupledPathCandidates,
    batch_size_cells: int,
    paths_per_cell: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cell_ids = torch.randperm(n_cells, generator=generator)[: min(batch_size_cells, n_cells)]
    cell_ids = cell_ids.repeat_interleave(paths_per_cell)
    group_ids = torch.randint(len(candidates.peaks), (len(cell_ids),), generator=generator)

    def sample_neighbor(
        flat: torch.Tensor, offsets: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        local = torch.floor(
            torch.rand(len(group_ids), generator=generator) * counts[group_ids]
        ).long()
        return flat[offsets[group_ids] + local]

    return (
        cell_ids,
        sample_neighbor(candidates.tf_flat, candidates.tf_offsets, candidates.tf_counts),
        candidates.peaks[group_ids],
        sample_neighbor(
            candidates.gene_flat, candidates.gene_offsets, candidates.gene_counts
        ),
    )


def _activity_values(
    matrix: sparse.csr_matrix,
    rows: torch.Tensor,
    columns: torch.Tensor,
) -> torch.Tensor:
    row_ids = rows.detach().cpu().numpy().astype(np.int64, copy=True)
    column_ids = columns.detach().cpu().numpy().astype(np.int64, copy=True)
    selected = matrix[row_ids, column_ids]
    values = np.asarray(selected).reshape(-1).astype(np.float32, copy=False)
    return torch.from_numpy(values)


def train_stage2_lite(
    build: RegulatoryGraphBuildResult,
    state_embeddings: dict[str, torch.Tensor],
    config: dict,
    output_dir: str | Path,
    model: Stage2CellConditionedModel | None = None,
) -> tuple[Stage2CellConditionedModel, pd.DataFrame]:
    """Fit two cell-conditioned decoders while Stage-1 tensors stay frozen."""

    output_dir = Path(output_dir)
    seed = int(config.get("seed", 1))
    set_seed(seed)
    train_cfg = config.get("stage2_lite", config.get("regulatory_training_lite", {}))
    embeddings = _frozen_embeddings(state_embeddings)
    snapshot = {name: value.clone() for name, value in embeddings.items()}
    hidden_dim = int(embeddings["cell"].shape[1])
    model = model or Stage2CellConditionedModel(
        hidden_dim,
        dropout=float(train_cfg.get("dropout", 0.2)),
    )
    if model.hidden_dim != hidden_dim:
        raise ValueError("Stage2 Lite hidden_dim must match Stage-1 embeddings.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("decoder_lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    path_candidates = _coupled_path_candidates(build)
    rna = _row_max_normalize(build.state.factors.rna_norm)
    atac = _row_max_normalize(build.state.factors.atac_tfidf)
    batch_cells = int(train_cfg.get("batch_size_cells", 32))
    fallback_paths = min(
        int(train_cfg.get("tp_edges_per_cell", 128)),
        int(train_cfg.get("pg_edges_per_cell", 128)),
    )
    paths_per_cell = int(train_cfg.get("paths_per_cell", fallback_paths))
    if min(batch_cells, paths_per_cell) <= 0:
        raise ValueError("Stage2 Lite batch sizes must be positive.")
    generator = torch.Generator().manual_seed(seed)
    rows: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(train_cfg.get("epochs", 100)) + 1):
        model.train()
        optimizer.zero_grad()
        path_cell, path_tf, path_peak, path_gene = _sample_coupled_paths(
            len(embeddings["cell"]),
            path_candidates,
            batch_cells,
            paths_per_cell,
            generator,
        )
        tp_logits, _ = model.score_tf_peak(
            embeddings["cell"][path_cell],
            embeddings["gene"][path_tf],
            embeddings["peak"][path_peak],
        )
        pg_logits, _ = model.score_peak_gene(
            embeddings["cell"][path_cell],
            embeddings["peak"][path_peak],
            embeddings["gene"][path_gene],
        )
        tp_target = _activity_values(rna, path_cell, path_tf) * _activity_values(
            atac, path_cell, path_peak
        )
        pg_target = _activity_values(atac, path_cell, path_peak) * _activity_values(
            rna, path_cell, path_gene
        )
        tp_loss = F.binary_cross_entropy_with_logits(tp_logits, tp_target)
        pg_loss = F.binary_cross_entropy_with_logits(pg_logits, pg_target)
        total = tp_loss + pg_loss
        total.backward()
        optimizer.step()
        current = float(total.detach())
        rows.append(
            {
                "epoch": float(epoch),
                "total_loss": current,
                "tf_peak_loss": float(tp_loss.detach()),
                "peak_gene_loss": float(pg_loss.detach()),
                "tp_triple_count": float(len(path_cell)),
                "pg_triple_count": float(len(path_cell)),
                "path_triple_count": float(len(path_cell)),
                "shared_training_peak_count": float(len(path_candidates.peaks)),
                "path_peak_alignment": 1.0,
                "stage1_embeddings_frozen": 1.0,
            }
        )
        if current < best_loss:
            best_loss = current
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Stage2 Lite training produced no checkpoint.")
    model.load_state_dict(best_state)
    if not all(torch.equal(snapshot[name], embeddings[name]) for name in snapshot):
        raise RuntimeError("Stage-1 embeddings changed during Stage2 Lite training.")
    log = pd.DataFrame(rows)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log.to_csv(output_dir / "logs" / "stage2_lite_training_log.csv", index=False)
    torch.save(
        {"model_state": best_state, "model_config": model.model_config, "config": config},
        output_dir / "checkpoints" / "stage2_lite_best.pt",
    )
    return model, log
