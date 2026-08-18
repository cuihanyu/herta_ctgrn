"""Small genomic interval helpers."""

from __future__ import annotations

import pandas as pd
import numpy as np

from herta.data.genomics import gene_peak_prior


def peak_centers(peaks: pd.DataFrame) -> pd.Series:
    """Return integer peak centers from chrom/start/end annotations."""

    return ((peaks["start"].astype(int) + peaks["end"].astype(int)) // 2).astype(int)


def peak_gene_prior(
    peaks: pd.DataFrame,
    genes: pd.DataFrame,
    window: int,
    decay: float,
    max_genes_per_peak: int,
) -> tuple[list[int], list[int], list[float]]:
    """Compatibility wrapper around the canonical genomic prior builder."""

    peak_table = peaks.reset_index(drop=True).copy()
    gene_table = genes.reset_index(drop=True).copy()
    peak_table["name"] = peak_table.index.astype(str)
    gene_table["name"] = gene_table.index.astype(str)
    prior = gene_peak_prior(
        gene_table,
        peak_table,
        gene_region="gene_body",
        extend_range=window,
        max_genes_per_peak=max_genes_per_peak,
    )
    src = prior["peak"].astype(int).tolist()
    dst = prior["gene"].astype(int).tolist()
    weights = np.exp(-prior["dist"].to_numpy(dtype=float) / decay).astype(float).tolist()
    return src, dst, weights
