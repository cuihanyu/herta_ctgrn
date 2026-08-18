"""GLUE-style genomic annotation helpers for AnnData objects."""

from __future__ import annotations

import bisect
import gzip
import heapq
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from anndata import AnnData

BED_COLUMNS = [
    "chrom",
    "chromStart",
    "chromEnd",
    "name",
    "score",
    "strand",
    "thickStart",
    "thickEnd",
    "itemRgb",
    "blockCount",
    "blockSizes",
    "blockStarts",
]

GTF_COLUMNS = [
    "seqname",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attribute",
]


def _validate_genome_build(genome_build: str | None, *, required: bool) -> str | None:
    if genome_build is None:
        if required:
            raise ValueError("A non-empty genome_build is required for genomic annotation.")
        return None
    normalized = str(genome_build).strip()
    if not normalized:
        raise ValueError("genome_build must be a non-empty string.")
    return normalized


def _validate_intervals(
    table: pd.DataFrame,
    start_column: str,
    end_column: str,
    *,
    context: str,
    one_based: bool = False,
) -> None:
    starts = pd.to_numeric(table[start_column], errors="raise")
    ends = pd.to_numeric(table[end_column], errors="raise")
    minimum = 1 if one_based else 0
    invalid = (starts < minimum) | (ends < starts if one_based else ends <= starts)
    if bool(invalid.any()):
        examples = table.loc[invalid, [start_column, end_column]].head(3).to_dict("records")
        raise ValueError(f"{context} contains {int(invalid.sum())} invalid intervals: {examples}")


def _standard_chromosome(chromosome: object) -> bool:
    value = str(chromosome)
    if value.startswith("chr"):
        value = value[3:]
    return bool(re.fullmatch(r"(?:[0-9]+|X|Y|M|MT)", value))


def read_bed(bed: str | Path) -> pd.DataFrame:
    """Read a plain or gzipped BED file with GLUE-compatible column names."""

    loaded = pd.read_csv(bed, sep="\t", header=None, comment="#")
    if loaded.shape[1] < 3:
        raise ValueError("BED files must contain at least three columns.")
    if loaded.shape[1] > len(BED_COLUMNS):
        loaded = loaded.iloc[:, : len(BED_COLUMNS)]
    loaded.columns = BED_COLUMNS[: loaded.shape[1]]
    for column in BED_COLUMNS:
        if column not in loaded:
            if column in {"chrom", "chromStart", "chromEnd"}:
                raise ValueError(f"Required BED column '{column}' is missing.")
            loaded[column] = "."
    loaded["chromStart"] = loaded["chromStart"].astype(int)
    loaded["chromEnd"] = loaded["chromEnd"].astype(int)
    if loaded["chrom"].isna().any() or (loaded["chrom"].astype(str).str.strip() == "").any():
        raise ValueError("BED chromosome values must be non-empty.")
    _validate_intervals(loaded, "chromStart", "chromEnd", context="BED")
    for column in BED_COLUMNS:
        if column not in {"chromStart", "chromEnd"}:
            loaded[column] = loaded[column].astype(str)
    return loaded.loc[:, BED_COLUMNS]


def read_motif_tf_names(bed: str | Path) -> list[str]:
    """Stream unique TF names from a plain or gzipped motif BED.

    This lightweight reader is sufficient for Stage 1, where motif-supported
    TFs define the gene-node universe but motif intervals are not regulatory
    training edges. Names retain their first-occurrence order.
    """

    source = Path(bed)
    if not source.exists():
        raise FileNotFoundError(source)
    opener = gzip.open if source.suffix == ".gz" else open
    names: dict[str, None] = {}
    with opener(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4 or not fields[3].strip():
                continue
            names.setdefault(fields[3].strip(), None)
    if not names:
        raise ValueError("Motif BED contains no non-empty TF names in column 4.")
    return list(names)


def stream_filter_motif_bed(
    source: str | Path,
    destination: str | Path,
    peaks: AnnData | pd.DataFrame,
    *,
    allowed_tfs: pd.Index | list[str] | set[str] | None = None,
    max_records_per_tf: int | None = None,
    reuse_existing: bool = False,
) -> dict[str, object]:
    """Stream a large four-column motif BED into a peak-restricted cache.

    A motif record is retained when its genomic interval overlaps at least one
    supplied peak and its TF is present in ``allowed_tfs`` (when provided).
    The source is never loaded into memory and is never modified.
    """

    source, destination = Path(source), Path(destination)
    if not source.exists():
        raise FileNotFoundError(source)
    peak_bed = _as_peak_bed(peaks)
    intervals: dict[str, tuple[list[int], list[int]]] = {}
    for chrom, frame in peak_bed.groupby("chrom", sort=False, observed=True):
        ordered = frame.sort_values(["chromStart", "chromEnd"], kind="stable")
        starts = ordered["chromStart"].astype(int).tolist()
        ends = ordered["chromEnd"].astype(int).tolist()
        prefix_max: list[int] = []
        current = -1
        for end in ends:
            current = max(current, end)
            prefix_max.append(current)
        intervals[str(chrom)] = starts, prefix_max
    if max_records_per_tf is not None and max_records_per_tf <= 0:
        raise ValueError("max_records_per_tf must be positive or None.")
    tf_filter = None if allowed_tfs is None else set(map(str, allowed_tfs))

    if reuse_existing and destination.exists():
        cached = read_bed(destination)
        retained_tfs = sorted(cached["name"].astype(str).unique())
        return {
            "source": str(source),
            "destination": str(destination),
            "input_records": None,
            "retained_records": int(len(cached)),
            "invalid_records": 0,
            "input_peak_count": int(len(peak_bed)),
            "retained_tf_count": int(len(retained_tfs)),
            "max_records_per_tf": max_records_per_tf,
            "retained_tfs": retained_tfs,
            "cache_hit": True,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if source.suffix == ".gz" else open
    output_opener = gzip.open if destination.suffix == ".gz" else open
    temporary = destination.with_name(destination.name + ".tmp")
    input_records = retained_records = invalid_records = 0
    retained_tfs: set[str] = set()
    retained_per_tf: dict[str, int] = {}
    try:
        with opener(source, "rt") as reader, output_opener(temporary, "wt") as writer:
            for line in reader:
                if not line.strip() or line.startswith("#"):
                    continue
                input_records += 1
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 4:
                    invalid_records += 1
                    continue
                chrom, tf = fields[0], fields[3].strip()
                if tf_filter is not None and tf not in tf_filter:
                    continue
                if (
                    max_records_per_tf is not None
                    and retained_per_tf.get(tf, 0) >= max_records_per_tf
                ):
                    continue
                try:
                    start, end = int(fields[1]), int(fields[2])
                except ValueError:
                    invalid_records += 1
                    continue
                if start < 0 or end <= start:
                    invalid_records += 1
                    continue
                chrom_intervals = intervals.get(chrom)
                if chrom_intervals is None:
                    continue
                starts, prefix_max = chrom_intervals
                right = bisect.bisect_left(starts, end)
                if right == 0 or prefix_max[right - 1] <= start:
                    continue
                writer.write("\t".join((chrom, str(start), str(end), tf)) + "\n")
                retained_records += 1
                retained_tfs.add(tf)
                retained_per_tf[tf] = retained_per_tf.get(tf, 0) + 1
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "source": str(source),
        "destination": str(destination),
        "input_records": input_records,
        "retained_records": retained_records,
        "invalid_records": invalid_records,
        "input_peak_count": int(len(peak_bed)),
        "retained_tf_count": int(len(retained_tfs)),
        "max_records_per_tf": max_records_per_tf,
        "retained_tfs": sorted(retained_tfs),
        "cache_hit": False,
    }


def read_gtf(gtf: str | Path) -> pd.DataFrame:
    """Read a plain or gzipped GTF file with GLUE-compatible column names."""

    df = pd.read_csv(gtf, sep="\t", header=None, comment="#", names=GTF_COLUMNS)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    _validate_intervals(df, "start", "end", context="GTF", one_based=True)
    return df


def split_gtf_attributes(gtf: pd.DataFrame) -> pd.DataFrame:
    """Append parsed GTF attributes as columns."""

    pattern = re.compile(r'([^\s]+) "([^"]+)";')
    attrs = pd.DataFrame.from_records(
        [{key: val for key, val in pattern.findall(item)} for item in gtf["attribute"]],
        index=gtf.index,
    )
    return gtf.assign(**attrs)


def _gtf_to_bed(gtf: pd.DataFrame, name: str) -> pd.DataFrame:
    bed = gtf.loc[:, ["seqname", "start", "end", "score", "strand"]].copy()
    bed.insert(3, "name", gtf[name].astype(str))
    bed["start"] -= 1
    bed.columns = ["chrom", "chromStart", "chromEnd", "name", "score", "strand"]
    return bed


def get_gene_annotation(
    adata: AnnData,
    var_by: str | None = None,
    gtf: str | Path | None = None,
    gtf_by: str | None = None,
    by_func: Callable[[object], object] | None = None,
    *,
    genome_build: str | None = None,
    require_genome_build: bool = False,
    strip_version: bool = False,
    duplicate_strategy: str = "prefer_standard",
) -> None:
    """Annotate genes with GLUE-style BED coordinates from a GTF file."""

    if gtf is None:
        raise ValueError("Missing required argument `gtf`.")
    if gtf_by is None:
        raise ValueError("Missing required argument `gtf_by`.")
    if gtf_by not in {"gene_name", "gene_id"}:
        raise ValueError("`gtf_by` must be either 'gene_name' or 'gene_id'.")
    if duplicate_strategy not in {"error", "first", "last", "prefer_standard"}:
        raise ValueError(
            "duplicate_strategy must be 'error', 'first', 'last', or 'prefer_standard'."
        )
    genome_build = _validate_genome_build(genome_build, required=require_genome_build)

    var_key = pd.Index(adata.var_names if var_by is None else adata.var[var_by])
    gtf_df = split_gtf_attributes(read_gtf(gtf).query("feature == 'gene'"))
    if gtf_by not in gtf_df:
        raise ValueError(f"GTF attributes do not contain '{gtf_by}'.")

    if by_func is not None:
        transform = np.vectorize(by_func)
        var_key = pd.Index(transform(var_key))
        gtf_df[gtf_by] = transform(gtf_df[gtf_by])
    if strip_version:
        var_key = var_key.astype(str).str.replace(r"\.\d+$", "", regex=True)
        gtf_df[gtf_by] = gtf_df[gtf_by].astype(str).str.replace(
            r"\.\d+$", "", regex=True
        )

    duplicate_mask = gtf_df[gtf_by].duplicated(keep=False)
    duplicate_records = int(duplicate_mask.sum())
    duplicate_keys = int(gtf_df.loc[duplicate_mask, gtf_by].nunique())
    if duplicate_keys and duplicate_strategy == "error":
        examples = gtf_df.loc[duplicate_mask, gtf_by].astype(str).drop_duplicates().head(3)
        raise ValueError(
            f"GTF contains {duplicate_keys} duplicated {gtf_by} keys: {examples.tolist()}"
        )
    if duplicate_strategy == "prefer_standard":
        gtf_df = gtf_df.assign(
            _standard_chrom=gtf_df["seqname"].map(_standard_chromosome)
        ).sort_values([gtf_by, "_standard_chrom", "seqname"])
        gtf_df = gtf_df.drop_duplicates(subset=[gtf_by], keep="last").drop(
            columns="_standard_chrom"
        )
    elif duplicate_strategy in {"first", "last"}:
        gtf_df = gtf_df.drop_duplicates(subset=[gtf_by], keep=duplicate_strategy)

    annotation = pd.DataFrame(
        {
            "gene_id": (
                gtf_df["gene_id"].astype(str).to_numpy()
                if "gene_id" in gtf_df
                else np.full(len(gtf_df), None, dtype=object)
            ),
            "gene_name": (
                gtf_df["gene_name"].astype(str).to_numpy()
                if "gene_name" in gtf_df
                else np.full(len(gtf_df), None, dtype=object)
            ),
            "chrom": gtf_df["seqname"].astype(str).to_numpy(),
            "chromStart": gtf_df["start"].astype(int).to_numpy() - 1,
            "chromEnd": gtf_df["end"].astype(int).to_numpy(),
            "strand": gtf_df["strand"].astype(str).to_numpy(),
        },
        index=gtf_df[gtf_by].astype(str),
    )
    annotation["tss"] = np.where(
        annotation["strand"] == "+",
        annotation["chromStart"],
        annotation["chromEnd"] - 1,
    )
    if "gene_type" in gtf_df:
        annotation["gene_biotype"] = gtf_df["gene_type"].to_numpy()
    elif "gene_biotype" in gtf_df:
        annotation["gene_biotype"] = gtf_df["gene_biotype"].to_numpy()
    if genome_build is not None:
        annotation["genome_build"] = genome_build

    annotation = annotation.reindex(var_key.astype(str))
    annotation.index = adata.var.index
    adata.var = adata.var.assign(**annotation)
    missing = int(adata.var["chrom"].isna().sum())
    nonstandard = int(
        annotation["chrom"].dropna().map(lambda value: not _standard_chromosome(value)).sum()
    )
    adata.uns["herta_gene_annotation"] = {
        "gtf": str(gtf),
        "gtf_by": gtf_by,
        "genome_build": genome_build,
        "coordinate_system": "0-based_half-open",
        "matched": int(adata.n_vars - missing),
        "missing": missing,
        "duplicate_records": duplicate_records,
        "duplicate_keys": duplicate_keys,
        "duplicate_strategy": duplicate_strategy,
        "nonstandard_contigs": nonstandard,
        "strip_version": strip_version,
    }


def parse_peak_names(
    peak_names: pd.Index | list[str],
    *,
    drop_invalid: bool = False,
) -> pd.DataFrame:
    """Parse colon- or hyphen-delimited peak names into BED-like columns."""

    peak_index = pd.Index(map(str, peak_names))
    if peak_index.has_duplicates:
        examples = peak_index[peak_index.duplicated()].unique()[:3].tolist()
        raise ValueError(f"Peak identifiers must be unique; duplicates include: {examples}")
    rows: list[dict[str, object]] = []
    invalid: list[str] = []
    for name in peak_index:
        match = re.fullmatch(r"(.+):(\d+)-(\d+)", name)
        if match is None:
            match = re.fullmatch(r"(.+)-(\d+)-(\d+)", name)
        if match is None:
            invalid.append(name)
            continue
        chrom, start, end = match.groups()
        start_value, end_value = int(start), int(end)
        if start_value < 0 or end_value <= start_value:
            invalid.append(name)
            continue
        rows.append(
            {
                "peak": name,
                "original_peak_id": name,
                "chrom": chrom,
                "chromStart": start_value,
                "chromEnd": end_value,
            }
        )
    if invalid and not drop_invalid:
        preview = ", ".join(invalid[:3])
        raise ValueError(f"Failed to parse {len(invalid)} peak names: {preview}")
    result = pd.DataFrame(
        rows,
        columns=["peak", "original_peak_id", "chrom", "chromStart", "chromEnd"],
    )
    result.attrs["n_invalid"] = len(invalid)
    result.attrs["invalid_examples"] = invalid[:3]
    return result


def parse_peak_coordinates(
    adata: AnnData,
    *,
    genome_build: str | None = None,
    require_genome_build: bool = False,
    use_existing: bool = True,
    drop_invalid: bool = False,
) -> None:
    """Annotate ATAC peaks in ``adata.var`` with GLUE-style BED coordinates."""

    genome_build = _validate_genome_build(genome_build, required=require_genome_build)
    if adata.var_names.has_duplicates:
        raise ValueError("ATAC peak identifiers must be unique.")
    n_input = int(adata.n_vars)
    existing = {"chrom", "chromStart", "chromEnd"}.issubset(adata.var.columns)
    if use_existing and existing:
        peaks = adata.var.loc[:, ["chrom", "chromStart", "chromEnd"]].copy()
        chrom = peaks["chrom"].astype("string").str.strip()
        starts = pd.to_numeric(peaks["chromStart"], errors="coerce")
        ends = pd.to_numeric(peaks["chromEnd"], errors="coerce")
        valid = (
            chrom.notna()
            & ~chrom.isin(["", ".", "nan", "None"])
            & starts.notna()
            & ends.notna()
            & (starts >= 0)
            & (ends > starts)
        ).to_numpy(dtype=bool)
        if not bool(valid.all()):
            if not drop_invalid:
                _validate_intervals(peaks, "chromStart", "chromEnd", context="ATAC peaks")
                raise ValueError(f"ATAC peaks contain {int((~valid).sum())} invalid chromosomes.")
            adata._inplace_subset_var(valid)
            peaks = adata.var.loc[:, ["chrom", "chromStart", "chromEnd"]].copy()
        peaks["chromStart"] = pd.to_numeric(peaks["chromStart"], errors="raise").astype(int)
        peaks["chromEnd"] = pd.to_numeric(peaks["chromEnd"], errors="raise").astype(int)
        peaks.insert(0, "original_peak_id", adata.var_names.astype(str))
        peaks.insert(0, "peak", adata.var_names.astype(str))
        _validate_intervals(peaks, "chromStart", "chromEnd", context="ATAC peaks")
    else:
        peaks = parse_peak_names(adata.var_names, drop_invalid=drop_invalid)
        if drop_invalid and len(peaks) != adata.n_vars:
            valid_names = pd.Index(peaks["peak"].astype(str))
            valid = adata.var_names.astype(str).isin(valid_names)
            adata._inplace_subset_var(np.asarray(valid, dtype=bool))
            peaks = peaks.set_index("peak").loc[adata.var_names.astype(str)]
            peaks.index.name = "peak"
            peaks = peaks.reset_index()
    peaks = peaks.set_index("peak")
    peaks = peaks.reindex(adata.var_names).set_index(adata.var.index)
    if genome_build is not None:
        peaks["genome_build"] = genome_build
    adata.var = adata.var.assign(**peaks)
    adata.var["coordinate_valid"] = True
    adata.uns["herta_peak_annotation"] = {
        "genome_build": genome_build,
        "coordinate_system": "0-based_half-open",
        "n_input": n_input,
        "parsed": int(adata.n_vars),
        "dropped_invalid": int(n_input - adata.n_vars),
        "missing": int(adata.var["chrom"].isna().sum()),
        "source": "existing_columns" if use_existing and existing else "var_names",
        "nonstandard_contigs": int(
            adata.var["chrom"].map(lambda value: not _standard_chromosome(value)).sum()
        ),
    }


def dist_power_decay(distance: int | float | np.ndarray) -> float | np.ndarray:
    """GLUE-style genomic distance decay weight."""

    distance = np.asarray(distance)
    if np.any(distance < 0):
        raise ValueError("Genomic distance must be non-negative.")
    weight = ((distance + 1000) / 1000) ** -0.75
    return float(weight) if weight.ndim == 0 else weight


def _as_gene_bed(genes: AnnData | pd.DataFrame) -> pd.DataFrame:
    if isinstance(genes, AnnData):
        gene_bed = genes.var.copy()
        gene_bed["name"] = genes.var_names.astype(str)
    else:
        gene_bed = genes.copy()
        if "name" not in gene_bed:
            gene_bed["name"] = gene_bed["gene"].astype(str) if "gene" in gene_bed else gene_bed.index.astype(str)

    if "chromStart" not in gene_bed and "start" in gene_bed:
        gene_bed["chromStart"] = gene_bed["start"]
    if "chromEnd" not in gene_bed and "end" in gene_bed:
        gene_bed["chromEnd"] = gene_bed["end"]
    if "chromStart" not in gene_bed and "tss" in gene_bed:
        gene_bed["chromStart"] = gene_bed["tss"]
        gene_bed["chromEnd"] = gene_bed["tss"].astype(int) + 1
    if "strand" not in gene_bed:
        gene_bed["strand"] = "."

    required = {"chrom", "chromStart", "chromEnd", "name", "strand"}
    missing = required.difference(gene_bed.columns)
    if missing:
        raise ValueError(f"Gene annotations are missing required columns: {sorted(missing)}")
    gene_bed["chromStart"] = gene_bed["chromStart"].astype(int)
    gene_bed["chromEnd"] = gene_bed["chromEnd"].astype(int)
    return gene_bed.loc[:, ["chrom", "chromStart", "chromEnd", "name", "strand"]]


def _gene_regions(genes: pd.DataFrame, gene_region: str, promoter_len: int) -> pd.DataFrame:
    if gene_region not in {"gene_body", "promoter", "combined"}:
        raise ValueError("`gene_region` must be 'gene_body', 'promoter', or 'combined'.")
    regions = genes.copy()
    if gene_region == "gene_body":
        return regions
    if not regions["strand"].isin({"+", "-"}).all():
        raise ValueError("Strand annotations '+' or '-' are required for promoter regions.")

    positive = regions["strand"] == "+"
    negative = regions["strand"] == "-"
    if gene_region == "promoter":
        regions.loc[positive, "chromEnd"] = regions.loc[positive, "chromStart"] + 1
        regions.loc[negative, "chromStart"] = regions.loc[negative, "chromEnd"] - 1
    regions.loc[positive, "chromStart"] -= promoter_len
    regions.loc[negative, "chromEnd"] += promoter_len
    regions["chromStart"] = regions["chromStart"].clip(lower=0)
    return regions


def _interval_distance(
    left_start: int,
    left_end: int,
    right_start: np.ndarray,
    right_end: np.ndarray,
) -> np.ndarray:
    overlap = (left_start < right_end) & (right_start < left_end)
    left_gap = left_end <= right_start
    distances = np.where(left_gap, right_start - left_end + 1, left_start - right_end + 1)
    return np.where(overlap, 0, distances).astype(int)


def gene_peak_prior(
    genes: AnnData | pd.DataFrame,
    peaks: AnnData | pd.DataFrame,
    gene_region: str = "combined",
    promoter_len: int = 2000,
    extend_range: int = 0,
    max_genes_per_peak: int | None = None,
) -> pd.DataFrame:
    """Construct GLUE-style gene-peak prior links from genomic intervals."""

    if promoter_len < 0 or extend_range < 0:
        raise ValueError("`promoter_len` and `extend_range` must be non-negative.")
    if max_genes_per_peak is not None and max_genes_per_peak <= 0:
        raise ValueError("`max_genes_per_peak` must be positive when specified.")

    gene_bed = _gene_regions(_as_gene_bed(genes), gene_region, promoter_len)
    peak_bed = _as_peak_bed(peaks)
    rows: list[dict[str, object]] = []
    for chrom, peak_sub in peak_bed.groupby("chrom", sort=False):
        gene_sub = gene_bed.loc[gene_bed["chrom"].astype(str) == str(chrom)]
        if gene_sub.empty:
            continue
        gene_starts = gene_sub["chromStart"].to_numpy(dtype=int)
        gene_ends = gene_sub["chromEnd"].to_numpy(dtype=int)
        for peak in peak_sub.itertuples(index=False):
            distances = _interval_distance(
                int(peak.chromStart),
                int(peak.chromEnd),
                gene_starts,
                gene_ends,
            )
            keep = np.flatnonzero(distances <= extend_range)
            if keep.size == 0:
                continue
            keep = keep[np.argsort(distances[keep], kind="stable")]
            if max_genes_per_peak is not None:
                keep = keep[:max_genes_per_peak]
            for idx in keep:
                dist = int(distances[idx])
                rows.append(
                    {
                        "gene": str(gene_sub.iloc[idx]["name"]),
                        "peak": str(peak.name),
                        "dist": dist,
                        "weight": dist_power_decay(dist),
                        "sign": 1,
                    }
                )
    return pd.DataFrame(rows, columns=["gene", "peak", "dist", "weight", "sign"])


def load_jaspar_motifs(
    bed: str | Path,
    *,
    genome_build: str,
    source_version: str,
    tf_column: str = "name",
    motif_id_column: str | None = None,
    pvalue_column: str | None = None,
) -> pd.DataFrame:
    """Load JASPAR/TFBS BED records into a provenance-preserving table.

    The returned intervals are candidate binding evidence only. They are not
    interpreted as observed TF binding or as activation/repression labels.
    """

    genome_build = _validate_genome_build(genome_build, required=True)
    source_version = str(source_version).strip()
    if not source_version:
        raise ValueError("source_version must be a non-empty string.")
    motifs = read_bed(bed)
    for requested in (tf_column, motif_id_column, pvalue_column):
        if requested is not None and requested not in motifs:
            raise ValueError(f"JASPAR BED does not contain requested column '{requested}'.")
    tf_values = motifs[tf_column].astype(str).str.strip()
    invalid_tf = tf_values.isin({"", ".", "nan", "None"})
    if bool(invalid_tf.any()):
        raise ValueError(f"JASPAR BED contains {int(invalid_tf.sum())} empty TF identifiers.")
    motif_ids = (
        motifs[motif_id_column].astype(str)
        if motif_id_column is not None
        else tf_values
    )
    motif_scores = pd.to_numeric(motifs["score"], errors="coerce")
    pvalues = (
        pd.to_numeric(motifs[pvalue_column], errors="coerce")
        if pvalue_column is not None
        else pd.Series(np.nan, index=motifs.index, dtype=float)
    )
    standardized = motifs.copy()
    standardized["tf"] = tf_values
    standardized["motif_id_or_site_id"] = motif_ids
    standardized["motif_score"] = motif_scores
    standardized["pvalue"] = pvalues
    standardized["source"] = "JASPAR"
    standardized["source_version"] = source_version
    standardized["genome_build"] = genome_build
    standardized["coordinate_system"] = "0-based_half-open"
    return standardized


def extract_tf_list(
    motifs: pd.DataFrame | str | Path,
    genes: AnnData | pd.DataFrame | pd.Index | list[str],
    *,
    genome_build: str | None = None,
    source_version: str | None = None,
    gene_column: str = "gene",
) -> pd.DataFrame:
    """Return motif-supported TF genes with inspectable hit counts."""

    if isinstance(motifs, str | Path):
        if genome_build is None or source_version is None:
            raise ValueError(
                "genome_build and source_version are required when loading motifs from a path."
            )
        motif_table = load_jaspar_motifs(
            motifs,
            genome_build=genome_build,
            source_version=source_version,
        )
    else:
        motif_table = motifs.copy()
    tf_col = "tf" if "tf" in motif_table else "name"
    if tf_col not in motif_table:
        raise ValueError("Motif table must contain either a 'tf' or 'name' column.")

    if isinstance(genes, AnnData):
        gene_names = pd.Index(genes.var_names.astype(str))
    elif isinstance(genes, pd.DataFrame):
        if gene_column in genes:
            gene_names = pd.Index(genes[gene_column].astype(str))
        else:
            gene_names = pd.Index(genes.index.astype(str))
    else:
        gene_names = pd.Index(genes).astype(str)
    if gene_names.has_duplicates:
        raise ValueError("Gene identifiers must be unique when extracting TFs.")

    motif_table[tf_col] = motif_table[tf_col].astype(str)
    hit_counts = motif_table.groupby(tf_col, sort=False).size()
    motif_id_col = (
        "motif_id_or_site_id"
        if "motif_id_or_site_id" in motif_table
        else tf_col
    )
    motif_ids = motif_table.groupby(tf_col, sort=False)[motif_id_col].agg(
        lambda values: ",".join(sorted(set(map(str, values))))
    )
    supported = gene_names[gene_names.isin(hit_counts.index)]
    return pd.DataFrame(
        {
            "gene": supported,
            "is_tf": True,
            "n_motif_hits": supported.map(hit_counts).astype(int),
            "motif_ids": supported.map(motif_ids).astype(str),
        }
    ).reset_index(drop=True)


def extract_eligible_tfs(motif_bed: pd.DataFrame | str | Path, rna: AnnData | pd.Index | list[str]) -> pd.Index:
    """Return TFs covered by both motif hits and RNA genes, following GLUE."""

    motif = read_bed(motif_bed) if isinstance(motif_bed, str | Path) else motif_bed
    return pd.Index(extract_tf_list(motif, rna)["gene"])


def _as_peak_bed(peaks: AnnData | pd.DataFrame) -> pd.DataFrame:
    if isinstance(peaks, AnnData):
        peak_bed = peaks.var.copy()
        peak_bed["name"] = peaks.var_names.astype(str)
    else:
        peak_bed = peaks.copy()
        if "name" not in peak_bed:
            peak_bed["name"] = peak_bed["peak"].astype(str) if "peak" in peak_bed else peak_bed.index.astype(str)
    if "chromStart" not in peak_bed and "start" in peak_bed:
        peak_bed["chromStart"] = peak_bed["start"]
    if "chromEnd" not in peak_bed and "end" in peak_bed:
        peak_bed["chromEnd"] = peak_bed["end"]
    required = {"chrom", "chromStart", "chromEnd", "name"}
    missing = required.difference(peak_bed.columns)
    if missing:
        raise ValueError(f"Peak annotations are missing required columns: {sorted(missing)}")
    peak_bed["chromStart"] = peak_bed["chromStart"].astype(int)
    peak_bed["chromEnd"] = peak_bed["chromEnd"].astype(int)
    return peak_bed.loc[:, ["chrom", "chromStart", "chromEnd", "name"]]


def peak_tf_links(
    peaks: AnnData | pd.DataFrame,
    motif_bed: pd.DataFrame | str | Path,
    tfs: pd.Index | list[str] | None = None,
) -> pd.DataFrame:
    """Build a simple peak-TF link table from peak/motif genomic overlaps."""

    peak_bed = _as_peak_bed(peaks)
    motif = read_bed(motif_bed) if isinstance(motif_bed, str | Path) else motif_bed.copy()
    tf_column = "tf" if "tf" in motif else "name"
    if tfs is not None:
        motif = motif.loc[motif[tf_column].astype(str).isin(set(map(str, tfs)))]

    motif = motif.copy()
    motif["chrom"] = motif["chrom"].astype(str)
    motif["chromStart"] = pd.to_numeric(motif["chromStart"], errors="raise").astype(int)
    motif["chromEnd"] = pd.to_numeric(motif["chromEnd"], errors="raise").astype(int)
    motif_by_chrom = {
        str(chrom): frame.sort_values(["chromStart", "chromEnd"], kind="stable").reset_index(drop=True)
        for chrom, frame in motif.groupby("chrom", sort=False)
    }

    rows: list[dict[str, object]] = []
    for chrom, peak_sub in peak_bed.groupby("chrom", sort=False):
        motif_sub = motif_by_chrom.get(str(chrom))
        if motif_sub is None:
            continue
        if motif_sub.empty:
            continue
        peak_sub = peak_sub.sort_values(["chromStart", "chromEnd"], kind="stable")
        motif_starts = motif_sub["chromStart"].to_numpy(dtype=int)
        motif_ends = motif_sub["chromEnd"].to_numpy(dtype=int)
        active: set[int] = set()
        expiry_heap: list[tuple[int, int]] = []
        motif_cursor = 0
        for _, peak in peak_sub.iterrows():
            peak_start = int(peak["chromStart"])
            peak_end = int(peak["chromEnd"])
            while motif_cursor < len(motif_sub) and motif_starts[motif_cursor] < peak_end:
                active.add(motif_cursor)
                heapq.heappush(
                    expiry_heap, (int(motif_ends[motif_cursor]), motif_cursor)
                )
                motif_cursor += 1
            while expiry_heap and expiry_heap[0][0] <= peak_start:
                _, expired = heapq.heappop(expiry_heap)
                active.discard(expired)
            hit_ids = sorted(
                index
                for index in active
                if motif_starts[index] < peak_end and motif_ends[index] > peak_start
            )
            for hit_id in hit_ids:
                hit = motif_sub.iloc[hit_id]
                overlap_bp = min(peak_end, int(hit["chromEnd"])) - max(
                    peak_start, int(hit["chromStart"])
                )
                rows.append(
                    {
                        "peak": peak["name"],
                        "tf": str(hit[tf_column]),
                        "chrom": chrom,
                        "peak_chromStart": int(peak["chromStart"]),
                        "peak_chromEnd": int(peak["chromEnd"]),
                        "motif_chromStart": int(hit["chromStart"]),
                        "motif_chromEnd": int(hit["chromEnd"]),
                        "motif_id_or_site_id": str(
                            hit.get("motif_id_or_site_id", hit[tf_column])
                        ),
                        "motif_score": pd.to_numeric(
                            pd.Series([hit.get("motif_score", hit.get("score", np.nan))]),
                            errors="coerce",
                        ).iloc[0],
                        "pvalue": pd.to_numeric(
                            pd.Series([hit.get("pvalue", np.nan)]), errors="coerce"
                        ).iloc[0],
                        "strand": str(hit.get("strand", ".")),
                        "source": str(hit.get("source", "JASPAR")),
                        "source_version": str(hit.get("source_version", "unspecified")),
                        "genome_build": hit.get("genome_build"),
                        "overlap_bp": int(overlap_bp),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "peak",
                "tf",
                "weight",
                "n_motif_hits",
                "chrom",
                "peak_chromStart",
                "peak_chromEnd",
                "motif_chromStart",
                "motif_chromEnd",
                "motif_ids",
                "motif_score",
                "pvalue",
                "strand",
                "source",
                "source_version",
                "genome_build",
                "overlap_bp",
            ]
        )
    links = pd.DataFrame(rows)
    grouped = (
        links.groupby(["peak", "tf"], as_index=False, sort=False)
        .agg(
            n_motif_hits=("tf", "size"),
            chrom=("chrom", "first"),
            peak_chromStart=("peak_chromStart", "first"),
            peak_chromEnd=("peak_chromEnd", "first"),
            motif_chromStart=("motif_chromStart", "min"),
            motif_chromEnd=("motif_chromEnd", "max"),
            motif_ids=(
                "motif_id_or_site_id",
                lambda values: ",".join(sorted(set(map(str, values)))),
            ),
            motif_score=("motif_score", "max"),
            pvalue=("pvalue", "min"),
            strand=("strand", lambda values: ",".join(sorted(set(map(str, values))))),
            source=("source", lambda values: ",".join(sorted(set(map(str, values))))),
            source_version=(
                "source_version",
                lambda values: ",".join(sorted(set(map(str, values)))),
            ),
            genome_build=(
                "genome_build",
                lambda values: ",".join(
                    sorted(set(str(value) for value in values if pd.notna(value)))
                ),
            ),
            overlap_bp=("overlap_bp", "max"),
        )
    )
    grouped.insert(2, "weight", 1.0)
    return grouped


def tf_gene_prior(peak_tf: pd.DataFrame, gene_peak: pd.DataFrame) -> pd.DataFrame:
    """Compose TF-peak and gene-peak links into TF-gene path priors."""

    required_peak_tf = {"peak", "tf"}
    required_gene_peak = {"gene", "peak"}
    if missing := required_peak_tf.difference(peak_tf.columns):
        raise ValueError(f"Peak-TF links are missing required columns: {sorted(missing)}")
    if missing := required_gene_peak.difference(gene_peak.columns):
        raise ValueError(f"Gene-peak links are missing required columns: {sorted(missing)}")

    peak_tf = peak_tf.copy()
    gene_peak = gene_peak.copy()
    if "weight" not in peak_tf:
        peak_tf["weight"] = 1.0
    if "weight" not in gene_peak:
        gene_peak["weight"] = 1.0
    paths = peak_tf.loc[:, ["tf", "peak", "weight"]].merge(
        gene_peak.loc[:, ["gene", "peak", "weight"]],
        on="peak",
        suffixes=("_tf_peak", "_peak_gene"),
    )
    if paths.empty:
        return pd.DataFrame(columns=["tf", "gene", "weight", "n_mediator_peaks", "mediator_peaks"])
    paths["path_weight"] = paths["weight_tf_peak"].astype(float) * paths["weight_peak_gene"].astype(float)
    return (
        paths.groupby(["tf", "gene"], as_index=False, sort=False)
        .agg(
            weight=("path_weight", "sum"),
            n_mediator_peaks=("peak", "nunique"),
            mediator_peaks=("peak", lambda x: ",".join(sorted(set(map(str, x))))),
        )
        .sort_values("weight", ascending=False, ignore_index=True)
    )
