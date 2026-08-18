"""Optional gold-standard loaders for regulatory validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from herta.data.genomics import read_bed
from herta.data.mouse_tf_target import parse_dorothea_mouse, parse_trrust_mouse


SUPPORTED_SPECIES = {"mouse", "human"}


@dataclass
class GoldStandardResult:
    """Structured optional input that never fabricates missing gold data."""

    status: str
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == "ok" and not self.data.empty


def _species(species: str) -> str:
    normalized = str(species).strip().lower()
    if normalized not in SUPPORTED_SPECIES:
        raise ValueError("species must be 'mouse' or 'human'.")
    return normalized


def _skipped(reason: str, *, species: str, source: str) -> GoldStandardResult:
    return GoldStandardResult(
        status="skipped",
        reason=reason,
        metadata={"species": species, "source": source},
    )


def _load_table(value: pd.DataFrame | str | Path | None) -> tuple[pd.DataFrame | None, str | None]:
    if value is None:
        return None, "no input was provided"
    if isinstance(value, pd.DataFrame):
        return value.copy(), None
    path = Path(value)
    if not path.exists():
        return None, f"file does not exist: {path}"
    suffixes = "".join(path.suffixes).lower()
    separator = "\t" if any(token in suffixes for token in (".tsv", ".txt", ".bed")) else ","
    return pd.read_csv(path, sep=separator, comment="#"), None


def _find_column(frame: pd.DataFrame, *names: str, required: bool = True) -> str | None:
    lookup = {str(column).strip().casefold(): str(column) for column in frame.columns}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    if required:
        raise ValueError(f"Expected one of these columns: {list(names)}")
    return None


def _tf_target_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    species: str,
) -> pd.DataFrame:
    tf_col = _find_column(frame, "tf", "source", "regulator")
    gene_col = _find_column(frame, "gene", "target", "target_gene", "target gene")
    label_col = _find_column(frame, "label", "is_positive", required=False)
    result = pd.DataFrame(
        {
            "tf": frame[tf_col].astype(str).str.strip(),
            "gene": frame[gene_col].astype(str).str.strip(),
            "label": (
                pd.to_numeric(frame[label_col], errors="coerce").fillna(0).astype(int)
                if label_col
                else 1
            ),
            "source": source,
            "species": species,
        }
    )
    for optional in ("cell_type", "confidence", "evidence", "mor", "effect"):
        column = _find_column(frame, optional, required=False)
        if column:
            result[optional] = frame[column].to_numpy()
    result = result.dropna(subset=["tf", "gene"])
    result = result[(result["tf"] != "") & (result["gene"] != "")]
    return result.drop_duplicates().reset_index(drop=True)


def load_trrust(
    path: pd.DataFrame | str | Path | None = None,
    *,
    species: str = "mouse",
) -> GoldStandardResult:
    """Load TRRUST; missing optional input returns ``status='skipped'``."""

    species = _species(species)
    if path is None:
        return _skipped("TRRUST input was not provided", species=species, source="TRRUST")
    if isinstance(path, pd.DataFrame):
        frame = path.copy()
    else:
        file_path = Path(path)
        if not file_path.exists():
            return _skipped(
                f"TRRUST file does not exist: {file_path}",
                species=species,
                source="TRRUST",
            )
        if species == "mouse":
            parsed = parse_trrust_mouse(file_path).rename(columns={"target": "gene"})
            parsed["label"] = 1
            parsed["species"] = species
            return GoldStandardResult(
                "ok",
                parsed,
                metadata={"species": species, "source": "TRRUSTv2", "path": str(file_path)},
            )
        frame = pd.read_csv(file_path, sep="\t", comment="#")
    data = _tf_target_frame(frame, source="TRRUST", species=species)
    return GoldStandardResult("ok", data, metadata={"species": species, "source": "TRRUST"})


def load_dorothea(
    path: pd.DataFrame | str | Path | None = None,
    *,
    species: str = "mouse",
    levels: tuple[str, ...] = ("A", "B", "C"),
) -> GoldStandardResult:
    """Load a local DoRothEA table without downloading external resources."""

    species = _species(species)
    frame, reason = _load_table(path)
    if frame is None:
        return _skipped(reason or "DoRothEA unavailable", species=species, source="DoRothEA")
    if species == "mouse":
        try:
            parsed = parse_dorothea_mouse(frame, levels).rename(columns={"target": "gene"})
        except ValueError:
            parsed = _tf_target_frame(frame, source="DoRothEA", species=species)
        else:
            parsed["label"] = 1
            parsed["species"] = species
        data = parsed
    else:
        data = _tf_target_frame(frame, source="DoRothEA", species=species)
    return GoldStandardResult(
        "ok",
        data.reset_index(drop=True),
        metadata={"species": species, "source": "DoRothEA", "levels": list(levels)},
    )


def load_tf_target_gold(
    gold: (
        GoldStandardResult
        | pd.DataFrame
        | str
        | Path
        | Sequence[GoldStandardResult | pd.DataFrame | str | Path]
        | None
    ) = None,
    *,
    species: str = "mouse",
    source: str = "TF-target",
) -> GoldStandardResult:
    """Load a generic TF-target table with optional context/evidence columns."""

    species = _species(species)
    if isinstance(gold, GoldStandardResult):
        return gold
    if isinstance(gold, Sequence) and not isinstance(gold, (str, bytes, Path)):
        loaded = [
            load_tf_target_gold(item, species=species, source=source) for item in gold
        ]
        available = [item.data for item in loaded if item.available]
        if not available:
            reasons = "; ".join(filter(None, (item.reason for item in loaded)))
            return _skipped(
                reasons or "all TF-target inputs were unavailable",
                species=species,
                source=source,
            )
        data = pd.concat(available, ignore_index=True, sort=False).drop_duplicates()
        return GoldStandardResult(
            "ok",
            data.reset_index(drop=True),
            metadata={"species": species, "source": source, "n_inputs": len(loaded)},
        )
    frame, reason = _load_table(gold)
    if frame is None:
        return _skipped(reason or "TF-target gold unavailable", species=species, source=source)
    data = _tf_target_frame(frame, source=source, species=species)
    return GoldStandardResult("ok", data, metadata={"species": species, "source": source})


def load_chipseq_peaks(
    peaks: GoldStandardResult | pd.DataFrame | str | Path | None = None,
    *,
    species: str = "mouse",
    assay: str = "ChIP-seq",
    genome_build: str | None = None,
    tf_column: str | None = None,
) -> GoldStandardResult:
    """Load TF-binding intervals from ChIP-seq/CUT&RUN/CUT&Tag tables."""

    species = _species(species)
    if isinstance(peaks, GoldStandardResult):
        return peaks
    if peaks is None:
        return _skipped("independent TF-binding peaks were not provided", species=species, source=assay)
    if isinstance(peaks, pd.DataFrame):
        frame = peaks.copy()
    else:
        path = Path(peaks)
        if not path.exists():
            return _skipped(f"TF-binding file does not exist: {path}", species=species, source=assay)
        try:
            frame = read_bed(path)
        except (ValueError, TypeError):
            frame = pd.read_csv(path, sep="\t", comment="#")
    chrom = _find_column(frame, "chrom", "chr", "chromosome")
    start = _find_column(frame, "start", "chromStart", "chrom_start")
    end = _find_column(frame, "end", "chromEnd", "chrom_end")
    tf_col = tf_column or _find_column(frame, "tf", "name", "factor", "target")
    data = pd.DataFrame(
        {
            "tf": frame[tf_col].astype(str).str.strip(),
            "chrom": frame[chrom].astype(str),
            "start": pd.to_numeric(frame[start], errors="raise").astype(int),
            "end": pd.to_numeric(frame[end], errors="raise").astype(int),
            "source": assay,
            "species": species,
            "genome_build": genome_build,
        }
    )
    data = data[(data["end"] > data["start"]) & (data["start"] >= 0)]
    return GoldStandardResult(
        "ok",
        data.drop_duplicates().reset_index(drop=True),
        metadata={
            "species": species,
            "source": assay,
            "assay": assay,
            "genome_build": genome_build,
        },
    )


def load_peak_gene_gold(
    links: GoldStandardResult | pd.DataFrame | str | Path | None = None,
    *,
    species: str = "mouse",
    source: str = "peak-gene",
    genome_build: str | None = None,
) -> GoldStandardResult:
    """Load pCHi-C/Hi-C/enhancer-gene links with exact peak IDs or intervals."""

    species = _species(species)
    if isinstance(links, GoldStandardResult):
        return links
    frame, reason = _load_table(links)
    if frame is None:
        return _skipped(reason or "peak-gene gold unavailable", species=species, source=source)
    gene_col = _find_column(frame, "gene", "target", "target_gene")
    peak_col = _find_column(frame, "peak", "peak_id", "source_id", required=False)
    result = pd.DataFrame({"gene": frame[gene_col].astype(str).str.strip()})
    if peak_col:
        result["peak"] = frame[peak_col].astype(str).str.strip()
    chrom = _find_column(frame, "chrom", "chr", "chromosome", required=False)
    start = _find_column(frame, "start", "chromStart", "chrom_start", required=False)
    end = _find_column(frame, "end", "chromEnd", "chrom_end", required=False)
    if chrom and start and end:
        result["chrom"] = frame[chrom].astype(str)
        result["start"] = pd.to_numeric(frame[start], errors="raise").astype(int)
        result["end"] = pd.to_numeric(frame[end], errors="raise").astype(int)
    if "peak" not in result and not {"chrom", "start", "end"}.issubset(result.columns):
        raise ValueError("peak-gene gold requires a peak ID or chrom/start/end columns.")
    result["label"] = 1
    result["source"] = source
    result["species"] = species
    result["genome_build"] = genome_build
    return GoldStandardResult(
        "ok",
        result.drop_duplicates().reset_index(drop=True),
        metadata={"species": species, "source": source, "genome_build": genome_build},
    )


def load_eqtl_links(
    links: GoldStandardResult | pd.DataFrame | str | Path | None = None,
    *,
    species: str = "mouse",
    source: str = "eQTL",
    genome_build: str | None = None,
) -> GoldStandardResult:
    """Load eQTL links already mapped to peaks or carrying variant coordinates."""

    species = _species(species)
    if isinstance(links, GoldStandardResult):
        return links
    frame, reason = _load_table(links)
    if frame is None:
        return _skipped(reason or "eQTL links unavailable", species=species, source=source)
    gene_col = _find_column(frame, "gene", "target", "target_gene", "egene")
    peak_col = _find_column(frame, "peak", "peak_id", required=False)
    variant_col = _find_column(frame, "variant", "variant_id", "snp", "rsid", required=False)
    result = pd.DataFrame({"gene": frame[gene_col].astype(str).str.strip()})
    if peak_col:
        result["peak"] = frame[peak_col].astype(str).str.strip()
    if variant_col:
        result["variant"] = frame[variant_col].astype(str)
    chrom = _find_column(frame, "chrom", "chr", "chromosome", required=False)
    position = _find_column(frame, "position", "pos", "bp", required=False)
    start = _find_column(frame, "start", "chromStart", required=False)
    end = _find_column(frame, "end", "chromEnd", required=False)
    if chrom and position:
        pos = pd.to_numeric(frame[position], errors="raise").astype(int)
        result["chrom"] = frame[chrom].astype(str)
        result["start"] = pos
        result["end"] = pos + 1
    elif chrom and start and end:
        result["chrom"] = frame[chrom].astype(str)
        result["start"] = pd.to_numeric(frame[start], errors="raise").astype(int)
        result["end"] = pd.to_numeric(frame[end], errors="raise").astype(int)
    if "peak" not in result and not {"chrom", "start", "end"}.issubset(result.columns):
        raise ValueError("eQTL links require mapped peak IDs or genomic variant coordinates.")
    result["label"] = 1
    result["source"] = source
    result["species"] = species
    result["genome_build"] = genome_build
    return GoldStandardResult(
        "ok",
        result.drop_duplicates().reset_index(drop=True),
        metadata={"species": species, "source": source, "genome_build": genome_build},
    )


def load_perturbation_targets(
    targets: GoldStandardResult | pd.DataFrame | str | Path | None = None,
    *,
    species: str = "mouse",
    source: str = "perturbation",
) -> GoldStandardResult:
    """Load TF perturbation-responsive target genes."""

    species = _species(species)
    if isinstance(targets, GoldStandardResult):
        return targets
    frame, reason = _load_table(targets)
    if frame is None:
        return _skipped(reason or "perturbation targets unavailable", species=species, source=source)
    data = _tf_target_frame(frame, source=source, species=species)
    return GoldStandardResult("ok", data, metadata={"species": species, "source": source})
