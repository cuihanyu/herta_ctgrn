"""Prepare curated mouse TF-target gold standards for HERTA evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


TRRUST_MOUSE_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv"
TRRUST_DOWNLOAD_PAGE = "https://www.grnpedia.org/trrust/downloadnetwork.php"
REACTOME_MOUSE_URL = "https://reactome.org/download/current/NCBI2Reactome_All_Levels.txt"
DOROTHEA_BIOCONDUCTOR_URL = (
    "https://bioconductor.org/packages/3.22/data/experiment/src/contrib/dorothea_1.22.0.tar.gz"
)
STANDARD_COLUMNS = ["tf", "target", "mor", "confidence", "source", "evidence"]


def _dorothea_archive_path(raw_dir: str | Path) -> Path:
    """Return the single shared DoRothEA package archive used by both species."""

    project_root = Path(__file__).resolve().parents[2]
    shared = project_root / "data" / "human_tf_target" / "raw" / "dorothea_1.22.0.tar.gz"
    return shared if shared.exists() else Path(raw_dir) / "dorothea_1.22.0.tar.gz"


def normalize_mouse_symbol(value: object) -> str | None:
    """Normalize whitespace and common all-upper/all-lower mouse symbols."""

    if pd.isna(value):
        return None
    symbol = re.sub(r"\s+", "", str(value).strip())
    if not symbol:
        return None
    letters = "".join(char for char in symbol if char.isalpha())
    if letters and (letters.isupper() or letters.islower()):
        symbol = "-".join(part[:1].upper() + part[1:].lower() for part in symbol.split("-"))
    return symbol


def parse_trrust_mouse(path: str | Path) -> pd.DataFrame:
    """Parse the four-column TRRUST v2 mouse table into HERTA columns."""

    table = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["tf", "target", "mode", "evidence"],
        dtype=str,
        comment="#",
    )
    mode = table["mode"].fillna("").str.strip().str.lower()
    table["mor"] = mode.map({"activation": 1, "repression": -1}).fillna(0).astype(int)
    table["confidence"] = "curated"
    table["source"] = "TRRUSTv2"
    return clean_tf_target_table(table[STANDARD_COLUMNS])


def parse_dorothea_mouse(
    table: pd.DataFrame | str | Path,
    levels: Sequence[str] = ("A", "B", "C"),
) -> pd.DataFrame:
    """Normalize DoRothEA tables returned by decoupler, OmniPath, or R."""

    frame = pd.read_csv(table, sep="\t", dtype=str) if isinstance(table, str | Path) else table.copy()
    tf_col = _find_column(frame, "tf", "source", "source_genesymbol")
    target_col = _find_column(frame, "target", "gene", "target_genesymbol")
    confidence_col = _find_column(
        frame, "confidence", "level", "dorothea_level", "dorothea_levels", required=False
    )
    if confidence_col is None:
        raise ValueError("DoRothEA table does not contain a confidence-level column.")

    requested = {str(level).strip().upper() for level in levels if str(level).strip()}
    confidence = frame[confidence_col].fillna("").astype(str).str.strip().str.upper()
    frame = frame.loc[confidence.isin(requested)].copy()
    confidence = confidence.loc[frame.index]
    mor = _dorothea_mor(frame)
    evidence_col = _find_column(
        frame, "evidence", "references", "reference", "pmid", "sources", required=False
    )
    evidence = frame[evidence_col].fillna("").astype(str) if evidence_col else ""
    normalized = pd.DataFrame(
        {
            "tf": frame[tf_col],
            "target": frame[target_col],
            "mor": mor,
            "confidence": confidence,
            "source": "DoRothEA",
            "evidence": evidence,
        }
    )
    return clean_tf_target_table(normalized)


def clean_tf_target_table(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize symbols and remove incomplete, duplicate, and self-loop edges."""

    missing = set(STANDARD_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(f"TF-target table is missing columns: {sorted(missing)}")
    result = table[STANDARD_COLUMNS].copy()
    result["tf"] = result["tf"].map(normalize_mouse_symbol)
    result["target"] = result["target"].map(normalize_mouse_symbol)
    result = result.dropna(subset=["tf", "target"])
    result = result[result["tf"].str.casefold() != result["target"].str.casefold()]
    result["mor"] = pd.to_numeric(result["mor"], errors="coerce").fillna(0).map(np.sign).astype(int)
    for column in ("confidence", "source", "evidence"):
        result[column] = result[column].fillna("").astype(str).str.strip()
    return result.drop_duplicates().reset_index(drop=True)


def merge_tf_target_sources(*tables: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate TF-target edges while retaining provenance and sign conflicts."""

    combined = clean_tf_target_table(pd.concat(tables, ignore_index=True))
    rows: list[dict[str, object]] = []
    for (tf, target), group in combined.groupby(["tf", "target"], sort=True, observed=True):
        nonzero = set(group.loc[group["mor"] != 0, "mor"].astype(int))
        conflict = nonzero == {-1, 1}
        mor = 0 if conflict or not nonzero else next(iter(nonzero))
        rows.append(
            {
                "tf": tf,
                "target": target,
                "mor": mor,
                "confidence": _join_unique(group["confidence"]),
                "source": _join_unique(group["source"], preferred=("TRRUSTv2", "DoRothEA")),
                "evidence": _join_unique(group["evidence"]),
                "mor_conflict": conflict,
            }
        )
    columns = [*STANDARD_COLUMNS, "mor_conflict"]
    return pd.DataFrame(rows, columns=columns)


def filter_to_gene_list(table: pd.DataFrame, gene_list: str | Path | Iterable[str]) -> pd.DataFrame:
    """Keep edges whose TF and target are both present in a mouse gene list."""

    genes = read_gene_list(gene_list) if isinstance(gene_list, str | Path) else gene_list
    allowed = {symbol for value in genes if (symbol := normalize_mouse_symbol(value))}
    keep = table["tf"].isin(allowed) & table["target"].isin(allowed)
    return table.loc[keep].reset_index(drop=True)


def read_gene_list(path: str | Path) -> list[str]:
    """Read the first whitespace-, comma-, or tab-delimited field from each line."""

    genes: list[str] = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                genes.append(re.split(r"[\t,\s]+", line, maxsplit=1)[0])
    return genes


def fetch_dorothea_mouse(levels: Sequence[str] = ("A", "B", "C")) -> pd.DataFrame:
    """Fetch mouse DoRothEA with an installed decoupler or omnipath client."""

    errors: list[str] = []
    try:
        import decoupler as dc

        if hasattr(dc, "op") and hasattr(dc.op, "dorothea"):
            raw = dc.op.dorothea(organism="mouse", levels=list(levels))
        elif hasattr(dc, "get_dorothea"):
            raw = dc.get_dorothea(organism="mouse", levels=list(levels))
        else:
            raise AttributeError("no supported DoRothEA accessor")
        return parse_dorothea_mouse(raw, levels)
    except Exception as exc:
        errors.append(f"decoupler: {exc}")

    try:
        import omnipath as op

        try:
            raw = op.interactions.Dorothea.get(
                organism="mouse", dorothea_levels=list(levels)
            )
        except TypeError:
            raw = op.interactions.Dorothea.get(organism="mouse")
        return parse_dorothea_mouse(raw, levels)
    except Exception as exc:
        errors.append(f"omnipath: {exc}")
    raise RuntimeError("Unable to fetch mouse DoRothEA with Python clients. " + " | ".join(errors))


def parse_reactome_mouse(path: str | Path) -> pd.DataFrame:
    """Parse Reactome NCBI-to-pathway mappings as a separate mouse auxiliary table."""

    columns = ["gene_id", "pathway_id", "pathway_url", "pathway_name", "evidence", "species"]
    table = pd.read_csv(path, sep="\t", header=None, names=columns, dtype=str, comment="#")
    if table.shape[1] != len(columns):
        raise ValueError("Unexpected Reactome mapping format.")
    table = table[table["species"].fillna("").str.casefold() == "mus musculus"].copy()
    if table.empty:
        raise ValueError("Reactome mapping contains no Mus musculus records.")
    table.insert(0, "identifier_type", "NCBI Gene")
    return table.drop_duplicates().sort_values(["pathway_id", "gene_id"]).reset_index(drop=True)


def prepare_mouse_tf_target(
    output_root: str | Path,
    dorothea_levels: Sequence[str] = ("A", "B", "C"),
    gene_list: str | Path | None = None,
    out_prefix: str = "chen2019",
    include_reactome: bool = False,
    force: bool = False,
    r_script: str | Path | None = None,
) -> dict[str, Path]:
    """Download, normalize, merge, and optionally subset mouse gold standards."""

    root = Path(output_root)
    raw_dir, processed_dir = root / "raw", root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    trrust_path = raw_dir / "trrust_rawdata.mouse.tsv"
    if force or not trrust_path.exists():
        try:
            _download(TRRUST_MOUSE_URL, trrust_path)
        except (OSError, URLError) as exc:
            raise RuntimeError(
                f"TRRUST mouse download failed: {exc}. Download the mouse TSV manually from "
                f"{TRRUST_DOWNLOAD_PAGE} and save it as {trrust_path}."
            ) from exc
    trrust = parse_trrust_mouse(trrust_path)

    dorothea_path = raw_dir / "dorothea_mouse.tsv"
    if force or not dorothea_path.exists():
        try:
            dorothea = fetch_dorothea_mouse(dorothea_levels)
        except RuntimeError as python_error:
            try:
                dorothea = fetch_dorothea_bioconductor(raw_dir, dorothea_levels)
            except Exception as package_error:
                combined = RuntimeError(f"{python_error}\nBioconductor package fallback failed: {package_error}")
                _run_dorothea_r_fallback(dorothea_path, dorothea_levels, r_script, combined)
            else:
                dorothea.to_csv(dorothea_path, sep="\t", index=False)
        else:
            dorothea.to_csv(dorothea_path, sep="\t", index=False)
    dorothea = parse_dorothea_mouse(dorothea_path, dorothea_levels)

    gold = merge_tf_target_sources(trrust, dorothea)
    main_path = processed_dir / "mouse_tf_target_gold_standard.tsv"
    gold.to_csv(main_path, sep="\t", index=False)
    outputs = {"gold_standard": main_path}

    if gene_list is not None:
        prefix = _safe_prefix(out_prefix)
        subset_path = processed_dir / f"{prefix}_mouse_tf_target_gold_standard.tsv"
        filter_to_gene_list(gold, gene_list).to_csv(subset_path, sep="\t", index=False)
        outputs["gene_subset"] = subset_path

    if include_reactome:
        reactome_raw = raw_dir / "NCBI2Reactome_All_Levels.txt"
        if force or not reactome_raw.exists():
            _download(REACTOME_MOUSE_URL, reactome_raw)
        reactome_path = processed_dir / "reactome_mouse_auxiliary.tsv"
        parse_reactome_mouse(reactome_raw).to_csv(reactome_path, sep="\t", index=False)
        outputs["reactome_auxiliary"] = reactome_path

    manifest = {
        "schema_version": 1,
        "dorothea_levels": list(dorothea_levels),
        "sources": {
            "TRRUSTv2": {"url": TRRUST_MOUSE_URL, "sha256": _sha256(trrust_path)},
            "DoRothEA": {"sha256": _sha256(dorothea_path)},
        },
    }
    dorothea_archive = _dorothea_archive_path(raw_dir)
    if dorothea_archive.exists():
        manifest["sources"]["DoRothEA"]["bioconductor_archive"] = {
            "url": DOROTHEA_BIOCONDUCTOR_URL,
            "sha256": _sha256(dorothea_archive),
        }
    if gene_list is not None:
        manifest["gene_list"] = {"name": Path(gene_list).name, "sha256": _sha256(Path(gene_list))}
    if include_reactome:
        manifest["sources"]["Reactome"] = {
            "url": REACTOME_MOUSE_URL,
            "sha256": _sha256(reactome_raw),
            "usage": "auxiliary_only",
        }
    manifest_path = processed_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs["manifest"] = manifest_path
    return outputs


def fetch_dorothea_bioconductor(
    raw_dir: str | Path,
    levels: Sequence[str] = ("A", "B", "C"),
) -> pd.DataFrame:
    """Read mouse DoRothEA from the pinned Bioconductor source package."""

    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError("Install pyreadr to use the Bioconductor package fallback.") from exc

    raw_dir = Path(raw_dir)
    archive = _dorothea_archive_path(raw_dir)
    rdata = raw_dir / "dorothea_mm.rda"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        _download(DOROTHEA_BIOCONDUCTOR_URL, archive)
    if not rdata.exists():
        with tarfile.open(archive, "r:gz") as package:
            member = next(
                (item for item in package.getmembers() if item.name.endswith("/data/dorothea_mm.rda")),
                None,
            )
            if member is None:
                raise ValueError("dorothea_mm.rda is missing from the Bioconductor source package.")
            source = package.extractfile(member)
            if source is None:
                raise ValueError("Unable to read dorothea_mm.rda from the source package.")
            with rdata.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    objects = pyreadr.read_r(str(rdata))
    if not objects:
        raise ValueError("No objects were found in dorothea_mm.rda.")
    table = objects.get("dorothea_mm", next(iter(objects.values())))
    return parse_dorothea_mouse(table, levels)


def _run_dorothea_r_fallback(
    output: Path,
    levels: Sequence[str],
    r_script: str | Path | None,
    python_error: Exception,
) -> None:
    script = Path(r_script) if r_script else Path(__file__).resolve().parents[2] / "scripts" / "export_dorothea_mouse.R"
    rscript = shutil.which("Rscript")
    level_arg = ",".join(levels)
    if rscript and script.exists():
        result = subprocess.run(
            [rscript, str(script), str(output), level_arg],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and output.exists():
            return
        r_error = result.stderr.strip() or result.stdout.strip() or "unknown R error"
    else:
        r_error = "Rscript or scripts/export_dorothea_mouse.R was not found"
    raise RuntimeError(
        f"{python_error}\nR fallback failed: {r_error}. Install Python support with "
        "`pip install omnipath`, or install Bioconductor dorothea and run: "
        f"Rscript {script} {output} {level_arg}"
    ) from python_error


def _download(url: str, destination: Path, timeout: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "HERTA-ctGRN/0.1 data preparation"})
    try:
        with urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _find_column(frame: pd.DataFrame, *names: str, required: bool = True) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    if required:
        raise ValueError(f"Expected one of these columns: {list(names)}")
    return None


def _dorothea_mor(frame: pd.DataFrame) -> pd.Series:
    direct = _find_column(frame, "mor", "weight", "effect", required=False)
    if direct:
        return pd.to_numeric(frame[direct], errors="coerce").fillna(0).map(np.sign).astype(int)
    stimulation = _find_column(frame, "is_stimulation", "consensus_stimulation", required=False)
    inhibition = _find_column(frame, "is_inhibition", "consensus_inhibition", required=False)
    if stimulation or inhibition:
        positive = _boolean_series(frame[stimulation]) if stimulation else pd.Series(False, index=frame.index)
        negative = _boolean_series(frame[inhibition]) if inhibition else pd.Series(False, index=frame.index)
        return pd.Series(np.where(positive & ~negative, 1, np.where(negative & ~positive, -1, 0)), index=frame.index)
    return pd.Series(0, index=frame.index, dtype=int)


def _boolean_series(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.casefold().isin({"true", "1", "yes"})


def _join_unique(values: pd.Series, preferred: Sequence[str] = ()) -> str:
    unique = {value.strip() for value in values.astype(str) for value in value.split(";") if value.strip()}
    ordered = [value for value in preferred if value in unique]
    ordered.extend(sorted(unique.difference(ordered), key=str.casefold))
    return ";".join(ordered)


def _safe_prefix(value: str) -> str:
    if not value or Path(value).name != value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("out_prefix must be a non-empty filename-safe value.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
