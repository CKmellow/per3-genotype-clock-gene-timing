"""Data acquisition and loading for GSE39445 (Part 1).

Every part of the project should load data **only** through these functions so
that sample recovery, time parsing and sample selection are identical
everywhere.

Public API
----------
load_metadata()                 tidy sample metadata (one row per sample, 438 rows)
load_clock_genes(collapse=...)  long-format clock-gene expression merged with metadata
load_probe_annotation()         GPL15331 probe annotation
download_series_matrix()        cached download of the GEO series matrix (~100 MB)
load_expression_matrix()        probes × samples whole-transcriptome matrix
load_series_characteristics()   per-sample characteristics parsed from the series matrix
load_soft_samples()             per-sample headers parsed from the committed SOFT file
collapse_probes_to_genes()      probe → gene aggregation
analysis_samples()              default sample set for Parts 3–5 (provisional)
"""

from __future__ import annotations

import gzip
import io
import re
import time
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from . import config

__all__ = [
    "load_metadata",
    "load_clock_genes",
    "load_probe_annotation",
    "download_series_matrix",
    "load_expression_matrix",
    "load_series_characteristics",
    "load_soft_samples",
    "collapse_probes_to_genes",
    "analysis_samples",
    "parse_clock_time",
    "clock_probe_columns",
]

_TITLE_RE = re.compile(r"^(?:Subject_)?(?P<subject>AF\d{4})_(?P<cond>SE|SR)_T(?P<tp>\d{1,2})$")
_PROBE_COL_RE = re.compile(r"^(?P<gene>[A-Z0-9]+)_(?P<probe>\d+)$")

METADATA_COLUMNS: list[str] = [
    "sample_id",
    "title",
    "subject",
    "per3_genotype",
    "sleep_condition",
    "condition_code",
    "timepoint",
    "timepoint_n",
    "hours_awake",
    "clock_time",
    "clock_time_h",
    "circadian_phase_h",
    "is_recovered",
]


# ---------------------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------------------
def parse_clock_time(values: pd.Series | list[str] | str) -> pd.Series | float:
    """Convert ``"HH:MM"`` clock times to decimal hours in ``[0, 24)``.

    ``"18:15"`` → 18.25, ``"0:15"`` → 0.25. ``"24:00"`` wraps to 0. Invalid strings
    raise ``ValueError``.
    """
    scalar = isinstance(values, str)
    s = pd.Series([values] if scalar else values, dtype="string").str.strip()
    parts = s.str.extract(r"^(\d{1,2}):(\d{2})$")
    if parts.isna().any().any():
        bad = s[parts.isna().any(axis=1)].tolist()
        raise ValueError(f"Unparseable clock time(s): {bad[:5]}")
    h = parts[0].astype(int)
    m = parts[1].astype(int)
    if ((h > 24) | (m > 59)).any():
        raise ValueError("Clock time out of range")
    out = ((h + m / 60.0) % 24.0).astype(float)
    out.index = s.index if not scalar else out.index
    return float(out.iloc[0]) if scalar else out


def load_metadata(path: Path | str | None = None) -> pd.DataFrame:
    """Load tidy per-sample metadata for all 438 GSE39445 samples.

    Steps (all verified by tests):

    * subject, condition and timepoint are parsed from ``title``
      (``Subject_AF0004_SE_T1`` or ``AF0033_SE_T2``);
    * the 11 samples GSM1526434–GSM1526444 have blank ``subject``/``per3_genotype`` in
      the source CSV. Subject is recovered from the title and genotype from that
      subject's other samples (``is_recovered = True``). GEO itself stores their
      genotype under a different key (``genotype:``), and ``scripts/verify_data.py``
      cross-checks against it;
    * ``clock_time`` ``"HH:MM"`` → ``clock_time_h`` (decimal hours, 0–24).

    Returns
    -------
    DataFrame with columns ``METADATA_COLUMNS``, one row per sample, sorted by
    subject, condition_code, timepoint_n. Types: ``hours_awake``, ``clock_time_h``,
    ``circadian_phase_h`` float; ``timepoint_n`` int; ``is_recovered`` bool;
    ``timepoint`` is ``"T1"``…``"T10"``; ``condition_code`` is ``"SE"``/``"SR"``.
    ``circadian_phase_h`` is hours relative to the participant's plasma-melatonin
    rhythm as provided by GEO (see data/README.md for caveats).
    """
    path = Path(path) if path is not None else config.METADATA_CSV
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw = raw.replace({"": np.nan})

    parsed = raw["title"].str.extract(_TITLE_RE)
    if parsed.isna().any().any():
        bad = raw.loc[parsed.isna().any(axis=1), "title"].tolist()
        raise ValueError(f"Unparseable sample titles: {bad}")

    df = pd.DataFrame({"sample_id": raw["sample_id"], "title": raw["title"]})
    is_recovered = raw["subject"].isna() | raw["per3_genotype"].isna()

    # Subject: given value must agree with title; fill blanks from the title.
    given = raw["subject"]
    mismatch = given.notna() & (given != parsed["subject"])
    if mismatch.any():
        raise ValueError(f"subject/title mismatch: {raw.loc[mismatch, 'sample_id'].tolist()}")
    df["subject"] = given.fillna(parsed["subject"])

    # Genotype: one genotype per subject, taken from the non-blank samples.
    gt = raw.assign(subject=df["subject"]).dropna(subset=["per3_genotype"])
    per_subject = gt.groupby("subject")["per3_genotype"].unique()
    conflicting = per_subject[per_subject.map(len) > 1]
    if len(conflicting):
        raise ValueError(f"Subjects with conflicting genotypes: {conflicting.to_dict()}")
    gt_map = per_subject.map(lambda a: a[0])
    df["per3_genotype"] = raw["per3_genotype"].fillna(df["subject"].map(gt_map))
    if df["per3_genotype"].isna().any():
        missing = df.loc[df["per3_genotype"].isna(), "subject"].unique().tolist()
        raise ValueError(f"Could not recover genotype for subjects: {missing}")

    df["sleep_condition"] = raw["sleep_condition"]
    df["condition_code"] = parsed["cond"]
    expected = df["condition_code"].map(config.CONDITIONS)
    if (expected != df["sleep_condition"]).any():
        raise ValueError("sleep_condition does not match condition code in title")

    df["timepoint_n"] = parsed["tp"].astype(int)
    df["timepoint"] = "T" + df["timepoint_n"].astype(str)
    df["hours_awake"] = raw["hours_awake"].astype(float)
    df["clock_time"] = raw["clock_time"]
    df["clock_time_h"] = parse_clock_time(raw["clock_time"]).to_numpy()
    df["circadian_phase_h"] = raw["circadian_phase_h"].astype(float)
    df["is_recovered"] = is_recovered.to_numpy(dtype=bool)

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in metadata")

    df = df[METADATA_COLUMNS].sort_values(["subject", "condition_code", "timepoint_n"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Clock genes (small CSV extracted from the series matrix)
# ---------------------------------------------------------------------------------------
def clock_probe_columns(columns: list[str] | pd.Index) -> dict[str, tuple[str, int]]:
    """Map clock-gene CSV column names ``GENE_PROBEID`` → ``(gene, probe_id)``."""
    out: dict[str, tuple[str, int]] = {}
    for c in columns:
        m = _PROBE_COL_RE.match(str(c))
        if m:
            out[str(c)] = (m["gene"], int(m["probe"]))
    return out


def load_clock_genes(
    collapse: Literal["mean", "median"] | None = "mean",
    path: Path | str | None = None,
) -> pd.DataFrame:
    """Load clock-gene expression in long format, merged with :func:`load_metadata`.

    Parameters
    ----------
    collapse : ``"mean"`` (default), ``"median"`` or ``None``
        How to combine multiple probes of the same gene (CRY1, NR1D1, NR1D2, PER2 have
        2 probes; PER3 has 3). ``None`` keeps probe-level rows.
    path : optional path to the clock-gene CSV (default ``config.CLOCK_GENES_CSV``).

    Returns
    -------
    Long DataFrame, one row per sample × gene (or sample × probe), with columns
    ``sample_id, gene, [probe_id,] expression, n_probes`` followed by all metadata
    columns. ``expression`` is the GEO quantile-normalised log2 signal.
    """
    path = Path(path) if path is not None else config.CLOCK_GENES_CSV
    wide = pd.read_csv(path)
    probe_map = clock_probe_columns(wide.columns)
    if not probe_map:
        raise ValueError("No GENE_PROBEID columns found in clock-gene CSV")
    long = wide.melt(
        id_vars="sample_id", value_vars=list(probe_map), var_name="probe", value_name="expression"
    )
    long["gene"] = long["probe"].map(lambda c: probe_map[c][0])
    long["probe_id"] = long["probe"].map(lambda c: probe_map[c][1]).astype(int)
    long = long.drop(columns="probe")
    n_probes = long.groupby("gene")["probe_id"].nunique().rename("n_probes")

    if collapse is None:
        out = long.join(n_probes, on="gene")[
            ["sample_id", "gene", "probe_id", "expression", "n_probes"]
        ]
    elif collapse in ("mean", "median"):
        out = long.groupby(["sample_id", "gene"], as_index=False)["expression"].agg(collapse)
        out = out.join(n_probes, on="gene")
    else:
        raise ValueError("collapse must be 'mean', 'median' or None")

    meta = load_metadata()
    out = out.merge(meta, on="sample_id", how="left", validate="many_to_one")
    if out["subject"].isna().any():
        raise ValueError("Clock-gene samples missing from metadata")
    sort_cols = ["subject", "condition_code", "timepoint_n", "gene"]
    if collapse is None:
        sort_cols.append("probe_id")
    return out.sort_values(sort_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Probe annotation
# ---------------------------------------------------------------------------------------
def load_probe_annotation(path: Path | str | None = None) -> pd.DataFrame:
    """Load the GPL15331 (Agilent) probe annotation.

    Returns
    -------
    DataFrame indexed by integer probe ``ID`` (matches the series-matrix ``ID_REF``)
    with columns ``PROBE_NAME, GENE_SYMBOL, GENE_NAME, ENSEMBL_ID,
    CHROMOSOMAL_LOCATION``. Blank strings are NaN (control probes have no symbol).
    """
    path = Path(path) if path is not None else config.PROBE_ANNOTATION_CSV
    annot = pd.read_csv(path, dtype=str, keep_default_na=False).replace({"": np.nan})
    annot["ID"] = annot["ID"].astype(int)
    return annot.set_index("ID").sort_index()


# ---------------------------------------------------------------------------------------
# Series matrix (whole transcriptome)
# ---------------------------------------------------------------------------------------
def download_series_matrix(
    force: bool = False,
    *,
    url: str = config.SERIES_MATRIX_URL,
    dest: Path | str | None = None,
    retries: int = 4,
    timeout: float = 60.0,
    expected_bytes: int | None = config.SERIES_MATRIX_BYTES,
    verbose: bool = True,
) -> Path:
    """Download the GSE39445 series matrix (~100 MB, gzipped) with caching.

    The file is cached in ``data/external/`` (gitignored). A cached copy is reused
    unless ``force=True`` or it fails the gzip integrity check. Downloads go to a
    ``.part`` file and are renamed only after the integrity check passes, so an
    interrupted download never leaves a corrupt cache. Network errors are retried
    with exponential back-off.

    Sanity checks: the gzip stream must decompress fully, and the size should equal
    ``expected_bytes`` (a mismatch only *warns*, because GEO occasionally
    regenerates matrix files). Row and sample counts are checked in
    :func:`load_expression_matrix`.

    Returns
    -------
    Path to the cached ``.txt.gz`` file.
    """
    import requests  # local import keeps `import per3clock.data` light

    dest = Path(dest) if dest is not None else config.SERIES_MATRIX_GZ
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        if _gzip_ok(dest):
            _check_size(dest, expected_bytes)
            return dest
        warnings.warn(f"Cached {dest.name} is corrupt; re-downloading", stacklevel=2)

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if verbose:
                print(f"Downloading {url} (attempt {attempt}/{retries}) ...")
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            if not _gzip_ok(tmp):
                raise OSError("downloaded file is not a valid gzip stream")
            tmp.replace(dest)
            _check_size(dest, expected_bytes)
            if verbose:
                print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except (requests.RequestException, OSError) as err:
            last_err = err
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_err


def _gzip_ok(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 24):
                pass
        return True
    except (OSError, EOFError):
        return False


def _check_size(path: Path, expected: int | None) -> None:
    size = path.stat().st_size
    if expected is not None and size != expected:
        warnings.warn(
            f"{path.name} is {size:,} bytes; expected {expected:,} (GEO may have "
            "regenerated the file). Row/sample checks in load_expression_matrix still apply.",
            stacklevel=3,
        )


def _open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def load_series_characteristics(path: Path | str | None = None) -> pd.DataFrame:
    """Parse per-sample header lines (``!Sample_...``) of the series matrix.

    Returns
    -------
    DataFrame indexed by GSM accession with ``title``, ``source_name`` and one column
    per ``Sample_characteristics_ch1`` key (e.g. ``subject``, ``sleepprotocol``,
    ``genotype/variation``, ``genotype``, ``hoursawake``, ``timesampletaken``,
    ``circadianphase``). Missing characteristics are NaN. Use it to check that
    the metadata CSV matches GEO.
    """
    path = Path(path) if path is not None else download_series_matrix(verbose=False)
    header: dict[str, list[list[str]]] = {}
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_"):
                key, *vals = line.rstrip("\n").split("\t")
                header.setdefault(key, []).append([v.strip('"') for v in vals])
    gsm = header["!Sample_geo_accession"][0]
    out = pd.DataFrame(index=pd.Index(gsm, name="sample_id"))
    out["title"] = header["!Sample_title"][0]
    out["source_name"] = header["!Sample_source_name_ch1"][0]
    chars: dict[str, list[object]] = {}
    for row in header.get("!Sample_characteristics_ch1", []):
        for i, cell in enumerate(row):
            if ": " in cell:
                k, v = cell.split(": ", 1)
                chars.setdefault(k.strip(), [np.nan] * len(gsm))[i] = v.strip()
    for k, v in chars.items():
        out[k] = v
    return out


def load_expression_matrix(
    path: Path | str | None = None,
    *,
    download: bool = True,
    with_characteristics: bool = False,
    dtype: str = "float64",
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Load the whole-transcriptome matrix (probes × samples).

    Parameters
    ----------
    path : optional local ``.txt`` or ``.txt.gz`` series matrix. Default: the cached
        download (fetched with :func:`download_series_matrix` if ``download``).
    with_characteristics : also return :func:`load_series_characteristics`.
    dtype : value dtype (``"float32"`` halves memory, ~77 MB vs ~153 MB).

    Returns
    -------
    ``expr`` (DataFrame, index ``ID_REF`` int probe IDs matching
    :func:`load_probe_annotation`, columns GSM accessions, values GEO
    quantile-normalised log2 signal), or ``(expr, characteristics)``.

    Raises
    ------
    ValueError if the matrix does not have 438 samples or has an implausible number
    of probes (<40,000 or >46,000).
    """
    if path is None:
        if download:
            path = download_series_matrix(verbose=True)
        else:
            path = config.SERIES_MATRIX_GZ
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; call download_series_matrix() first")

    with _open_text(path) as fh:
        expr = pd.read_csv(fh, sep="\t", comment="!", index_col=0, dtype={"ID_REF": "int64"})
    expr = expr.astype(dtype)
    expr.index.name = "ID_REF"
    expr.columns.name = "sample_id"
    if expr.shape[1] != config.N_SAMPLES:
        raise ValueError(f"Expected {config.N_SAMPLES} samples, got {expr.shape[1]}")
    if not 40_000 <= expr.shape[0] <= 46_000:
        raise ValueError(f"Implausible probe count: {expr.shape[0]}")
    if expr.index.duplicated().any():
        raise ValueError("Duplicate probe IDs in series matrix")
    if with_characteristics:
        return expr, load_series_characteristics(path)
    return expr


def collapse_probes_to_genes(
    expr: pd.DataFrame,
    annot: pd.DataFrame | None = None,
    method: Literal["mean", "median", "max_mean"] = "mean",
) -> pd.DataFrame:
    """Aggregate a probes × samples matrix to genes × samples.

    Probes without a ``GENE_SYMBOL`` (controls, unannotated) are dropped.

    Parameters
    ----------
    expr : probes × samples, index = probe ID (int) as in the series matrix.
    annot : output of :func:`load_probe_annotation` (loaded if None).
    method : ``"mean"`` / ``"median"`` across probes, or ``"max_mean"`` = keep the
        single probe with the highest mean expression for each gene.

    Returns
    -------
    genes × samples DataFrame (index ``gene``), sorted by gene symbol.
    """
    annot = load_probe_annotation() if annot is None else annot
    symbols = annot["GENE_SYMBOL"].reindex(expr.index)
    keep = symbols.notna()
    sub = expr.loc[keep]
    sym = symbols[keep].rename("gene")
    if method in ("mean", "median"):
        out = sub.groupby(sym).agg(method)
    elif method == "max_mean":
        best = sub.mean(axis=1).groupby(sym).idxmax()
        out = sub.loc[best.to_numpy()]
        out.index = pd.Index(best.index, name="gene")
    else:
        raise ValueError("method must be 'mean', 'median' or 'max_mean'")
    out.index.name = "gene"
    return out.sort_index()


# ---------------------------------------------------------------------------------------
# SOFT file (committed provenance)
# ---------------------------------------------------------------------------------------
def load_soft_samples(path: Path | str | None = None) -> pd.DataFrame:
    """Parse the committed GEO SOFT sample headers (``data/raw/GSE39445.txt``).

    Returns one row per ``^SAMPLE`` with ``title``, ``source_name`` and every
    ``Sample_characteristics_ch1`` key as a column (missing keys NaN).
    """
    path = Path(path) if path is not None else config.SOFT_SAMPLES_TXT
    rows: list[dict[str, str]] = []
    cur: dict[str, str] | None = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("^SAMPLE"):
                cur = {"sample_id": line.split("=", 1)[1].strip()}
                rows.append(cur)
            elif cur is not None and line.startswith("!Sample_"):
                key, _, val = line.partition(" = ")
                key = key[len("!Sample_") :]
                if key == "characteristics_ch1" and ": " in val:
                    k, v = val.split(": ", 1)
                    cur[k.strip()] = v.strip()
                elif key in ("title", "source_name_ch1"):
                    cur["source_name" if key == "source_name_ch1" else key] = val.strip()
    return pd.DataFrame(rows).set_index("sample_id")


# ---------------------------------------------------------------------------------------
# Default analysis set
# ---------------------------------------------------------------------------------------
def analysis_samples(
    condition: str = config.MAIN_CONDITION,
    min_timepoints: int = 6,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Default sample set for Parts 3–5 (**provisional**; Part 2 may refine QC).

    Keeps samples from ``condition`` for subjects with at least ``min_timepoints``
    samples in that condition (6 of 10 timepoints is enough to fit a 3-parameter
    cosinor with spare degrees of freedom across a ~27 h span).

    Documented counts (asserted in tests):

    * ``"Sleep Extension"``, 6 → 24 subjects (11 × 4/4, 13 × 5/5), 221 samples
      (AF0079 and AF0091 have no SE samples).
    * ``"Sleep Restriction"``, 6 → 24 subjects (11 × 4/4, 13 × 5/5), 217 samples
      (AF0186 and AF0295 have no SR samples).

    Returns
    -------
    Metadata rows (same columns as :func:`load_metadata`).
    """
    if condition not in config.CONDITIONS.values():
        raise ValueError(f"condition must be one of {list(config.CONDITIONS.values())}")
    meta = load_metadata() if metadata is None else metadata
    sub = meta[meta["sleep_condition"] == condition]
    counts = sub.groupby("subject")["sample_id"].transform("size")
    return sub[counts >= min_timepoints].reset_index(drop=True)
