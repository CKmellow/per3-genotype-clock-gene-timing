"""Project-wide paths and constants.

All paths are derived from the repository root so that code works the same way
locally, in CI and in Google Colab. Never hard-code absolute paths elsewhere:
import them from here.

The root can be overridden with the ``PER3CLOCK_ROOT`` environment variable
(useful if the package is installed non-editably).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def _find_root() -> Path:
    env = os.environ.get("PER3CLOCK_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "per3clock").exists():
            return parent
    # Fallback: current working directory (e.g. notebook launched from repo root)
    return Path.cwd()


ROOT: Path = _find_root()

DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
EXTERNAL_DIR: Path = DATA_DIR / "external"  # gitignored downloads
PROCESSED_DIR: Path = DATA_DIR / "processed"

RESULTS_DIR: Path = ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
TABLES_DIR: Path = RESULTS_DIR / "tables"

# --- Raw files committed in data/raw -------------------------------------------------
METADATA_CSV: Path = RAW_DIR / "GSE39445_sample_metadata.csv"
CLOCK_GENES_CSV: Path = RAW_DIR / "GSE39445_clock_genes.csv"
PROBE_ANNOTATION_CSV: Path = RAW_DIR / "GPL15331_probe_annotation.csv"
SOFT_SAMPLES_TXT: Path = RAW_DIR / "GSE39445.txt"

# --- GEO series matrix (downloaded on demand, never committed) ------------------------
GEO_ACCESSION: str = "GSE39445"
PLATFORM: str = "GPL15331"
SERIES_MATRIX_URL: str = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE39nnn/GSE39445/matrix/"
    "GSE39445_series_matrix.txt.gz"
)
SERIES_MATRIX_GZ: Path = EXTERNAL_DIR / "GSE39445_series_matrix.txt.gz"
# Size observed on the NCBI FTP server on 2026-11-11 (Last-Modified 2026-08-14).
SERIES_MATRIX_BYTES: int = 99_904_009
N_SAMPLES: int = 438
N_PROBES: int = 43_758

# --- Biology / analysis constants -------------------------------------------------------
GENES_PRIMARY: tuple[str, ...] = ("PER2", "PER3", "NR1D1")
GENES_SECONDARY: tuple[str, ...] = ("CLOCK", "CRY1", "ARNTL", "PER1", "NR1D2")
CLOCK_GENES: tuple[str, ...] = GENES_PRIMARY + GENES_SECONDARY

GENOTYPES: tuple[str, ...] = ("4/4", "5/5")
CONDITIONS: dict[str, str] = {"SE": "Sleep Extension", "SR": "Sleep Restriction"}
MAIN_CONDITION: str = "Sleep Extension"
SENSITIVITY_CONDITION: str = "Sleep Restriction"

PERIOD_H: float = 24.0
SEED: int = 20261118


def set_seeds(seed: int = SEED) -> None:
    """Seed Python's and NumPy's global RNGs (call at the top of every notebook)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    """Create output/cache directories if they do not exist."""
    for d in (EXTERNAL_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
