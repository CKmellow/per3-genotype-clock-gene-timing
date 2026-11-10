"""Shared plotting style and figure-saving helpers.

Use ``set_style()`` once per notebook and ``save_fig(fig, "name")`` for every
figure that should appear in the report, so file names are predictable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

# Consistent genotype colours across all parts (colour-blind safe).
GENOTYPE_COLORS: dict[str, str] = {"4/4": "#D55E00", "5/5": "#0072B2"}
CONDITION_COLORS: dict[str, str] = {"Sleep Extension": "#009E73", "Sleep Restriction": "#CC79A7"}


def set_style() -> None:
    """Apply the project matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
        }
    )


def save_fig(fig: plt.Figure | None, name: str, *, fmt: str = "png") -> Path:
    """Save ``fig`` (or the current figure) to ``results/figures/<name>.<fmt>``.

    Parameters
    ----------
    fig : matplotlib Figure or None
        Figure to save; ``None`` saves ``plt.gcf()``.
    name : str
        File stem, e.g. ``"eda_clock_gene_profiles"``. No directories or extension.
    fmt : str
        File format (default ``"png"``).

    Returns
    -------
    Path of the written file.
    """
    if "/" in name or "\\" in name:
        raise ValueError("name must be a bare file stem, not a path")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = fig if fig is not None else plt.gcf()
    path = config.FIGURES_DIR / f"{name}.{fmt}"
    fig.savefig(path)
    return path
