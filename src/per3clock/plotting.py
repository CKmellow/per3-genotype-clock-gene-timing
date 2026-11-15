"""Shared plotting style and figure-saving helpers.

Use ``set_style()`` once per notebook and ``save_fig(fig, "name")`` for every
figure that should appear in the report, so file names are predictable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def save_fig(
    fig: plt.Figure | None, name: str, *, fmt: str = "png", dpi: float | None = None
) -> Path:
    """Save ``fig`` (or the current figure) to ``results/figures/<name>.<fmt>``.

    Parameters
    ----------
    fig : matplotlib Figure or None
        Figure to save; ``None`` saves ``plt.gcf()``.
    name : str
        File stem, e.g. ``"eda_clock_gene_profiles"``. No directories or extension.
    fmt : str
        File format (default ``"png"``).
    dpi : float, optional
        Resolution; default is the style's ``savefig.dpi`` (200). Use 300 for
        report figures.

    Returns
    -------
    Path of the written file.
    """
    if "/" in name or "\\" in name:
        raise ValueError("name must be a bare file stem, not a path")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = fig if fig is not None else plt.gcf()
    path = config.FIGURES_DIR / f"{name}.{fmt}"
    fig.savefig(path, **({} if dpi is None else {"dpi": dpi}))
    return path


# ---------------------------------------------------------------------------------------
# Acrophase figures (Part 3)
# ---------------------------------------------------------------------------------------
TIME_BASE_LABELS: dict[str, str] = {
    "clock_time": "Clock time",
    "circadian_phase": "Circadian phase (0 ≈ melatonin midpoint, inferred)",
}


def _circ_mean_and_rbar(h: np.ndarray, period: float) -> tuple[float, float]:
    a = 2 * np.pi * np.asarray(h, dtype=float) / period
    c, s = np.cos(a).mean(), np.sin(a).mean()
    return float(np.arctan2(s, c)), float(np.hypot(c, s))


def plot_acrophase_rose(
    fits: pd.DataFrame,
    gene: str,
    time_bases: tuple[str, ...] = ("clock_time", "circadian_phase"),
    *,
    alpha: float = 0.05,
    period: float = config.PERIOD_H,
) -> plt.Figure:
    """Per-subject acrophases of one gene on a 24-h dial, one panel per time base.

    Each dot is one subject's peak time (4/4 on the inner ring, 5/5 on the outer).
    Filled dots: that subject's own cosinor is significant (p < ``alpha``); hollow:
    not, so its peak time is mostly noise. Arrows: genotype circular mean, length =
    mean resultant length (1 = all subjects peak together). Name:
    ``acrophase_rose_<gene>``.
    """
    fig, axes = plt.subplots(1, len(time_bases), subplot_kw={"projection": "polar"},
                             figsize=(4.6 * len(time_bases), 4.9), squeeze=False,
                             layout="constrained")
    d_all = fits[fits["gene"] == gene]
    for ax, tb in zip(axes[0], time_bases, strict=True):
        d = d_all[d_all["time_base"] == tb]
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for gt, r0 in (("4/4", 0.72), ("5/5", 0.92)):
            color = GENOTYPE_COLORS[gt]
            g = d[d["per3_genotype"] == gt]
            theta = 2 * np.pi * g["acrophase_h"].to_numpy() / period
            sig = (g["p"] < alpha).to_numpy()
            ax.scatter(theta[sig], np.full(sig.sum(), r0), s=46, color=color, zorder=3,
                       edgecolor="white", linewidth=0.8)
            ax.scatter(theta[~sig], np.full((~sig).sum(), r0), s=46, facecolor="white",
                       edgecolor=color, linewidth=1.4, zorder=3)
            if len(g):
                mu, rbar = _circ_mean_and_rbar(g["acrophase_h"], period)
                ax.annotate("", xy=(mu, rbar * 0.6), xytext=(0, 0),
                            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.2,
                                        "mutation_scale": 14})
        ticks = np.arange(0, period, 3)
        ax.set_xticks(2 * np.pi * ticks / period)
        if tb == "clock_time":
            ax.set_xticklabels([f"{int(t):02d}:00" for t in ticks], fontsize=8)
        else:
            ax.set_xticklabels([f"{int(t if t <= 12 else t - 24):+d} h" if t else "0"
                                for t in ticks], fontsize=8)
        ax.set_yticks([])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.35)
        n = d.groupby("per3_genotype").size()
        ax.set_title(f"{TIME_BASE_LABELS.get(tb, tb)}\n"
                     f"4/4 n={n.get('4/4', 0)}, 5/5 n={n.get('5/5', 0)}", fontsize=9, pad=12)
    handles = [plt.Line2D([], [], marker="o", ls="", color=GENOTYPE_COLORS[gt], ms=7,
                          label=f"PER3 {gt}") for gt in GENOTYPE_COLORS]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color="0.3", ms=7,
                   label=f"subject rhythm p < {alpha:g}"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                   markeredgecolor="0.3", ms=7, label=f"p ≥ {alpha:g} (peak time unreliable)"),
        plt.Line2D([], [], color="0.3", lw=2, label="genotype mean (length = concentration)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{gene}: peak time of each subject's rhythm (Sleep Extension)", fontsize=11)
    return fig


def plot_shift_forest(table: pd.DataFrame, *, label_cols=("gene", "time_base", "sample_set"),
                      q_col: str = "permutation_q") -> plt.Figure:
    """Forest plot of genotype shifts (5/5 − 4/4) with 95 % bootstrap CIs.

    One row per row of ``table`` (``shift_h, ci_lower_h, ci_upper_h`` and
    ``label_cols``); rows whose ``sample_set`` is ``"Primary"`` are filled, the
    sensitivity analyses hollow. Name: ``acrophase_shift_forest``.
    """
    t = table.reset_index(drop=True)
    y = np.arange(len(t))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(t) + 1.4), layout="constrained")
    ax.axvline(0, color="0.35", lw=1)
    primary = (t["sample_set"] == "Primary").to_numpy() if "sample_set" in t else np.ones(len(t),
                                                                                        bool)
    ax.hlines(y, t["ci_lower_h"], t["ci_upper_h"], color="0.25", lw=1.6)
    ax.scatter(t.loc[primary, "shift_h"], y[primary], s=48, color="0.15", zorder=3,
               label="primary analysis")
    ax.scatter(t.loc[~primary, "shift_h"], y[~primary], s=48, facecolor="white",
               edgecolor="0.15", lw=1.3, zorder=3, label="sensitivity analysis")
    labels = [" · ".join(str(r[c]).replace("_", " ") for c in label_cols if c in t)
              for _, r in t.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    if q_col in t:
        xmax = float(np.nanmax(t["ci_upper_h"]))
        for yi, q in zip(y, t[q_col], strict=True):
            ax.annotate(f"q={q:.2f}", (xmax, yi), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color="0.35", annotation_clip=False)
    ax.set_xlabel("Shift in peak time, 5/5 − 4/4 (h)   ← 5/5 earlier   |   5/5 later →")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor="white", edgecolor="0.85")
    ax.set_title("Genotype difference in peak time, 95 % bootstrap CI", loc="left")
    return fig
