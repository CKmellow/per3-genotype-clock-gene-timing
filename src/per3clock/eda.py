"""Exploratory data analysis helpers (Part 2). Each plot function returns a matplotlib Figure.

Save every figure with ``per3clock.plotting.save_fig(fig, name)``.

EDA is **descriptive**: genotype groups are summarised (mean ± SE) but never
tested here. Hypothesis tests are Part 3 (``rhythm``, ``circstats``).
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from . import config, data, plotting

#: Axis labels. Phase 0 is *probably* the plasma-melatonin midpoint (data/README.md,
#: Known issues 7): an inference, so the label says so.
TIME_LABELS: dict[str, str] = {
    "clock_time_h": "Clock time (h)",
    "clock_time_unwrapped_h": "Clock time (wake period day 1 → day 2)",
    "circadian_phase_h": "Circadian phase (h, 3-h bins; 0 ≈ melatonin midpoint, inferred)",
    "hours_awake": "Hours awake",
}

#: Bin width (h) for genotype means: the 3-h sampling interval.
BIN_H: float = 3.0

#: Minimum subjects in a genotype × time bin for its mean ± SE to be drawn.
MIN_SUBJECTS_PER_BIN: int = 3

#: Metadata columns treated as continuous (and cyclic) when colouring PCA plots.
_CYCLIC = {"clock_time_h": config.PERIOD_H, "circadian_phase_h": config.PERIOD_H}


# ---------------------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------------------
def unwrap_clock_time(meta: pd.DataFrame, period: float = config.PERIOD_H) -> pd.Series:
    """Clock time as continuous hours from midnight before the wake period.

    Sampling ran from the afternoon of day 1 through the night into day 2, so
    ``clock_time_h`` wraps at midnight. For each subject × condition this estimates
    the clock time of waking, ``w`` (circular mean of ``clock_time_h − hours_awake``),
    then puts each sample on the day closest to ``w + hours_awake``. Values run from
    about 14 h (14:00 day 1) to about 45 h (21:00 day 2). Robust to missing
    timepoints because it does not depend on sample order.

    Returns
    -------
    Series aligned with ``meta.index``, named ``clock_time_unwrapped_h``.
    """
    w_rad = 2 * np.pi * ((meta["clock_time_h"] - meta["hours_awake"]) % period) / period
    keys = [meta["subject"], meta["sleep_condition"]]
    s = np.sin(w_rad).groupby(keys).transform("mean")
    c = np.cos(w_rad).groupby(keys).transform("mean")
    wake = (np.arctan2(s, c) * period / (2 * np.pi)) % period
    expected = wake + meta["hours_awake"]
    day = np.round((expected - meta["clock_time_h"]) / period)
    return (meta["clock_time_h"] + day * period).rename("clock_time_unwrapped_h")


def _circ_mean_h(x: pd.Series, period: float = config.PERIOD_H) -> float:
    a = 2 * np.pi * (x % period) / period
    return float((np.arctan2(np.sin(a).mean(), np.cos(a).mean()) * period / (2 * np.pi)) % period)


def subject_timing_summary(meta: pd.DataFrame, period: float = config.PERIOD_H) -> pd.DataFrame:
    """Per-subject schedule and melatonin timing (descriptive, one condition).

    * ``wake_clock_h``: implied clock time of waking, circular mean of
      ``clock_time_h − hours_awake``.
    * ``phase0_clock_h``: implied clock time of circadian phase 0, circular mean of
      ``clock_time_h − circadian_phase_h`` (a proxy for melatonin timing; phase 0 is
      probably the melatonin midpoint, inferred). NaN for series in
      ``data.PHASE_LABEL_EXCLUSIONS`` (AF0182 SE).

    Returns
    -------
    DataFrame indexed by subject with ``per3_genotype, sleep_condition, n_samples,
    wake_clock_h, phase0_clock_h``.
    """
    if meta["sleep_condition"].nunique() > 1:
        raise ValueError("pass a single sleep_condition")
    rows = {}
    for subj, g in meta.groupby("subject"):
        cond = g["sleep_condition"].iloc[0]
        bad = (subj, cond) in data.PHASE_LABEL_EXCLUSIONS
        rows[subj] = {
            "per3_genotype": g["per3_genotype"].iloc[0],
            "sleep_condition": cond,
            "n_samples": len(g),
            "wake_clock_h": _circ_mean_h(g["clock_time_h"] - g["hours_awake"], period),
            "phase0_clock_h": np.nan if bad else _circ_mean_h(
                g["clock_time_h"] - g["circadian_phase_h"], period),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "subject"
    return out


def _with_unwrapped(clock_long: pd.DataFrame) -> pd.DataFrame:
    if "clock_time_unwrapped_h" in clock_long:
        return clock_long
    return clock_long.assign(clock_time_unwrapped_h=unwrap_clock_time(clock_long))


def _drop_phase_label_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Drop series listed in ``data.PHASE_LABEL_EXCLUSIONS`` (AF0182 SE)."""
    bad = pd.Series(list(zip(df["subject"], df["sleep_condition"], strict=True)),
                    index=df.index).isin(set(data.PHASE_LABEL_EXCLUSIONS))
    return df[~bad]


def _hhmm_ticks(ax, lo: float, hi: float, step: float = 6.0) -> None:
    ticks = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t % 24):02d}:00" for t in ticks])


# ---------------------------------------------------------------------------------------
# Clock-gene profiles
# ---------------------------------------------------------------------------------------
def genotype_profile_table(
    clock_long: pd.DataFrame,
    genes: tuple[str, ...] = config.GENES_PRIMARY,
    time_col: str = "clock_time_h",
    *,
    center: bool = False,
    bin_h: float = BIN_H,
) -> pd.DataFrame:
    """Genotype mean ± SE of expression per time bin (descriptive, no tests).

    Subjects are the unit: each subject's samples in a bin are averaged first, then
    the mean and SE (SD / √n) are taken across subjects.

    Parameters
    ----------
    clock_long : :func:`per3clock.data.load_clock_genes` rows (one condition).
    time_col : ``"clock_time_h"`` (binned on the unwrapped clock, since start times
        differ by up to 4 h between subjects), ``"circadian_phase_h"`` (already in
        3-h bins; series in ``data.PHASE_LABEL_EXCLUSIONS`` are dropped) or
        ``"hours_awake"``.
    center : subtract each subject's mean (per gene) first, removing level
        differences between people so the curves show timing.
    bin_h : bin width in hours.

    Returns
    -------
    DataFrame with ``gene, per3_genotype, time_bin_h, mean, se, n_subjects``.
    ``time_bin_h`` is the bin centre (unwrapped hours for clock time).
    """
    df = _prepare_profiles(clock_long, genes, time_col, center)
    t = df["_t"]
    df = df.assign(time_bin_h=np.floor(t / bin_h) * bin_h + bin_h / 2
                   if time_col != "circadian_phase_h" else t)
    per_subject = df.groupby(["gene", "per3_genotype", "time_bin_h", "subject"])["_y"].mean()
    g = per_subject.groupby(["gene", "per3_genotype", "time_bin_h"])
    out = pd.DataFrame({"mean": g.mean(), "se": g.sem(), "n_subjects": g.size()}).reset_index()
    return out


def _prepare_profiles(clock_long, genes, time_col, center) -> pd.DataFrame:
    df = clock_long[clock_long["gene"].isin(genes)].copy()
    if time_col == "circadian_phase_h":
        df = _drop_phase_label_issues(df)
    if time_col == "clock_time_h":
        df = _with_unwrapped(df)
        df["_t"] = df["clock_time_unwrapped_h"]
    else:
        df["_t"] = df[time_col].astype(float)
    df["_y"] = df["expression"]
    if center:
        keys = ["subject", "sleep_condition", "gene"]
        df["_y"] = df["_y"] - df.groupby(keys)["expression"].transform("mean")
    return df


def plot_clock_gene_profiles(
    clock_long: pd.DataFrame,
    genes: tuple[str, ...] = config.GENES_PRIMARY,
    time_col: str | Sequence[str] = "clock_time_h",
    *,
    condition: str = config.MAIN_CONDITION,
    center: bool = False,
    show_subjects: bool = True,
    bin_h: float = BIN_H,
) -> plt.Figure:
    """Expression vs time for each gene (one panel per gene), coloured by genotype.

    Shows individual subjects (thin lines) and the genotype mean ± SE (thick line
    and band, computed by :func:`genotype_profile_table`; bins with fewer than
    :data:`MIN_SUBJECTS_PER_BIN` subjects are not drawn). Suggested name:
    ``eda_clock_gene_profiles``.

    Parameters
    ----------
    clock_long : :func:`per3clock.data.load_clock_genes`, ideally restricted to
        ``analysis_samples()``. Only rows from ``condition`` are plotted, so the
        two sleep conditions are never mixed.
    genes : one column per gene.
    time_col : one time axis, or several (one row each), e.g.
        ``("clock_time_h", "circadian_phase_h")``. Rows against
        ``circadian_phase_h`` drop the series in ``data.PHASE_LABEL_EXCLUSIONS``
        (AF0182 SE), matching ``analysis_samples(for_phase_analysis=True)``.
    center : plot subject-centred expression (see :func:`genotype_profile_table`).
    """
    time_cols = [time_col] if isinstance(time_col, str) else list(time_col)
    df = clock_long[clock_long["sleep_condition"] == condition]
    if df.empty:
        raise ValueError(f"no rows for condition {condition!r}")
    nrow, ncol = len(time_cols), len(genes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol + 0.4, 3.1 * nrow + 0.6),
                             squeeze=False, constrained_layout=True)
    n_by_gt = {}
    for r, tc in enumerate(time_cols):
        prep = _prepare_profiles(df, genes, tc, center)
        prof = genotype_profile_table(df, genes, tc, center=center, bin_h=bin_h)
        n_by_gt[tc] = prep.drop_duplicates("subject")["per3_genotype"].value_counts()
        for c, gene in enumerate(genes):
            ax = axes[r, c]
            d = prep[prep["gene"] == gene]
            for gt, color in plotting.GENOTYPE_COLORS.items():
                dg = d[d["per3_genotype"] == gt]
                if show_subjects:
                    for _, ds in dg.groupby(["subject"]):
                        ds = ds.sort_values("_t")
                        ax.plot(ds["_t"], ds["_y"], color=color, lw=0.6, alpha=0.25)
                p = prof[(prof["gene"] == gene) & (prof["per3_genotype"] == gt)
                         & (prof["n_subjects"] >= MIN_SUBJECTS_PER_BIN)].sort_values("time_bin_h")
                ax.fill_between(p["time_bin_h"], p["mean"] - p["se"], p["mean"] + p["se"],
                                color=color, alpha=0.25, lw=0)
                ax.plot(p["time_bin_h"], p["mean"], color=color, lw=2.2, marker="o",
                        ms=3.5)
            if r == 0:
                role = "" if gene in config.GENES_PRIMARY else " (secondary)"
                ax.set_title(f"{gene}{role}", fontweight="bold")
            # full axis label once per row (centre column) so long labels never collide
            label = TIME_LABELS.get(tc, tc) if c == ncol // 2 else ""
            if tc == "clock_time_h":
                lo, hi = float(d["_t"].min()), float(d["_t"].max())
                _hhmm_ticks(ax, lo, hi)
                label = TIME_LABELS["clock_time_unwrapped_h"] if c == ncol // 2 else ""
            elif tc == "circadian_phase_h":
                ax.set_xticks(np.arange(-15, 16, 6))
            ax.set_xlabel(label)
            if c == 0:
                ax.set_ylabel("log2 expression" + (" (subject-centred)" if center else ""))
    handles = []
    tc0 = time_cols[0]
    for gt, color in plotting.GENOTYPE_COLORS.items():
        n = int(n_by_gt[tc0].get(gt, 0))
        handles.append(plt.Line2D([], [], color=color, lw=2.2, marker="o", ms=3.5,
                                  label=f"PER3 {gt} mean ± SE (n = {n})"))
    if show_subjects:
        handles.append(plt.Line2D([], [], color="0.5", lw=0.6, alpha=0.6,
                                  label="individual subjects"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=9)
    fig.suptitle(f"Clock-gene expression in blood, {condition}", fontsize=11)
    return fig


# ---------------------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------------------
def pca_samples(
    expr: pd.DataFrame, n_components: int = 5
) -> tuple[pd.DataFrame, pd.Series]:
    """PCA of samples on (filtered, centred) features.

    Each feature is centred across samples (not scaled: all features are on the same
    log2 scale). Uses a full SVD, so the result is deterministic.

    Returns
    -------
    ``(scores, explained)``: ``scores`` indexed by ``sample_id`` with columns
    ``PC1..PCk``; ``explained`` = explained-variance ratio indexed by ``PC1..PCk``.
    """
    x = expr.T.to_numpy(dtype=float)
    if np.isnan(x).any():
        raise ValueError("expr contains NaN; filter or impute first")
    k = min(n_components, *x.shape)
    pca = PCA(n_components=k, svd_solver="full")
    s = pca.fit_transform(x - x.mean(axis=0))
    cols = [f"PC{i + 1}" for i in range(k)]
    scores = pd.DataFrame(s, index=pd.Index(expr.columns, name="sample_id"), columns=cols)
    explained = pd.Series(pca.explained_variance_ratio_, index=cols, name="explained_variance")
    return scores, explained


def pca_factor_r2(
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    factors: tuple[str, ...] = ("subject", "per3_genotype", "timepoint", "circadian_phase_h"),
    explained: pd.Series | None = None,
) -> pd.DataFrame:
    """Share of each PC's variance explained by each metadata factor (η², descriptive).

    ``η² = 1 − SS_within / SS_total`` from grouping the PC scores by the factor
    (treated as categorical; ``circadian_phase_h`` is already in 3-h bins). No
    p-values. Note that subject is nested in genotype, so genotype's η² can never
    exceed subject's, and that with ~24 subjects η²(subject) is ≈ 0.1 by chance.

    Returns
    -------
    DataFrame indexed by PC (plus a final ``weighted`` row, the explained-variance
    weighted mean over the PCs, if ``explained`` is given) with one column per factor
    and ``explained`` (if given).
    """
    m = meta.drop_duplicates("sample_id").set_index("sample_id").loc[scores.index]
    rows = {}
    for pc in scores.columns:
        y = scores[pc]
        tot = float(((y - y.mean()) ** 2).sum())
        row = {}
        for f in factors:
            fitted = y.groupby(m[f].astype(str)).transform("mean")
            row[f] = 1 - float(((y - fitted) ** 2).sum()) / tot if tot > 0 else np.nan
        rows[pc] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "PC"
    if explained is not None:
        ev = explained.reindex(out.index)
        weighted = (out.mul(ev, axis=0).sum() / ev.sum()).rename("weighted")
        out["explained"] = ev
        out = pd.concat([out, weighted.to_frame().T.assign(explained=ev.sum())])
        out.index.name = "PC"
    return out


def _palette_for(values: pd.Series, color_by: str) -> dict:
    if color_by == "per3_genotype":
        return dict(plotting.GENOTYPE_COLORS)
    if color_by == "sleep_condition":
        return dict(plotting.CONDITION_COLORS)
    levels = list(dict.fromkeys(sorted(values.astype(str))))
    if color_by == "timepoint":
        levels = sorted(levels, key=lambda s: int(s.lstrip("T")))
        return dict(zip(levels, sns.color_palette("viridis", len(levels)), strict=True))
    return dict(zip(levels, sns.color_palette("husl", len(levels)), strict=True))


def plot_pca(
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    color_by: str = "subject",
    *,
    explained: pd.Series | None = None,
    pcs: tuple[str, str] = ("PC1", "PC2"),
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Scatter PC1 vs PC2 coloured by a metadata column. Name: ``eda_pca_<color_by>``.

    Categorical columns get a legend (genotype and condition use the shared project
    colours). ``clock_time_h`` and ``circadian_phase_h`` use a cyclic colour map;
    other numeric columns a sequential one. Pass ``explained`` to show the variance
    explained on the axes.
    """
    m = meta.drop_duplicates("sample_id").set_index("sample_id").loc[scores.index]
    fig = ax.figure if ax is not None else plt.figure(figsize=(6.2, 4.8), layout="constrained")
    ax = ax if ax is not None else fig.add_subplot()
    x, y = scores[pcs[0]], scores[pcs[1]]
    v = m[color_by]
    if color_by in _CYCLIC or (pd.api.types.is_numeric_dtype(v) and color_by != "timepoint_n"
                               and v.nunique() > 12):
        kw: dict = {"cmap": "viridis"}
        label = TIME_LABELS.get(color_by, color_by)
        if color_by == "clock_time_h":
            kw = {"cmap": "twilight_shifted", "vmin": 0, "vmax": 24}
        elif color_by == "circadian_phase_h":
            # phase spans -15..+15 h (> one cycle): wrap to [-12, 12) so the cyclic map
            # gives the same colour to the same circadian time
            half = _CYCLIC[color_by] / 2
            v = (v + half) % _CYCLIC[color_by] - half
            kw = {"cmap": "twilight_shifted", "vmin": -half, "vmax": half}
            label = "Circadian phase, wrapped to ±12 h\n(0 ≈ melatonin midpoint, inferred)"
        sc = ax.scatter(x, y, c=v, s=16, alpha=0.9, **kw)
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(label, fontsize=8)
    else:
        pal = _palette_for(v, color_by)
        for lev, col in pal.items():
            sel = v.astype(str) == lev
            if sel.any():
                ax.scatter(x[sel], y[sel], s=16, alpha=0.9, color=col, label=lev)
        many = len(pal) > 8
        ax.legend(title=color_by, fontsize=6.5 if many else 8, title_fontsize=8,
                  ncol=2 if many else 1, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  markerscale=1.2, handletextpad=0.2, columnspacing=0.6)
    for i, pc in enumerate(pcs):
        lab = pc if explained is None else f"{pc} ({100 * explained[pc]:.1f} % of variance)"
        (ax.set_xlabel if i == 0 else ax.set_ylabel)(lab)
    ax.set_title(title or f"PCA, coloured by {color_by}", loc="left")
    return fig


def plot_pca_grid(
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    color_bys: Sequence[str] = ("per3_genotype", "subject", "timepoint", "circadian_phase_h"),
    *,
    explained: pd.Series | None = None,
    pcs: tuple[str, str] = ("PC1", "PC2"),
    suptitle: str | None = None,
) -> plt.Figure:
    """Several :func:`plot_pca` panels side by side (same scores, different colours)."""
    n = len(color_bys)
    ncol = min(n, 2)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 4.6 * nrow), squeeze=False,
                             layout="constrained")
    letters = "abcdefghij"
    for i, (ax, cb) in enumerate(zip(axes.ravel(), color_bys, strict=False)):
        plot_pca(scores, meta, cb, explained=explained, pcs=pcs, ax=ax,
                 title=f"({letters[i]}) {cb}")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    return fig


# ---------------------------------------------------------------------------------------
# Sample correlation
# ---------------------------------------------------------------------------------------
def correlation_summary(expr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Sample-pair correlations grouped by relationship (descriptive).

    Pairs of distinct samples are classed as ``same subject``, ``different subject,
    same genotype`` or ``different genotype``.

    Returns
    -------
    DataFrame indexed by pair type with ``n_pairs, median_r, q25_r, q75_r``.
    """
    m = meta.drop_duplicates("sample_id").set_index("sample_id").loc[list(expr.columns)]
    corr = np.corrcoef(expr.to_numpy(dtype=float).T)
    subj = m["subject"].to_numpy()
    gt = m["per3_genotype"].to_numpy()
    iu = np.triu_indices(len(subj), k=1)
    same_s = subj[iu[0]] == subj[iu[1]]
    same_g = gt[iu[0]] == gt[iu[1]]
    kind = np.where(same_s, "same subject",
                    np.where(same_g, "different subject, same genotype", "different genotype"))
    r = pd.Series(corr[iu], name="r")
    out = r.groupby(kind).agg(n_pairs="size", median_r="median",
                              q25_r=lambda x: x.quantile(0.25), q75_r=lambda x: x.quantile(0.75))
    order = ["same subject", "different subject, same genotype", "different genotype"]
    out = out.reindex([k for k in order if k in out.index])
    out.index.name = "pair_type"
    return out


def plot_sample_correlation(expr: pd.DataFrame, meta: pd.DataFrame) -> plt.Figure:
    """Sample–sample correlation heatmap ordered by subject/condition/timepoint.

    Samples are ordered by genotype, subject, condition and timepoint. Colour bars on
    the left and top mark genotype and subject; thin lines separate subjects.
    Name: ``eda_sample_correlation``.
    """
    m = meta.drop_duplicates("sample_id").set_index("sample_id").loc[list(expr.columns)]
    order = m.sort_values(["per3_genotype", "subject", "condition_code", "timepoint_n"]).index
    mo = m.loc[order]
    corr = np.corrcoef(expr[order].to_numpy(dtype=float).T)
    n = len(order)

    fig = plt.figure(figsize=(8.6, 7.6), layout="constrained")
    gs = fig.add_gridspec(3, 3, width_ratios=[0.035, 0.035, 1], height_ratios=[0.035, 0.035, 1],
                          wspace=0.01, hspace=0.01)
    ax = fig.add_subplot(gs[2, 2])
    off = corr[~np.eye(n, dtype=bool)]
    lo = float(np.quantile(off, 0.01)) if n > 1 else 0.0
    im = ax.imshow(corr, cmap="magma", vmin=lo, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])

    subj = mo["subject"].to_numpy()
    edges = np.flatnonzero(subj[1:] != subj[:-1]) + 0.5
    for e in edges:
        ax.axhline(e, color="white", lw=0.4, alpha=0.6)
        ax.axvline(e, color="white", lw=0.4, alpha=0.6)

    subj_levels = list(dict.fromkeys(subj))
    subj_pal = dict(zip(subj_levels, sns.color_palette("husl", len(subj_levels)), strict=True))
    gt_rgb = np.array([plt.matplotlib.colors.to_rgb(plotting.GENOTYPE_COLORS.get(g, "grey"))
                       for g in mo["per3_genotype"]])
    sb_rgb = np.array([subj_pal[s] for s in subj])
    for pos, (rgb, name) in enumerate([(gt_rgb, "genotype"), (sb_rgb, "subject")]):
        axl = fig.add_subplot(gs[2, pos], sharey=ax)
        axl.imshow(rgb[:, None, :], aspect="auto", interpolation="nearest")
        axl.set_xticks([0])
        axl.set_xticklabels([name], rotation=90, fontsize=8)
        axl.set_yticks([])
        axl.grid(False)
        axt = fig.add_subplot(gs[pos, 2], sharex=ax)
        axt.imshow(rgb[None, :, :], aspect="auto", interpolation="nearest")
        axt.set_yticks([0])
        axt.set_yticklabels([name], fontsize=8)
        axt.set_xticks([])
        axt.grid(False)
        for a in (axl, axt):
            for sp in a.spines.values():
                sp.set_visible(False)
    ax.grid(False)

    # genotype labels at the block centres
    for gt in mo["per3_genotype"].unique():
        idx = np.flatnonzero(mo["per3_genotype"].to_numpy() == gt)
        ax.annotate(f"PER3 {gt}\n({mo.iloc[idx]['subject'].nunique()} subjects)",
                    (idx.mean(), n), xytext=(0, -6), textcoords="offset points",
                    ha="center", va="top", fontsize=9, color=plotting.GENOTYPE_COLORS.get(gt),
                    annotation_clip=False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Pearson r (expressed genes)")
    conds = ", ".join(dict.fromkeys(mo["sleep_condition"]))
    fig.suptitle(f"Sample-sample correlation ({n} samples, {conds}); "
                 "blocks = subjects", fontsize=10)
    return fig
