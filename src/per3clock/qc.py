"""Preprocessing and quality control (Part 2).

Stage 3 must report before/after counts for every step and justify each exclusion.

Decisions (Stage 3, ``notebooks/stages/03_preprocessing_qc.ipynb``)
------------------------------------------------------------------
* **No re-normalisation.** The GEO matrix is already log2 and quantile-normalised
  (every sample has identical quantiles; see :func:`check_log2_normalised`).
* **Expression filter, set from the array's own background:** a gene is kept if it
  is above :data:`DEFAULT_MIN_LOG2` (the 95th percentile of the negative-control
  probes plus 10 %, as in the limma user guide's Agilent example) in at least
  :data:`DEFAULT_MIN_FRAC` of the main-analysis samples. The rule ignores time,
  genotype and condition.
* **Sample outliers are flagged, not dropped.** :func:`apply_qc` reproduces the
  agreed sample sets from :func:`per3clock.data.analysis_samples` and removes
  nothing else unless ``drop_outliers=True`` (a proposal, not the default).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, data, plotting, rhythm

#: Agilent negative-control probes on GPL15331 (``ControlType = -1``): 153 ``(-)3xSLv``
#: and 604 ``DarkCorner`` spots in the series matrix.
NEGATIVE_CONTROL_PREFIXES: tuple[str, ...] = ("(-)3xSLv", "DarkCorner")

#: Margin added to the negative-control quantile: "10 % brighter than background"
#: on the linear scale, i.e. ``log2(1.1)`` on the log2 scale.
BACKGROUND_MARGIN_LOG2: float = float(np.log2(1.1))

#: :func:`background_threshold` on the main-analysis samples (Sleep Extension,
#: 221 samples): 95th percentile of the 757 negative-control probes (2.907) plus
#: ``log2(1.1)`` = **3.044**. Checked against the real data by
#: ``tests/test_qc.py::test_default_threshold_matches_background`` (network).
DEFAULT_MIN_LOG2: float = 3.044

#: Minimum fraction of samples above the threshold: 0.2 ≈ 2 of a subject's 10
#: three-hourly samples, so a gene that is expressed only around its daily peak is
#: kept. It is far below the smaller genotype's share of samples (~45 %), so a
#: gene expressed in only one genotype cannot be filtered out.
DEFAULT_MIN_FRAC: float = 0.2

#: Sex-marker genes used for the within-subject consistency check.
SEX_MARKERS: tuple[str, str] = ("XIST", "RPS4Y1")


# ---------------------------------------------------------------------------------------
# Normalisation and background
# ---------------------------------------------------------------------------------------
def check_log2_normalised(expr: pd.DataFrame, tol: float = 0.01) -> dict:
    """Check whether a features × samples matrix is already log2 and quantile-normalised.

    Run this **before** any transformation: re-logging log2 data or re-normalising a
    quantile-normalised matrix would distort it.

    Parameters
    ----------
    expr : features × samples.
    tol : maximum spread (max − min across samples) of the 5th–95th percentiles for
        the matrix to count as quantile-normalised.

    Returns
    -------
    dict with
        ``min``, ``median``, ``max``   overall values
        ``n_missing``                  number of NaN cells
        ``quantile_spread``            max − min across samples of each sample's
                                       5th, 25th, 50th, 75th and 95th percentile
                                       (a Series)
        ``looks_log2``                 values within [0, 25] (a linear Agilent signal
                                       spans ~10^0–10^6)
        ``quantile_normalised``        every ``quantile_spread`` ≤ ``tol``
    """
    vals = expr.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    q = expr.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    spread = (q.max(axis=1) - q.min(axis=1)).rename("quantile_spread")
    return {
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
        "n_missing": int(np.isnan(vals).sum()),
        "quantile_spread": spread,
        "looks_log2": bool(finite.min() >= 0 and finite.max() <= 25),
        "quantile_normalised": bool((spread <= tol).all()),
    }


def negative_control_ids(annot: pd.DataFrame | None = None) -> pd.Index:
    """Probe IDs of the Agilent negative controls (:data:`NEGATIVE_CONTROL_PREFIXES`).

    Parameters
    ----------
    annot : :func:`per3clock.data.load_probe_annotation` (loaded if None).
    """
    annot = data.load_probe_annotation() if annot is None else annot
    names = annot["PROBE_NAME"].fillna("")
    is_neg = names.str.startswith(NEGATIVE_CONTROL_PREFIXES)
    return annot.index[is_neg.to_numpy()]


def negative_control_p95(
    expr: pd.DataFrame, annot: pd.DataFrame | None = None, quantile: float = 0.95
) -> pd.Series:
    """Per-sample ``quantile`` of the negative-control probe signal (a background metric).

    Parameters
    ----------
    expr : **probe-level** matrix (series matrix, probes × samples).
    annot : probe annotation (loaded if None).

    Returns
    -------
    Series indexed by ``sample_id``, named ``neg_ctrl_p95``.
    """
    ids = negative_control_ids(annot).intersection(expr.index)
    if ids.empty:
        raise ValueError("expr contains no negative-control probes (pass the probe matrix)")
    return expr.loc[ids].quantile(quantile).rename("neg_ctrl_p95")


def background_threshold(
    expr: pd.DataFrame,
    annot: pd.DataFrame | None = None,
    quantile: float = 0.95,
    margin_log2: float = BACKGROUND_MARGIN_LOG2,
) -> float:
    """Expression threshold derived from the negative-control probes.

    ``threshold = Q(quantile) of all negative-control values in expr + margin_log2``,
    pooled over every sample in ``expr``. Pooling is appropriate because the matrix
    is quantile-normalised (one common intensity scale). This is the limma user
    guide's Agilent recipe ("at least 10 % brighter than the 95th percentile of the
    negative controls"), done on the log2 scale.

    Parameters
    ----------
    expr : **probe-level** matrix restricted to the samples the filter is for
        (main analysis: ``analysis_samples()``).
    annot : probe annotation (loaded if None).

    Returns
    -------
    Threshold in log2 units. 3.044 for the main-analysis samples.
    """
    ids = negative_control_ids(annot).intersection(expr.index)
    if ids.empty:
        raise ValueError("expr contains no negative-control probes (pass the probe matrix)")
    vals = expr.loc[ids].to_numpy(dtype=float).ravel()
    return float(np.nanquantile(vals, quantile) + margin_log2)


def filter_low_expression(
    expr: pd.DataFrame, min_log2: float = DEFAULT_MIN_LOG2, min_frac: float = DEFAULT_MIN_FRAC
) -> pd.DataFrame:
    """Keep probes/genes expressed above ``min_log2`` in at least ``min_frac`` of samples.

    Parameters
    ----------
    expr : features × samples (log2).
    min_log2 : expression threshold. Default :data:`DEFAULT_MIN_LOG2` (array
        background from :func:`background_threshold`).
    min_frac : fraction of samples (columns of ``expr``) that must exceed it.
        Default :data:`DEFAULT_MIN_FRAC`.

    Returns
    -------
    Filtered ``expr`` (same columns, subset of rows, original order). The rule uses
    only the expression values, never time, genotype or condition.
    """
    if not 0 <= min_frac <= 1:
        raise ValueError("min_frac must be in [0, 1]")
    frac_above = (expr > min_log2).mean(axis=1)
    return expr.loc[frac_above >= min_frac]


def expressed_genes(
    expr: pd.DataFrame,
    annot: pd.DataFrame | None = None,
    sample_ids=None,
    min_frac: float = DEFAULT_MIN_FRAC,
) -> tuple[pd.DataFrame, float]:
    """The Stage 3 feature pipeline in one call (used by Stage 4 and later parts).

    1. one probe per gene over **all** samples (:func:`per3clock.data.collapse_probes_to_genes`,
       the shared outcome-blind default);
    2. restrict to ``sample_ids`` (default: ``analysis_samples()``);
    3. threshold from the negative controls in those samples
       (:func:`background_threshold`) and :func:`filter_low_expression`.

    Parameters
    ----------
    expr : full probe-level series matrix (all 438 samples).

    Returns
    -------
    ``(genes, threshold)``: expressed genes × ``sample_ids`` and the threshold used.
    """
    annot = data.load_probe_annotation() if annot is None else annot
    if sample_ids is None:
        sample_ids = data.analysis_samples()["sample_id"]
    sample_ids = list(sample_ids)
    threshold = background_threshold(expr[sample_ids], annot)
    genes = data.collapse_probes_to_genes(expr, annot)[sample_ids]
    return filter_low_expression(genes, threshold, min_frac), threshold


def filter_sensitivity(
    expr: pd.DataFrame,
    min_log2_values=(2.5, DEFAULT_MIN_LOG2, 3.5, 4.0, 5.0, 6.0),
    min_frac_values=(0.1, DEFAULT_MIN_FRAC, 0.5),
    genes: tuple[str, ...] = config.CLOCK_GENES,
) -> pd.DataFrame:
    """Features kept, and clock genes lost, across a grid of filter settings.

    Returns
    -------
    DataFrame with ``min_log2, min_frac, kept, removed, clock_genes_lost`` (one row
    per setting; ``clock_genes_lost`` is a comma-separated list or ``""``).
    """
    rows = []
    for t in min_log2_values:
        for f in min_frac_values:
            kept = filter_low_expression(expr, t, f)
            lost = [g for g in genes if g in expr.index and g not in kept.index]
            rows.append({"min_log2": round(float(t), 3), "min_frac": f, "kept": len(kept),
                         "removed": len(expr) - len(kept), "clock_genes_lost": ",".join(lost)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Sample-level QC
# ---------------------------------------------------------------------------------------
def _robust_z(x: pd.Series) -> pd.Series:
    """(x − median) / (1.4826 · MAD); 0 where the MAD is 0, NaN stays NaN."""
    med = x.median()
    mad = 1.4826 * (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return (x * 0).where(x.notna())
    return (x - med) / mad


def _fisher_z(r: pd.Series) -> pd.Series:
    return np.arctanh(r.clip(-1 + 1e-12, 1 - 1e-12))


def sample_qc_table(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    background: pd.Series | None = None,
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    """Per-sample QC metrics.

    Parameters
    ----------
    expr : features × samples (log2), ideally the expression-filtered gene matrix.
    meta : metadata rows for (at least) the samples in ``expr``.
    background : optional per-sample background (:func:`negative_control_p95`),
        reported as ``neg_ctrl_p95`` with ``high_background`` (robust z >
        ``z_thresh``). Informational: it does not set ``outlier``.
    z_thresh : outlier cut-off in robust z units.

    Metrics
    -------
    ``median_log2``, ``iqr_log2``       sample distribution (identical after quantile
                                        normalisation, so a check rather than a filter)
    ``n_missing``                        NaN values in the sample
    ``corr_to_subject_median``          Pearson r with the median profile of the
                                        **same subject's other samples** (leave-one-out;
                                        NaN if the subject has one sample)
    ``corr_to_all_median``              Pearson r with the median of all samples
    ``z_subject``, ``z_all``            robust z of the Fisher-transformed correlations
                                        (median/MAD across samples)
    ``outlier``                         ``z_subject < −z_thresh`` or ``z_all < −z_thresh``
    ``outlier_reason``                  which rule fired (``""`` if none)

    Returns
    -------
    DataFrame indexed by ``sample_id`` with the metrics above plus the metadata
    columns.
    """
    m = meta.drop_duplicates("sample_id").set_index("sample_id")
    missing = [s for s in expr.columns if s not in m.index]
    if missing:
        raise ValueError(f"samples missing from meta: {missing[:5]}")
    m = m.loc[list(expr.columns)]
    x = expr.astype(float)

    out = pd.DataFrame(index=pd.Index(expr.columns, name="sample_id"))
    out["median_log2"] = x.median()
    out["iqr_log2"] = x.quantile(0.75) - x.quantile(0.25)
    out["n_missing"] = x.isna().sum().astype(int)
    out["corr_to_all_median"] = x.corrwith(x.median(axis=1))

    r_sub = pd.Series(np.nan, index=out.index)
    for _, ids in m.groupby("subject").groups.items():
        ids = list(ids)
        if len(ids) < 2:
            continue
        for s in ids:
            ref = x[[i for i in ids if i != s]].median(axis=1)
            r_sub[s] = x[s].corr(ref)
    out["corr_to_subject_median"] = r_sub

    out["z_subject"] = _robust_z(_fisher_z(out["corr_to_subject_median"]))
    out["z_all"] = _robust_z(_fisher_z(out["corr_to_all_median"]))
    low_sub = out["z_subject"] < -z_thresh
    low_all = out["z_all"] < -z_thresh
    out["outlier"] = (low_sub | low_all).astype(bool)
    reason = np.where(low_sub, f"corr to own subject: robust z < -{z_thresh:g}", "")
    reason_all = np.where(low_all, f"corr to all samples: robust z < -{z_thresh:g}", "")
    out["outlier_reason"] = [
        "; ".join(r for r in pair if r) for pair in zip(reason, reason_all, strict=True)
    ]

    if background is not None:
        bg = background.reindex(out.index)
        out["neg_ctrl_p95"] = bg
        out["high_background"] = (_robust_z(bg) > z_thresh).astype(bool)

    return out.join(m)


def infer_sex(genes: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Sex-marker consistency check (XIST vs RPS4Y1).

    Sex is **not** in the GEO metadata. It is inferred from expression only to check
    that every subject's samples agree (a mismatch would suggest a sample swap).

    Parameters
    ----------
    genes : genes × samples matrix containing ``XIST`` and ``RPS4Y1``. Use the
        unfiltered gene matrix: one of the two markers is near background in every
        sample by design.
    meta : metadata rows for the samples.

    Returns
    -------
    DataFrame indexed by ``sample_id`` with ``subject, per3_genotype, XIST, RPS4Y1,
    sex_score`` (= XIST − RPS4Y1), ``inferred_sex`` (``female`` if the score > 0, else
    ``male``), ``subject_sex`` (majority call for the subject) and ``consistent``.
    """
    missing = [g for g in SEX_MARKERS if g not in genes.index]
    if missing:
        raise ValueError(f"sex markers missing from genes: {missing}")
    m = meta.drop_duplicates("sample_id").set_index("sample_id").loc[list(genes.columns)]
    out = genes.loc[list(SEX_MARKERS)].T.astype(float)
    out.index.name = "sample_id"
    out["sex_score"] = out["XIST"] - out["RPS4Y1"]
    out["inferred_sex"] = np.where(out["sex_score"] > 0, "female", "male")
    out.insert(0, "per3_genotype", m["per3_genotype"])
    out.insert(0, "subject", m["subject"])
    out["subject_sex"] = out.groupby("subject")["inferred_sex"].transform(
        lambda s: s.value_counts().index[0]
    )
    out["consistent"] = out["inferred_sex"] == out["subject_sex"]
    return out


def clock_gene_background_check(
    genes: pd.DataFrame,
    threshold: float = DEFAULT_MIN_LOG2,
    clock_genes: tuple[str, ...] = config.CLOCK_GENES,
) -> pd.DataFrame:
    """Are the clock genes above background? (biology that must hold in blood)

    Parameters
    ----------
    genes : genes × samples (unfiltered, so the percentile is over the whole array).
    threshold : background threshold (:func:`background_threshold`).

    Returns
    -------
    DataFrame indexed by gene with ``mean_log2, min_log2, frac_above_threshold,
    array_percentile`` (percentile of the gene's mean among all gene means) and
    ``passes_filter`` (``frac_above_threshold`` ≥ :data:`DEFAULT_MIN_FRAC`).
    """
    means = genes.mean(axis=1)
    sub = genes.loc[[g for g in clock_genes if g in genes.index]]
    out = pd.DataFrame(
        {
            "mean_log2": sub.mean(axis=1),
            "min_log2": sub.min(axis=1),
            "frac_above_threshold": (sub > threshold).mean(axis=1),
            "array_percentile": [100 * float((means < means[g]).mean()) for g in sub.index],
        }
    )
    out["passes_filter"] = out["frac_above_threshold"] >= DEFAULT_MIN_FRAC
    out.index.name = "gene"
    return out


def pooled_clock_timing(
    clock_long: pd.DataFrame,
    genes: tuple[str, ...] = config.CLOCK_GENES,
    time_col: str = "clock_time_h",
    period: float = config.PERIOD_H,
) -> pd.DataFrame:
    """Sanity check that the clock genes peak in the expected order (genotypes pooled).

    Expression is centred within each subject (removing between-subject level
    differences) and one cosinor (:func:`per3clock.rhythm.cosinor_fit`) is fitted per
    gene to all samples together. Genotype is **not** used. This is a descriptive
    check of the data, not the per-subject analysis (Part 3), and no p-value is
    returned because the pooled samples are repeated measures, not independent.

    Parameters
    ----------
    clock_long : :func:`per3clock.data.load_clock_genes` rows for **one** condition.
    time_col : ``"clock_time_h"`` or ``"circadian_phase_h"``.

    Returns
    -------
    DataFrame indexed by gene with ``acrophase_h, amplitude, r2, n, n_subjects``.
    """
    if clock_long["sleep_condition"].nunique() > 1:
        raise ValueError("pass a single sleep_condition; do not pool SE and SR")
    rows = {}
    for g in genes:
        d = clock_long[clock_long["gene"] == g]
        if d.empty:
            continue
        y = d["expression"] - d.groupby("subject")["expression"].transform("mean")
        fit = rhythm.cosinor_fit(d[time_col], y, period=period)
        rows[g] = {"acrophase_h": fit["acrophase_h"], "amplitude": fit["amplitude"],
                   "r2": fit["r2"], "n": fit["n"], "n_subjects": d["subject"].nunique()}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "gene"
    return out


# ---------------------------------------------------------------------------------------
# Sample sets and counts
# ---------------------------------------------------------------------------------------
def apply_qc(
    meta: pd.DataFrame,
    expr: pd.DataFrame | None = None,
    *,
    condition: str = config.MAIN_CONDITION,
    min_timepoints: int = 6,
    strict: bool = False,
    for_phase_analysis: bool = False,
    drop_outliers: bool = False,
    z_thresh: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the agreed sample-level QC rules.

    The sample set is exactly :func:`per3clock.data.analysis_samples` with the same
    arguments (the single source of truth, issue #10):

    * samples from the other condition are excluded (Sleep Restriction is a
      sensitivity analysis only);
    * subjects with fewer than ``min_timepoints`` samples are excluded;
    * the authors' rule only with ``strict=True`` (sensitivity);
    * AF0182 SE (inconsistent ``circadian_phase_h`` labels at T4/T5) only with
      ``for_phase_analysis=True``. Its clock times are real, so it stays in
      clock-time analyses.

    Sample outliers from :func:`sample_qc_table` are **not** removed unless
    ``drop_outliers=True`` (needs ``expr``). That option exists to show the effect
    of a proposed exclusion; it is not an agreed default.

    Returns
    -------
    ``(meta_kept, exclusions)`` where ``exclusions`` has columns
    ``sample_id, subject, sleep_condition, reason`` (one row per excluded sample) and
    is saved by stage 3 as ``results/tables/qc_exclusions.csv``.
    """
    kept, subj_excl = data.analysis_samples(
        condition,
        min_timepoints,
        strict=strict,
        for_phase_analysis=for_phase_analysis,
        metadata=meta,
        return_exclusions=True,
    )
    subj_reason = subj_excl.set_index("subject")["reason"]
    dropped = meta[~meta["sample_id"].isin(kept["sample_id"])]
    reasons = np.where(
        dropped["sleep_condition"] != condition,
        dropped["sleep_condition"] + ": not the main condition (sensitivity analysis only)",
        dropped["subject"].map(subj_reason).fillna("excluded by analysis_samples()"),
    )
    excl = dropped[["sample_id", "subject", "sleep_condition"]].assign(reason=reasons)

    if drop_outliers:
        if expr is None:
            raise ValueError("drop_outliers=True needs expr")
        qct = sample_qc_table(expr[[s for s in kept["sample_id"] if s in expr.columns]], kept,
                              z_thresh=z_thresh)
        bad = qct[qct["outlier"]]
        extra = kept[kept["sample_id"].isin(bad.index)][["sample_id", "subject",
                                                         "sleep_condition"]]
        extra = extra.assign(reason="QC outlier: " + extra["sample_id"].map(bad["outlier_reason"]))
        excl = pd.concat([excl, extra], ignore_index=True)
        kept = kept[~kept["sample_id"].isin(bad.index)]

    excl = excl.sort_values(["sleep_condition", "subject", "sample_id"]).reset_index(drop=True)
    return kept.reset_index(drop=True), excl


def qc_counts(steps: list[tuple]) -> pd.DataFrame:
    """Build the before/after table from ``[(step_name, meta_after_step[, n_features]), ...]``.

    Parameters
    ----------
    steps : list of ``(step_name, meta)`` or ``(step_name, meta, n_features)``, where
        ``meta`` holds the samples remaining after the step and ``n_features`` is the
        number of probes/genes remaining (``NaN`` if not given).

    Returns
    -------
    DataFrame with columns ``step, samples, subjects, subjects_4_4, subjects_5_5,
    samples_SE, samples_SR, features``, one row per step, saved as
    ``results/tables/qc_counts.csv``.
    """
    rows = []
    for step in steps:
        if len(step) not in (2, 3):
            raise ValueError("each step is (name, meta) or (name, meta, n_features)")
        name, m = step[0], step[1]
        n_feat = step[2] if len(step) == 3 else np.nan
        subj_gt = m.drop_duplicates("subject")["per3_genotype"].value_counts()
        cond = m["condition_code"].value_counts()
        rows.append({
            "step": name,
            "samples": len(m),
            "subjects": m["subject"].nunique(),
            "subjects_4_4": int(subj_gt.get("4/4", 0)),
            "subjects_5_5": int(subj_gt.get("5/5", 0)),
            "samples_SE": int(cond.get("SE", 0)),
            "samples_SR": int(cond.get("SR", 0)),
            "features": n_feat,
        })
    out = pd.DataFrame(rows)
    out["features"] = out["features"].astype("Int64")
    return out


# ---------------------------------------------------------------------------------------
# QC figures
# ---------------------------------------------------------------------------------------
def plot_expression_qc(
    genes: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    threshold: float = DEFAULT_MIN_LOG2,
    negative_controls: pd.DataFrame | None = None,
    highlight: tuple[str, ...] = config.GENES_PRIMARY,
) -> plt.Figure:
    """Per-sample distributions and the expression density used to set the filter.

    (a) one box per sample (ordered by genotype, subject, timepoint): identical
    boxes confirm the quantile normalisation. (b) density of per-gene mean
    expression, with the negative-control signal, the threshold and the
    ``highlight`` genes marked. Name: ``qc_expression_density``.
    """
    m = meta.set_index("sample_id").loc[list(genes.columns)]
    order = m.sort_values(["per3_genotype", "subject", "timepoint_n"]).index
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1.1]}, constrained_layout=True
    )

    q = genes[order].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    xs = np.arange(len(order))
    colors = [plotting.GENOTYPE_COLORS.get(m.at[s, "per3_genotype"], "grey") for s in order]
    ax1.vlines(xs, q.loc[0.05], q.loc[0.95], colors=colors, lw=0.6, alpha=0.6)
    ax1.vlines(xs, q.loc[0.25], q.loc[0.75], colors=colors, lw=2.2)
    ax1.scatter(xs, q.loc[0.5], s=3, color="black", zorder=3, label="median")
    ax1.set_xlim(-1, len(order))
    ax1.set_xticks([])
    ax1.set_xlabel(f"{len(order)} samples, ordered by genotype, subject and timepoint")
    ax1.set_ylabel("log2 expression")
    ax1.set_title("(a) Per-sample distributions (bars: 25-75 % and 5-95 % of genes)", loc="left")
    handles = [plt.Line2D([], [], color=c, lw=2.2, label=f"PER3 {g}")
               for g, c in plotting.GENOTYPE_COLORS.items()]
    ax1.legend(handles=handles, loc="lower right", ncol=2, frameon=True, facecolor="white",
               framealpha=0.95, edgecolor="0.8")

    gmean = genes.mean(axis=1)
    bins = np.linspace(0, max(16.0, float(gmean.max()) + 0.5), 121)
    ax2.hist(gmean, bins=bins, density=True, color="0.6", alpha=0.8,
             label=f"gene means ({len(gmean):,} genes)")
    if negative_controls is not None:
        ax2.hist(negative_controls.to_numpy().ravel(), bins=bins, density=True, histtype="step",
                 color="black", lw=1.2, label="negative-control probes (all values)")
    ax2.axvline(threshold, color="#CC0000", ls="--", lw=1.4,
                label=f"threshold log2 {threshold:.2f}")
    ymax = ax2.get_ylim()[1]
    shown = sorted((h for h in highlight if h in gmean.index), key=lambda h: gmean[h])
    groups: list[list[str]] = []  # genes with near-identical means share one label
    for g in shown:
        if groups and gmean[g] - gmean[groups[-1][-1]] < 0.6:
            groups[-1].append(g)
        else:
            groups.append([g])
    for i, grp in enumerate(groups):
        x0 = float(np.mean([gmean[g] for g in grp]))
        label = "\n".join(f"{g} ({gmean[g]:.2f})" for g in grp)
        ax2.annotate(label, (x0, 0), xytext=(x0, ymax * (0.5 + 0.15 * i)), ha="center",
                     va="bottom", fontsize=9,
                     arrowprops={"arrowstyle": "-", "color": "0.3", "lw": 0.8})
    ax2.set_xlabel("log2 expression (mean over samples)")
    ax2.set_ylabel("density")
    ax2.set_title("(b) Expression density and background threshold", loc="left")
    ax2.legend(loc="upper right")
    return fig


def plot_sample_qc(qc_table: pd.DataFrame, z_thresh: float = 3.0) -> plt.Figure:
    """Sample correlation QC: each sample's correlation with its subject and with all.

    Outliers (``qc_table["outlier"]``) are ringed and labelled. Name:
    ``qc_sample_correlation``.
    """
    t = qc_table.sort_values(["per3_genotype", "subject", "timepoint_n"])
    subjects = list(dict.fromkeys(t["subject"]))
    pos = {s: i for i, s in enumerate(subjects)}
    x = t["subject"].map(pos).to_numpy(dtype=float)
    x = x + (t["timepoint_n"].to_numpy() - 5.5) * 0.04
    colors = t["per3_genotype"].map(plotting.GENOTYPE_COLORS).fillna("grey")

    panels = [("corr_to_subject_median", "z_subject",
               "(a) r with the median of the subject's other samples"),
              ("corr_to_all_median", "z_all", "(b) r with the median of all samples")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    for ax, (col, zcol, title) in zip(axes, panels, strict=True):
        ax.scatter(x, t[col], c=colors, s=14, alpha=0.85)
        z = np.arctanh(t[col].clip(upper=1 - 1e-12))
        med = np.nanmedian(z)
        mad = 1.4826 * np.nanmedian(np.abs(z - med))
        ax.axhline(np.tanh(med - z_thresh * mad), color="#CC0000", ls="--", lw=1,
                   label=f"robust z = -{z_thresh:g}")
        out = t[t[zcol] < -z_thresh]
        xo = x[t.index.get_indexer(out.index)]
        ax.scatter(xo, out[col], s=70, facecolors="none", edgecolors="black", lw=1.2)
        for k, (xi, (_, row)) in enumerate(zip(xo, out.iterrows(), strict=True)):
            ax.annotate(f"{row['subject']} {row['timepoint']}", (xi, row[col]),
                        xytext=(6, -12 if k % 2 == 0 else 6), textcoords="offset points",
                        fontsize=8)
        ax.set_ylabel("Pearson r")
        ax.set_title(title, loc="left")
        ax.legend(loc="lower right")
    axes[-1].set_xticks(range(len(subjects)))
    axes[-1].set_xticklabels(subjects, rotation=90, fontsize=8)
    for lab in axes[-1].get_xticklabels():
        gt = t.loc[t["subject"] == lab.get_text(), "per3_genotype"].iloc[0]
        lab.set_color(plotting.GENOTYPE_COLORS.get(gt, "black"))
    axes[-1].set_xlabel("subject (colour = PER3 genotype; points within a subject = T1…T10)")
    return fig


def plot_biology_checks(
    clock_long: pd.DataFrame,
    timing: pd.DataFrame,
    sex: pd.DataFrame,
    *,
    threshold: float = DEFAULT_MIN_LOG2,
    array_median: float | None = None,
) -> plt.Figure:
    """Three checks against biology that must hold. Name: ``qc_biology_checks``.

    (a) clock genes are expressed above background (box per gene, threshold line);
    (b) pooled, subject-centred acrophases (:func:`pooled_clock_timing`) on a 24-h
    clock: the PER/NR1D repressor arm should peak roughly opposite ARNTL;
    (c) XIST vs RPS4Y1 per sample (:func:`infer_sex`), coloured by subject
    consistency.
    """
    genes = [g for g in config.CLOCK_GENES if g in set(clock_long["gene"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.3, 1.1, 1]})

    ax = axes[0]
    vals = [clock_long.loc[clock_long["gene"] == g, "expression"].to_numpy() for g in genes]
    bp = ax.boxplot(vals, patch_artist=True, widths=0.6, showfliers=True,
                    flierprops={"markersize": 2})
    for patch, g in zip(bp["boxes"], genes, strict=True):
        patch.set_facecolor("#9ecae1" if g in config.GENES_PRIMARY else "0.85")
    ax.set_xticks(range(1, len(genes) + 1))
    ax.set_xticklabels(genes, rotation=45, ha="right")
    ax.axhline(threshold, color="#CC0000", ls="--", lw=1.2, label=f"background {threshold:.2f}")
    if array_median is not None:
        ax.axhline(array_median, color="0.4", ls=":", lw=1.2,
                   label=f"array median {array_median:.2f}")
    ax.set_ylabel("log2 expression")
    ax.set_title("(a) Clock genes vs background", loc="left")
    ax.set_ylim(bottom=min(threshold - 0.6, ax.get_ylim()[0]))
    ax.legend(loc="upper center", fontsize=8)

    ax = axes[1]
    tm = timing.loc[[g for g in genes if g in timing.index]]
    yy = np.arange(len(tm))[::-1]
    ax.scatter(tm["acrophase_h"], yy, s=30 + 400 * tm["amplitude"],
               c=["#08519c" if g in config.GENES_PRIMARY else "0.5" for g in tm.index])
    for y, (_, row) in zip(yy, tm.iterrows(), strict=True):
        ax.annotate(f"r²={row['r2']:.2f}", (row["acrophase_h"], y), xytext=(8, -3),
                    textcoords="offset points", fontsize=7, color="0.3")
    ax.set_yticks(yy)
    ax.set_yticklabels(tm.index)
    ax.set_xlim(0, config.PERIOD_H)
    ax.set_xticks(np.arange(0, config.PERIOD_H + 1, 6))
    ax.set_xticklabels([f"{int(h):02d}:00" for h in np.arange(0, config.PERIOD_H + 1, 6)])
    ax.set_xlabel("pooled acrophase (clock time)")
    ax.set_title("(b) Peak time, genotypes pooled", loc="left")

    ax = axes[2]
    ok = sex["consistent"]
    ax.scatter(sex.loc[ok, "XIST"], sex.loc[ok, "RPS4Y1"], s=10, color="0.35",
               label=f"consistent within subject (n={int(ok.sum())})")
    if (~ok).any():
        ax.scatter(sex.loc[~ok, "XIST"], sex.loc[~ok, "RPS4Y1"], s=40, color="#CC0000",
                   label=f"inconsistent (n={int((~ok).sum())})")
    ax.set_xlabel("XIST (log2)")
    ax.set_ylabel("RPS4Y1 (log2)")
    n_f = sex.drop_duplicates("subject")["subject_sex"].eq("female").sum()
    n_m = sex["subject"].nunique() - n_f
    ax.set_title(f"(c) Sex markers ({n_f} XIST-high, {n_m} RPS4Y1-high subjects)",
                 loc="left", fontsize=10)
    ax.legend(loc="center", fontsize=8)
    return fig
