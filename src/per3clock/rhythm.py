"""Rhythm analysis: cosinor fits and rhythmicity scans (Part 3).

``cosinor_fit`` is implemented by Part 1 because Parts 2, 4 and 5 depend on it.
**Part 3 owns extending and validating it** (see issue #3). Keep the signature and
the returned keys stable, or agree changes in the issue.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from scipy import stats

from . import config


class CosinorResult(TypedDict):
    mesor: float
    amplitude: float
    acrophase_h: float
    r2: float
    p: float
    n: int


def cosinor_fit(t_hours, y, period: float = config.PERIOD_H) -> CosinorResult:
    """Single-component cosinor fit by ordinary least squares.

    Model: ``y = M + A·cos(2π(t − φ)/T) + ε``, fitted linearly as
    ``y = M + β·cos(ωt) + γ·sin(ωt)`` with ``ω = 2π/T``.

    Parameters
    ----------
    t_hours : array-like
        Sample times in hours (any origin, e.g. clock time 0–24, hours awake or
        circadian phase). Times beyond one period are fine.
    y : array-like
        Expression values (e.g. log2). NaN pairs are dropped.
    period : float
        Period ``T`` in hours (default 24).

    Returns
    -------
    dict with
        ``mesor``       rhythm-adjusted mean ``M``
        ``amplitude``   ``A = sqrt(β² + γ²)`` (half peak-to-trough, same units as y)
        ``acrophase_h`` time of peak ``φ`` in hours, in ``[0, period)``, on the same
                        clock as ``t_hours`` (mod period)
        ``r2``          coefficient of determination
        ``p``           zero-amplitude F-test p-value (H0: β = γ = 0), df = (2, n − 3)
        ``n``           number of points used

    Raises
    ------
    ValueError if fewer than 4 valid points remain (needs > 3 for the F-test).
    """
    t = np.asarray(t_hours, dtype=float)
    yy = np.asarray(y, dtype=float)
    if t.shape != yy.shape:
        raise ValueError("t_hours and y must have the same shape")
    ok = np.isfinite(t) & np.isfinite(yy)
    t, yy = t[ok], yy[ok]
    n = int(t.size)
    if n < 4:
        raise ValueError(f"cosinor_fit needs at least 4 points, got {n}")

    w = 2 * np.pi / period
    X = np.column_stack([np.ones(n), np.cos(w * t), np.sin(w * t)])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    mesor, beta, gamma = (float(c) for c in coef)
    fitted = X @ coef
    rss = float(np.sum((yy - fitted) ** 2))
    tss = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    df_resid = n - 3
    if df_resid > 0 and rss > 0:
        f = ((tss - rss) / 2) / (rss / df_resid)
        p = float(stats.f.sf(f, 2, df_resid))
    else:  # perfect fit or no residual df
        p = 0.0 if tss > 0 else 1.0

    amplitude = float(np.hypot(beta, gamma))
    acrophase_h = float((np.arctan2(gamma, beta) / w) % period)
    return CosinorResult(
        mesor=mesor, amplitude=amplitude, acrophase_h=acrophase_h, r2=float(r2), p=p, n=n
    )


def fit_subject_cosinors(
    clock_long: pd.DataFrame,
    time_col: str = "clock_time_h",
    genes: tuple[str, ...] = config.CLOCK_GENES,
    period: float = config.PERIOD_H,
    min_points: int = 6,
) -> pd.DataFrame:
    """Fit one cosinor per subject, condition and gene."""

    required = [
        "subject", "per3_genotype", "sleep_condition",
        "gene", time_col, "expression"
    ]

    missing = [col for col in required if col not in clock_long.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected = clock_long[clock_long["gene"].isin(genes)].copy()
    results = []

    group_columns = ["subject", "sleep_condition", "gene"]

    for (subject, condition, gene), group in selected.groupby(
        group_columns, sort=False
    ):
        times = pd.to_numeric(group[time_col], errors="coerce").to_numpy()
        values = pd.to_numeric(group["expression"], errors="coerce").to_numpy()

        valid = np.isfinite(times) & np.isfinite(values)

        if valid.sum() < min_points:
            continue

        fit = cosinor_fit(
            times[valid],
            values[valid],
            period=period
        )

        results.append({
            "subject": subject,
            "per3_genotype": group["per3_genotype"].iloc[0],
            "sleep_condition": condition,
            "gene": gene,
            "time_col": time_col,
            "mesor": fit["mesor"],
            "amplitude": fit["amplitude"],
            "acrophase_h": fit["acrophase_h"],
            "r2": fit["r2"],
            "p": fit["p"],
            "n": fit["n"]
        })

    columns = [
        "subject", "per3_genotype", "sleep_condition", "gene",
        "time_col", "mesor", "amplitude", "acrophase_h",
        "r2", "p", "n"
    ]

    return pd.DataFrame(results, columns=columns)


def rhythmicity_scan(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    time_col: str = "circadian_phase_h",
    period: float = config.PERIOD_H,
) -> pd.DataFrame:
    """Genome-wide rhythmicity scan with Benjamini–Hochberg FDR (Part 3).

    Parameters
    ----------
    expr : genes (or probes) × samples (columns = ``sample_id``), e.g. from
        ``collapse_probes_to_genes(load_expression_matrix())`` restricted to the
        analysis samples.
    meta : metadata rows for those samples (``analysis_samples()``).
    time_col : time variable for the fit.

    Returns
    -------
    DataFrame indexed by gene with ``mesor, amplitude, acrophase_h, r2, p, q`` where
    ``q`` is the BH-adjusted p-value. Sorted by ``q``. Saved by stage 5 as
    ``results/tables/rhythmic_genes_SE.csv`` (input to Part 4).
    How subjects are combined (pooled fit, per-subject then meta-analysis, or
    mixed model) is Part 3's decision. Document it.
    """
    # Match expression columns to metadata sample IDs.
    if "sample_id" not in meta.columns:
        raise ValueError("meta must contain a sample_id column")

    sample_ids = [
        sample_id for sample_id in expr.columns
        if sample_id in set(meta["sample_id"])
    ]

    if not sample_ids:
        raise ValueError("No matching sample IDs between expr and meta")

    meta_indexed = meta.drop_duplicates("sample_id").set_index("sample_id")
    aligned_meta = meta_indexed.reindex(sample_ids)

    if time_col not in aligned_meta.columns:
        raise ValueError(f"Time column '{time_col}' not found in metadata")

    times = pd.to_numeric(aligned_meta[time_col], errors="coerce").to_numpy()

    results = []

    # Fit one pooled cosinor model for each gene.
    for gene in expr.index:
        values = pd.to_numeric(
            expr.loc[gene, sample_ids], errors="coerce"
        ).to_numpy(dtype=float)

        valid = np.isfinite(times) & np.isfinite(values)

        if valid.sum() < 4:
            continue

        try:
            fit = cosinor_fit(
                times[valid],
                values[valid],
                period=period
            )
        except ValueError:
            continue

        results.append({
            "gene": gene,
            "mesor": fit["mesor"],
            "amplitude": fit["amplitude"],
            "acrophase_h": fit["acrophase_h"],
            "r2": fit["r2"],
            "p": fit["p"],
            "n": fit["n"]
        })

    if not results:
        raise ValueError("No genes had enough valid samples for cosinor fitting")

    result_df = pd.DataFrame(results).set_index("gene")

    # Benjamini-Hochberg false discovery rate correction.
    p_values = result_df["p"].to_numpy()
    number_of_tests = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted = sorted_p * number_of_tests / np.arange(
        1, number_of_tests + 1
    )

    # Enforce monotonicity of adjusted p-values.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    q_values = np.empty(number_of_tests)
    q_values[order] = adjusted

    result_df["q"] = q_values

    return result_df.sort_values("q")

