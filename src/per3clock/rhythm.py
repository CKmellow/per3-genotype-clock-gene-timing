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
    """Fit one cosinor per subject × condition × gene (Part 3).

    Parameters
    ----------
    clock_long : output of ``per3clock.data.load_clock_genes()`` (optionally filtered
        to ``analysis_samples()``). For ``time_col="circadian_phase_h"`` use
        ``analysis_samples(for_phase_analysis=True)``.
    time_col : ``"clock_time_h"`` (objectives 1–2) or ``"circadian_phase_h"``
        (objective 3). Clock time wraps at midnight; the cosinor is invariant to
        adding multiples of T, so no unwrapping is needed.
    min_points : skip series with fewer valid samples.

    Returns
    -------
    DataFrame, one row per subject × sleep_condition × gene, columns
    ``subject, per3_genotype, sleep_condition, gene, time_col, mesor, amplitude,
    acrophase_h, r2, p, n``.
    """
    required = ["subject", "per3_genotype", "sleep_condition", "gene", time_col, "expression"]
    missing = [col for col in required if col not in clock_long.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected = clock_long[clock_long["gene"].isin(genes)]
    results = []
    for (subject, condition, gene), group in selected.groupby(
        ["subject", "sleep_condition", "gene"], sort=False
    ):
        times = pd.to_numeric(group[time_col], errors="coerce").to_numpy()
        values = pd.to_numeric(group["expression"], errors="coerce").to_numpy()
        valid = np.isfinite(times) & np.isfinite(values)
        if valid.sum() < min_points:
            continue
        fit = cosinor_fit(times[valid], values[valid], period=period)
        results.append({
            "subject": subject,
            "per3_genotype": group["per3_genotype"].iloc[0],
            "sleep_condition": condition,
            "gene": gene,
            "time_col": time_col,
            **fit,
        })

    columns = ["subject", "per3_genotype", "sleep_condition", "gene", "time_col",
               "mesor", "amplitude", "acrophase_h", "r2", "p", "n"]
    return pd.DataFrame(results, columns=columns)


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (NaN-safe)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    out = np.empty(m)
    out[order] = ranked
    q[ok] = out
    return q


def rhythmicity_scan(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    time_col: str = "circadian_phase_h",
    period: float = config.PERIOD_H,
    min_points: int = 6,
) -> pd.DataFrame:
    """Genome-wide rhythmicity scan with Benjamini–Hochberg FDR (Part 3).

    **Method: population-mean cosinor** (Nelson et al. 1979, *Chronobiologia* 6:305;
    Cornelissen 2014, *Theor Biol Med Model* 11:16). Subjects are the unit:

    1. one cosinor per subject (``y = M + β·cos ωt + γ·sin ωt``) for every gene,
       which also removes each person's own expression level (``M``);
    2. a one-sample Hotelling T² test of whether the mean ``(β, γ)`` across the
       ``k`` subjects is zero: ``F = (k − 2) / (2(k − 1)) · T²`` with ``df = (2, k − 2)``.

    A gene is called rhythmic only if subjects share a rhythm with a **consistent
    phase**: rhythms with random phases cancel in the mean. This avoids two problems
    of the alternatives: pooling all samples in one fit treats repeated measures of
    the same person as independent (p-values far too small), and combining
    per-subject p-values (Fisher) flags genes that are rhythmic in *any* subject.

    Parameters
    ----------
    expr : genes (or probes) × samples (columns = ``sample_id``), no NaN. Use the
        Stage 3 expressed genes (``qc.expressed_genes``), the agreed feature set.
    meta : metadata rows for those samples, **one** sleep condition. For
        ``circadian_phase_h`` use ``analysis_samples(for_phase_analysis=True)``:
        series in ``data.PHASE_LABEL_EXCLUSIONS`` raise an error.
    time_col : time variable for the fit.
    min_points : subjects with fewer samples are left out (need ≥ 3 subjects).

    Returns
    -------
    DataFrame indexed by gene with ``mesor`` (mean of subject mesors),
    ``amplitude, acrophase_h`` (of the population-mean rhythm, i.e. of the mean
    ``(β, γ)``), ``r2`` (mean per-subject R²), ``mean_subject_amplitude``,
    ``F, p, q`` (``q`` = BH-adjusted p across all genes), ``n_subjects`` and ``n``
    (samples used). Sorted by ``q``. Saved by stage 5 as
    ``results/tables/rhythmic_genes_SE.csv`` (input to Part 4).
    """
    from . import data  # local import: data does not depend on rhythm

    if "sample_id" not in meta.columns:
        raise ValueError("meta must contain a sample_id column")
    m = meta.drop_duplicates("sample_id").set_index("sample_id")
    m = m.loc[[s for s in expr.columns if s in m.index]]
    if m.empty:
        raise ValueError("No matching sample IDs between expr and meta")
    if time_col not in m.columns:
        raise ValueError(f"Time column '{time_col}' not found in metadata")
    if "sleep_condition" in m.columns and m["sleep_condition"].nunique() > 1:
        raise ValueError("pass one sleep_condition; do not pool SE and SR")
    if time_col == "circadian_phase_h" and "sleep_condition" in m.columns:
        bad = set(zip(m["subject"], m["sleep_condition"], strict=True)) & set(
            data.PHASE_LABEL_EXCLUSIONS)
        if bad:
            raise ValueError(f"{sorted(bad)} have unreliable circadian_phase_h labels; "
                             "use analysis_samples(for_phase_analysis=True)")

    y_all = expr[list(m.index)].to_numpy(dtype=float)
    if np.isnan(y_all).any():
        raise ValueError("expr contains NaN; filter or impute first")
    times = pd.to_numeric(m[time_col], errors="coerce").to_numpy(dtype=float)
    w = 2 * np.pi / period

    coefs, r2s, n_used = [], [], 0
    for _, idx in m.groupby("subject", sort=False).indices.items():
        idx = idx[np.isfinite(times[idx])]
        if idx.size < min_points:
            continue
        t = times[idx]
        X = np.column_stack([np.ones(idx.size), np.cos(w * t), np.sin(w * t)])
        Y = y_all[:, idx].T  # samples × genes
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)  # 3 × genes
        rss = ((Y - X @ coef) ** 2).sum(axis=0)
        tss = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2s.append(np.where(tss > 0, 1 - rss / tss, 0.0))
        coefs.append(coef)
        n_used += idx.size

    k = len(coefs)
    if k < 3:
        raise ValueError(f"need at least 3 subjects with >= {min_points} samples, got {k}")
    C = np.stack(coefs)  # subjects × 3 × genes
    mesor, beta, gamma = C[:, 0], C[:, 1], C[:, 2]
    b, g = beta.mean(axis=0), gamma.mean(axis=0)
    s_bb, s_gg = beta.var(axis=0, ddof=1), gamma.var(axis=0, ddof=1)
    s_bg = ((beta - b) * (gamma - g)).sum(axis=0) / (k - 1)
    det = s_bb * s_gg - s_bg**2
    with np.errstate(invalid="ignore", divide="ignore"):
        t2 = k * (s_gg * b**2 - 2 * s_bg * b * g + s_bb * g**2) / det
    f = (k - 2) / (2 * (k - 1)) * t2
    p = stats.f.sf(f, 2, k - 2)

    out = pd.DataFrame({
        "mesor": mesor.mean(axis=0),
        "amplitude": np.hypot(b, g),
        "acrophase_h": (np.arctan2(g, b) / w) % period,
        "r2": np.mean(r2s, axis=0),
        "mean_subject_amplitude": np.hypot(beta, gamma).mean(axis=0),
        "F": f, "p": p,
    }, index=pd.Index(expr.index, name="gene"))
    out["q"] = _bh(out["p"].to_numpy())
    out["n_subjects"] = k
    out["n"] = n_used
    return out.sort_values(["q", "p"])
