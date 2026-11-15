"""Circular statistics for comparing acrophases between genotypes (Part 3).

All times are in hours on a circle of length ``period`` (default 24 h).

Sign convention used throughout: differences are ``a − b`` with ``a = 5/5`` and
``b = 4/4``, so a **negative** shift means 5/5 peaks **earlier** (the hypothesis).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from . import config


def _clean(values) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    return x[np.isfinite(x)]


def _mean_resultant_length(angles: np.ndarray) -> float:
    if angles.size == 0:
        return float("nan")
    return float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles))))


def circular_mean_h(times_h, period: float = config.PERIOD_H) -> float:
    """Circular mean of times in hours, returned in ``[0, period)``.

    Example: ``circular_mean_h([23, 1]) == 0`` (not 12). NaNs are ignored. Returns
    NaN for an empty input or when the mean direction is undefined (resultant ≈ 0,
    e.g. ``[0, 12]``).
    """
    times = _clean(times_h)
    if times.size == 0:
        return float("nan")

    angles = times * (2 * np.pi / period)
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    if np.hypot(sin_mean, cos_mean) < 1e-12:
        return float("nan")

    result = float((np.arctan2(sin_mean, cos_mean) * period / (2 * np.pi)) % period)
    return 0.0 if np.isclose(result, period, atol=1e-10) else result


def circular_diff_h(a_h, b_h, period: float = config.PERIOD_H):
    """Signed shortest difference ``a − b`` in hours, in ``[-period/2, period/2)``.

    Example: ``circular_diff_h(1, 23) == 2``; ``circular_diff_h(23, 1) == -2``.
    Positive = ``a`` is later than ``b``. Works element-wise on arrays; returns a
    float for scalar input.
    """
    diff = (np.asarray(a_h, dtype=float) - np.asarray(b_h, dtype=float)) % period
    diff = np.where(diff >= period / 2, diff - period, diff)
    return float(diff) if np.ndim(diff) == 0 else diff


def kappa_from_rbar(rbar: float, n: int | None = None) -> float:
    """Von Mises concentration κ estimated from a mean resultant length.

    Fisher's (1993) approximation to the ML estimate (``A1⁻¹``), with Fisher's
    small-sample correction when ``n < 15`` (as in CircStat / pycircstat).
    """
    r = float(rbar)
    if r < 0.53:
        k = 2 * r + r**3 + 5 * r**5 / 6
    elif r < 0.85:
        k = -0.4 + 1.39 * r + 0.43 / (1 - r)
    else:
        k = 1 / (r**3 - 4 * r**2 + 3 * r) if r < 1 else np.inf
    if n is not None and n < 15 and np.isfinite(k):
        k = max(k - 2 / (n * k), 0.0) if k < 2 else (n - 1) ** 3 * k / (n**3 + n)
    return float(k)


def watson_williams_test(*groups_h, period: float = config.PERIOD_H) -> dict:
    """Watson–Williams test for equal circular means across groups.

    ``F = K · (N − k)(ΣRᵢ − R) / ((k − 1)(N − ΣRᵢ))`` with ``df = (k − 1, N − k)``,
    where ``Rᵢ`` are the group resultant lengths, ``R`` the pooled one and
    ``K = 1 + 3/(8κ)`` the correction factor (κ estimated from the pooled
    within-group mean resultant length ``ΣRᵢ/N``; Zar, *Biostatistical Analysis*,
    ch. 27).

    Assumptions: each group is von Mises distributed with a **common** concentration,
    and the data are not too dispersed (the approximation is poor when the pooled
    within-group mean resultant length is below ~0.45). Both are checked and
    reported in ``notes``, and a :class:`RuntimeWarning` is raised. When they fail,
    prefer :func:`permutation_test_acrophase`, which does not need them.

    Parameters
    ----------
    *groups_h : two or more array-likes of times in hours (NaNs ignored, ≥ 2 each).

    Returns
    -------
    ``{"F": float, "p": float, "df": (int, int), "kappa": float, "notes": str}``.
    """
    if len(groups_h) < 2:
        raise ValueError("At least two groups are required.")
    groups = [_clean(g) for g in groups_h]
    if any(g.size < 2 for g in groups):
        raise ValueError("Each group must contain at least two observations.")

    angles = [g * (2 * np.pi / period) for g in groups]
    k = len(groups)
    sizes = np.array([g.size for g in groups])
    n_total = int(sizes.sum())
    df = (k - 1, n_total - k)

    resultants = np.array([np.hypot(np.cos(a).sum(), np.sin(a).sum()) for a in angles])
    group_rbar = resultants / sizes
    pooled = np.concatenate(angles)
    pooled_resultant = np.hypot(np.cos(pooled).sum(), np.sin(pooled).sum())
    sum_resultants = resultants.sum()

    rbar_within = sum_resultants / n_total
    kappa = kappa_from_rbar(rbar_within, n_total)

    notes = []
    if rbar_within < 0.45:
        notes.append(f"low concentration (mean resultant length {rbar_within:.2f} < 0.45)")
    if np.ptp(group_rbar) > 0.20:
        notes.append("unequal concentrations (group mean resultant lengths "
                     + ", ".join(f"{r:.2f}" for r in group_rbar) + ")")
    for note in notes:
        warnings.warn(f"Watson-Williams assumption: {note}", RuntimeWarning, stacklevel=2)

    if sum_resultants <= 0 or n_total <= k or kappa == 0:
        return {"F": float("nan"), "p": float("nan"), "df": df, "kappa": kappa,
                "notes": "; ".join(notes)}

    correction = 1 + 3 / (8 * kappa) if np.isfinite(kappa) else 1.0
    numerator = (n_total - k) * (sum_resultants - pooled_resultant)
    denominator = (k - 1) * (n_total - sum_resultants)
    if denominator <= 0:
        f_stat = 0.0 if numerator <= 0 else float("inf")
    else:
        f_stat = max(0.0, correction * numerator / denominator)

    return {"F": float(f_stat), "p": float(stats.f.sf(f_stat, *df)), "df": df,
            "kappa": kappa, "notes": "; ".join(notes)}


def permutation_test_acrophase(
    a_h, b_h, n_perm: int = 10_000, period: float = config.PERIOD_H, seed: int = config.SEED
) -> dict:
    """Permutation test on the circular-mean difference between two groups.

    Shuffles group labels ``n_perm`` times (with ``np.random.default_rng(seed)``).
    Two-sided: counts permutations with ``|diff| ≥ |observed|``; the p-value is
    ``(count + 1) / (n_perm + 1)``. Needs no distributional assumption.

    Returns
    -------
    ``{"diff_h": circular_diff_h(mean(a), mean(b)), "p": two-sided p-value,
    "n_perm": n_perm}``. Use ``a = 5/5``, ``b = 4/4`` so that a negative ``diff_h``
    means 5/5 peaks earlier (the hypothesis).
    """
    a, b = _clean(a_h), _clean(b_h)
    if a.size == 0 or b.size == 0:
        raise ValueError("Both groups must contain valid observations.")

    observed = circular_diff_h(circular_mean_h(a, period), circular_mean_h(b, period), period)
    combined = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(combined)
        perm = circular_diff_h(circular_mean_h(shuffled[: a.size], period),
                               circular_mean_h(shuffled[a.size:], period), period)
        if abs(perm) >= abs(observed):
            count += 1
    return {"diff_h": float(observed), "p": float((count + 1) / (n_perm + 1)),
            "n_perm": n_perm}


def bootstrap_ci_diff_h(
    a_h,
    b_h,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    period: float = config.PERIOD_H,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI (resampling subjects within group) for the
    circular-mean difference ``a − b`` in hours.

    Each bootstrap difference is unwrapped to lie within ±period/2 of the observed
    difference before taking quantiles, so a CI that straddles ±12 h is not split.
    """
    a, b = _clean(a_h), _clean(b_h)
    if a.size == 0 or b.size == 0:
        raise ValueError("Both groups must contain valid observations.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    observed = circular_diff_h(circular_mean_h(a, period), circular_mean_h(b, period), period)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        d = circular_diff_h(circular_mean_h(rng.choice(a, a.size), period),
                            circular_mean_h(rng.choice(b, b.size), period), period)
        diffs[i] = observed + circular_diff_h(d, observed, period)
    lower, upper = np.nanquantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def genotype_shift_table(
    fits: pd.DataFrame,
    by: tuple[str, ...] = ("gene", "time_base"),
    *,
    n_perm: int = 10_000,
    n_boot: int = 10_000,
    period: float = config.PERIOD_H,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Compare per-subject acrophases between genotypes, one row per ``by`` group.

    Parameters
    ----------
    fits : per-subject cosinor fits (:func:`per3clock.rhythm.fit_subject_cosinors`)
        with ``per3_genotype`` and ``acrophase_h`` plus the ``by`` columns.
    by : columns that define one comparison (e.g. gene × time base).

    Returns
    -------
    DataFrame with the ``by`` columns and ``n_44, n_55, mean_acrophase_44_h,
    mean_acrophase_55_h, rbar_44, rbar_55`` (mean resultant length: 1 = all subjects
    peak together, 0 = no common peak), ``shift_h`` (5/5 − 4/4; negative = 5/5
    earlier), ``ci_lower_h, ci_upper_h`` (95 % bootstrap), ``permutation_p,
    watson_williams_F, watson_williams_p, ww_assumptions`` (empty if the
    Watson–Williams assumptions look fine) and the Benjamini–Hochberg
    ``permutation_q, watson_williams_q``, computed **across all rows of the returned
    table** (one family of tests: call once per family).
    """
    gt44, gt55 = config.GENOTYPES
    w = 2 * np.pi / period
    rows = []
    for key, g in fits.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        a = _clean(g.loc[g["per3_genotype"] == gt55, "acrophase_h"])
        b = _clean(g.loc[g["per3_genotype"] == gt44, "acrophase_h"])
        perm = permutation_test_acrophase(a, b, n_perm=n_perm, period=period, seed=seed)
        lo, hi = bootstrap_ci_diff_h(a, b, n_boot=n_boot, period=period, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ww = watson_williams_test(b, a, period=period)
        rows.append({
            **dict(zip(by, key, strict=True)),
            "n_44": b.size, "n_55": a.size,
            "mean_acrophase_44_h": circular_mean_h(b, period),
            "mean_acrophase_55_h": circular_mean_h(a, period),
            "rbar_44": _mean_resultant_length(b * w),
            "rbar_55": _mean_resultant_length(a * w),
            "shift_h": perm["diff_h"], "ci_lower_h": lo, "ci_upper_h": hi,
            "permutation_p": perm["p"],
            "watson_williams_F": ww["F"], "watson_williams_p": ww["p"],
            "ww_assumptions": ww["notes"],
        })
    out = pd.DataFrame(rows)
    for col in ("permutation", "watson_williams"):
        p = out[f"{col}_p"]
        q = pd.Series(np.nan, index=out.index)
        ok = p.notna()
        if ok.any():
            q[ok] = multipletests(p[ok], method="fdr_bh")[1]
        out[f"{col}_q"] = q
    return out
