"""Circular statistics for comparing acrophases between genotypes."""

from __future__ import annotations

import warnings
import numpy as np
from scipy import stats
from . import config


def circular_mean_h(times_h, period: float = config.PERIOD_H) -> float:
    """Circular mean of times in hours, returned in [0, period)."""
    times = np.asarray(times_h, dtype=float).ravel()
    times = times[np.isfinite(times)]

    if times.size == 0:
        return float("nan")

    angles = times * (2 * np.pi / period)
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))

    if np.hypot(sin_mean, cos_mean) < 1e-12:
        return float("nan")

    mean_angle = np.arctan2(sin_mean, cos_mean)
    result = float((mean_angle * period / (2 * np.pi)) % period)

    if np.isclose(result, period, atol=1e-10):
        result = 0.0

    return result


def circular_diff_h(a_h, b_h, period: float = config.PERIOD_H):
    """Signed shortest difference a - b in [-period/2, period/2)."""
    diff = (np.asarray(a_h) - np.asarray(b_h)) % period
    diff = np.where(diff >= period / 2, diff - period, diff)

    if np.ndim(diff) == 0:
        return float(diff)

    return diff


def watson_williams_test(
    *groups_h, period: float = config.PERIOD_H
) -> dict:
    """Watson-Williams test for equal circular means.

    Assumes approximately von Mises-distributed groups with
    similar concentration. The test is less reliable for very
    dispersed data or substantially unequal concentrations.
    """
    if len(groups_h) < 2:
        raise ValueError("At least two groups are required.")

    groups = []
    for group in groups_h:
        values = np.asarray(group, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size < 2:
            raise ValueError("Each group must contain at least two observations.")
        groups.append(values)

    angles = [
        values * (2 * np.pi / period)
        for values in groups
    ]

    n_groups = len(groups)
    sizes = np.array([len(g) for g in groups])
    n_total = int(sizes.sum())

    resultants = np.array([
        np.hypot(np.sum(np.cos(a)), np.sum(np.sin(a)))
        for a in angles
    ])

    group_rbar = resultants / sizes
    pooled_cos = sum(np.sum(np.cos(a)) for a in angles)
    pooled_sin = sum(np.sum(np.sin(a)) for a in angles)
    pooled_resultant = np.hypot(pooled_cos, pooled_sin)

    sum_resultants = resultants.sum()

    if sum_resultants <= 0 or n_total <= n_groups:
        return {"F": float("nan"), "p": float("nan"),
                "df": (n_groups - 1, n_total - n_groups)}

    # Check the concentration assumptions.
    if pooled_resultant / n_total < 0.45:
        warnings.warn(
            "Low pooled circular concentration; Watson-Williams "
            "results may be unreliable.",
            RuntimeWarning
        )

    if np.ptp(group_rbar) > 0.20:
        warnings.warn(
            "Group concentrations differ considerably; the "
            "Watson-Williams equal-concentration assumption may "
            "not hold.",
            RuntimeWarning
        )

    numerator = (n_total - n_groups) * (
        sum_resultants - pooled_resultant
    )
    denominator = (n_groups - 1) * (
        n_total - sum_resultants
    )

    if denominator <= 0:
        f_stat = 0.0 if numerator <= 0 else float("inf")
    else:
        f_stat = max(0.0, numerator / denominator)

    df = (n_groups - 1, n_total - n_groups)
    p_value = float(stats.f.sf(f_stat, *df))

    return {"F": float(f_stat), "p": p_value, "df": df}


def permutation_test_acrophase(
    a_h,
    b_h,
    n_perm: int = 10_000,
    period: float = config.PERIOD_H,
    seed: int = config.SEED
) -> dict:
    """Permutation test on the circular-mean difference a - b."""
    a = np.asarray(a_h, dtype=float).ravel()
    b = np.asarray(b_h, dtype=float).ravel()

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both groups must contain valid observations.")

    observed = circular_diff_h(
        circular_mean_h(a, period),
        circular_mean_h(b, period),
        period
    )

    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)

    count = 0

    for _ in range(n_perm):
        shuffled = rng.permutation(combined)
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]

        perm_diff = circular_diff_h(
            circular_mean_h(perm_a, period),
            circular_mean_h(perm_b, period),
            period
        )

        if abs(perm_diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_perm + 1)

    return {
        "diff_h": float(observed),
        "p": float(p_value),
        "n_perm": n_perm
    }


def bootstrap_ci_diff_h(
    a_h,
    b_h,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    period: float = config.PERIOD_H,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Bootstrap CI for the circular-mean difference a - b."""
    a = np.asarray(a_h, dtype=float).ravel()
    b = np.asarray(b_h, dtype=float).ravel()

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both groups must contain valid observations.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    observed = circular_diff_h(
        circular_mean_h(a, period),
        circular_mean_h(b, period),
        period
    )

    rng = np.random.default_rng(seed)
    differences = []

    for _ in range(n_boot):
        boot_a = rng.choice(a, size=len(a), replace=True)
        boot_b = rng.choice(b, size=len(b), replace=True)

        difference = circular_diff_h(
            circular_mean_h(boot_a, period),
            circular_mean_h(boot_b, period),
            period
        )

        # Align bootstrap differences around the observed difference.
        difference = observed + circular_diff_h(
            difference, observed, period
        )

        differences.append(difference)

    lower, upper = np.quantile(
        differences,
        [alpha / 2, 1 - alpha / 2]
    )

    return float(lower), float(upper)
