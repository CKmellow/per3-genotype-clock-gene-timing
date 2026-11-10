"""Circular statistics for comparing acrophases between genotypes (Part 3).

All times are in hours on a circle of length ``period`` (default 24 h).
"""

from __future__ import annotations

from . import config


def circular_mean_h(times_h, period: float = config.PERIOD_H) -> float:
    """Circular mean of times in hours, returned in ``[0, period)``.

    Example: ``circular_mean_h([23, 1]) == 0`` (not 12).
    """
    raise NotImplementedError("Part 3: implement circular_mean_h")


def circular_diff_h(a_h, b_h, period: float = config.PERIOD_H):
    """Signed shortest difference ``a − b`` in hours, in ``[-period/2, period/2)``.

    Example: ``circular_diff_h(1, 23) == 2``; ``circular_diff_h(23, 1) == -2``.
    Positive = ``a`` is later than ``b``.
    """
    raise NotImplementedError("Part 3: implement circular_diff_h")


def watson_williams_test(*groups_h, period: float = config.PERIOD_H) -> dict:
    """Watson–Williams test for equal circular means across groups.

    Parameters
    ----------
    *groups_h : two or more array-likes of times in hours.

    Returns
    -------
    ``{"F": float, "p": float, "df": (int, int)}``. Document the assumptions
    (von Mises, similar concentration) and check them.
    """
    raise NotImplementedError("Part 3: implement watson_williams_test")


def permutation_test_acrophase(
    a_h, b_h, n_perm: int = 10_000, period: float = config.PERIOD_H, seed: int = config.SEED
) -> dict:
    """Permutation test on the circular-mean difference between two groups.

    Shuffles group labels ``n_perm`` times (with ``np.random.default_rng(seed)``).

    Returns
    -------
    ``{"diff_h": circular_diff_h(mean(a), mean(b)), "p": two-sided p-value,
    "n_perm": n_perm}``. Use ``a = 5/5``, ``b = 4/4`` so that a negative ``diff_h``
    means 5/5 peaks earlier (the hypothesis).
    """
    raise NotImplementedError("Part 3: implement permutation_test_acrophase")


def bootstrap_ci_diff_h(
    a_h,
    b_h,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    period: float = config.PERIOD_H,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI (resampling subjects within group) for the
    circular-mean difference ``a − b`` in hours."""
    raise NotImplementedError("Part 3: implement bootstrap_ci_diff_h")
