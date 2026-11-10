import numpy as np
import pytest

from per3clock import circstats

part3 = pytest.mark.xfail(reason="Part 3", strict=True, raises=NotImplementedError)


@part3
def test_circular_mean_wraps():
    assert circstats.circular_mean_h([23, 1]) == pytest.approx(0, abs=1e-9) or \
        circstats.circular_mean_h([23, 1]) == pytest.approx(24, abs=1e-9)
    assert circstats.circular_mean_h([5, 7]) == pytest.approx(6)


@part3
def test_circular_diff():
    assert circstats.circular_diff_h(1, 23) == pytest.approx(2)
    assert circstats.circular_diff_h(23, 1) == pytest.approx(-2)


@part3
def test_watson_williams_detects_shift():
    rng = np.random.default_rng(0)
    a = (2 + rng.normal(0, 1, 15)) % 24
    b = (6 + rng.normal(0, 1, 15)) % 24
    assert circstats.watson_williams_test(a, b)["p"] < 0.001
    assert circstats.watson_williams_test(a, (2 + rng.normal(0, 1, 15)) % 24)["p"] > 0.01


@part3
def test_permutation_test_sign_and_reproducible():
    rng = np.random.default_rng(1)
    a = (22 + rng.normal(0, 1, 12)) % 24  # earlier (5/5)
    b = (1 + rng.normal(0, 1, 12)) % 24   # later (4/4)
    r1 = circstats.permutation_test_acrophase(a, b, n_perm=2000)
    r2 = circstats.permutation_test_acrophase(a, b, n_perm=2000)
    assert r1["diff_h"] < 0 and r1["p"] < 0.01 and r1 == r2


@part3
def test_bootstrap_ci_contains_true_shift():
    rng = np.random.default_rng(2)
    a = (3 + rng.normal(0, 0.5, 20)) % 24
    b = (5 + rng.normal(0, 0.5, 20)) % 24
    lo, hi = circstats.bootstrap_ci_diff_h(a, b, n_boot=2000)
    assert lo < -2 + 0.6 and hi > -2 - 0.6 and lo < hi
