import numpy as np
import pytest

from per3clock import circstats


def test_circular_mean_wraps():
    assert circstats.circular_mean_h([23, 1]) == pytest.approx(0, abs=1e-9) or \
        circstats.circular_mean_h([23, 1]) == pytest.approx(24, abs=1e-9)
    assert circstats.circular_mean_h([5, 7]) == pytest.approx(6)


def test_circular_diff():
    assert circstats.circular_diff_h(1, 23) == pytest.approx(2)
    assert circstats.circular_diff_h(23, 1) == pytest.approx(-2)


def test_watson_williams_detects_shift():
    rng = np.random.default_rng(0)
    a = (2 + rng.normal(0, 1, 15)) % 24
    b = (6 + rng.normal(0, 1, 15)) % 24
    assert circstats.watson_williams_test(a, b)["p"] < 0.001
    assert circstats.watson_williams_test(a, (2 + rng.normal(0, 1, 15)) % 24)["p"] > 0.01


def test_permutation_test_sign_and_reproducible():
    rng = np.random.default_rng(1)
    a = (22 + rng.normal(0, 1, 12)) % 24  # earlier (5/5)
    b = (1 + rng.normal(0, 1, 12)) % 24   # later (4/4)
    r1 = circstats.permutation_test_acrophase(a, b, n_perm=2000)
    r2 = circstats.permutation_test_acrophase(a, b, n_perm=2000)
    assert r1["diff_h"] < 0 and r1["p"] < 0.01 and r1 == r2


def test_bootstrap_ci_contains_true_shift():
    rng = np.random.default_rng(2)
    a = (3 + rng.normal(0, 0.5, 20)) % 24
    b = (5 + rng.normal(0, 0.5, 20)) % 24
    lo, hi = circstats.bootstrap_ci_diff_h(a, b, n_boot=2000)
    assert lo < -2 + 0.6 and hi > -2 - 0.6 and lo < hi


def test_kappa_from_rbar_inverts_a1():
    from scipy.special import i0, i1

    for k in (0.5, 1.0, 2.0, 5.0, 10.0):
        rbar = i1(k) / i0(k)
        assert circstats.kappa_from_rbar(rbar) == pytest.approx(k, rel=0.05)


def test_watson_williams_applies_correction_factor():
    rng = np.random.default_rng(3)
    a = (2 + rng.normal(0, 2, 20)) % 24
    b = (4 + rng.normal(0, 2, 20)) % 24
    res = circstats.watson_williams_test(a, b)
    w = 2 * np.pi / 24
    r_i = [np.hypot(np.cos(x * w).sum(), np.sin(x * w).sum()) for x in (a, b)]
    pooled = np.concatenate([a, b]) * w
    r = np.hypot(np.cos(pooled).sum(), np.sin(pooled).sum())
    n = 40
    uncorrected = (n - 2) * (sum(r_i) - r) / (n - sum(r_i))
    k = circstats.kappa_from_rbar(sum(r_i) / n, n)
    assert res["kappa"] == pytest.approx(k)
    assert res["F"] == pytest.approx((1 + 3 / (8 * k)) * uncorrected)
    assert res["df"] == (1, 38) and res["notes"] == ""


def test_watson_williams_reports_assumption_problems():
    rng = np.random.default_rng(4)
    tight = (6 + rng.normal(0, 0.5, 15)) % 24
    spread = rng.uniform(0, 24, 15)
    with pytest.warns(RuntimeWarning):
        res = circstats.watson_williams_test(tight, spread)
    assert "unequal concentrations" in res["notes"]


def test_genotype_shift_table():
    import pandas as pd

    rng = np.random.default_rng(5)
    rows = []
    for gene, shift in (("A", -3.0), ("B", 0.0)):
        for gt, centre, n in (("4/4", 6.0, 11), ("5/5", 6.0 + shift, 13)):
            for i, t in enumerate((centre + rng.normal(0, 0.7, n)) % 24):
                rows.append({"gene": gene, "time_base": "clock_time", "subject": f"{gt}{i}",
                             "per3_genotype": gt, "acrophase_h": t})
    out = circstats.genotype_shift_table(pd.DataFrame(rows), n_perm=999, n_boot=999)
    out = out.set_index("gene")
    assert list(out[["n_44", "n_55"]].loc["A"]) == [11, 13]
    assert out.loc["A", "shift_h"] == pytest.approx(-3, abs=0.8)
    assert out.loc["A", "ci_lower_h"] < out.loc["A", "shift_h"] < out.loc["A", "ci_upper_h"]
    assert out.loc["A", "permutation_q"] < 0.01 < out.loc["B", "permutation_q"]
    assert (out["permutation_q"] >= out["permutation_p"]).all()
