import numpy as np
import pandas as pd
import pytest

from per3clock import data, rhythm

part3 = pytest.mark.xfail(reason="Part 3", strict=True, raises=NotImplementedError)


def _circ_err(a, b, period=24.0):
    d = (a - b) % period
    return min(d, period - d)


# --- cosinor_fit (implemented by Part 1; Part 3 owns extending it) ------------------------
@pytest.mark.parametrize("phi", [0.0, 3.3, 7.5, 13.0, 21.9, 23.95])
def test_cosinor_recovers_acrophase_noiseless(phi):
    t = np.arange(18, 18 + 30, 3.0)  # 10 samples every 3 h, like the protocol
    y = 7 + 0.8 * np.cos(2 * np.pi * (t - phi) / 24)
    r = rhythm.cosinor_fit(t, y)
    assert _circ_err(r["acrophase_h"], phi) < 0.1
    assert r["mesor"] == pytest.approx(7, abs=1e-9)
    assert r["amplitude"] == pytest.approx(0.8, abs=1e-9)
    assert r["r2"] == pytest.approx(1.0) and r["n"] == 10


def test_cosinor_recovers_acrophase_with_noise():
    rng = np.random.default_rng(1)
    t = np.tile(np.arange(0, 27, 3.0), 20)  # pooled samples, as in a population fit
    phi = 4.2
    y = 6 + 0.5 * np.cos(2 * np.pi * (t - phi) / 24) + rng.normal(0, 0.1, t.size)
    r = rhythm.cosinor_fit(t, y)
    assert _circ_err(r["acrophase_h"], phi) < 0.1
    assert r["p"] < 1e-10 and 0 <= r["acrophase_h"] < 24


def test_cosinor_null_signal_not_significant():
    rng = np.random.default_rng(2)
    t = np.arange(0, 30, 3.0)
    r = rhythm.cosinor_fit(t, rng.normal(5, 1, t.size))
    assert r["p"] > 0.01


def test_cosinor_time_origin_invariant():
    t = np.arange(18, 48, 3.0)
    y = 5 + np.cos(2 * np.pi * (t - 2) / 24)
    a = rhythm.cosinor_fit(t, y)["acrophase_h"]
    b = rhythm.cosinor_fit(t % 24, y)["acrophase_h"]  # clock time wrapped at midnight
    assert _circ_err(a, b) < 1e-9


def test_cosinor_drops_nan_and_validates():
    t = np.array([0, 3, 6, 9, 12, np.nan])
    y = np.array([1, 2, 3, 2, 1, 5.0])
    assert rhythm.cosinor_fit(t, y)["n"] == 5
    with pytest.raises(ValueError):
        rhythm.cosinor_fit([0, 3, 6], [1, 2, 3])
    with pytest.raises(ValueError):
        rhythm.cosinor_fit([0, 3, 6, 9], [1, 2, 3])


def test_cosinor_other_period():
    t = np.linspace(0, 20, 30)
    y = np.cos(2 * np.pi * (t - 5) / 12)
    assert _circ_err(rhythm.cosinor_fit(t, y, period=12)["acrophase_h"], 5, 12) < 0.01


# --- Part 3 contracts ---------------------------------------------------------------------
@part3
def test_fit_subject_cosinors_contract():
    clock = data.load_clock_genes()
    clock = clock[clock.sample_id.isin(data.analysis_samples().sample_id)]
    out = rhythm.fit_subject_cosinors(clock, time_col="clock_time_h")
    expected = {"subject", "per3_genotype", "sleep_condition", "gene", "mesor", "amplitude",
                "acrophase_h", "r2", "p", "n"}
    assert expected <= set(out.columns)
    assert out.groupby(["subject", "gene"]).size().max() == 1
    assert out["acrophase_h"].between(0, 24, inclusive="left").all()


@part3
def test_rhythmicity_scan_contract():
    rng = np.random.default_rng(0)
    meta = data.analysis_samples().head(60)
    t = meta["circadian_phase_h"].to_numpy()
    rhythmic = 5 + np.cos(2 * np.pi * t / 24) + rng.normal(0, 0.1, len(t))
    expr = pd.DataFrame(
        [rhythmic, rng.normal(5, 1, len(t))], index=["RHY", "FLAT"], columns=meta["sample_id"]
    )
    out = rhythm.rhythmicity_scan(expr, meta)
    assert {"amplitude", "acrophase_h", "p", "q"} <= set(out.columns)
    assert out.loc["RHY", "q"] < 0.05 < out.loc["FLAT", "q"]
