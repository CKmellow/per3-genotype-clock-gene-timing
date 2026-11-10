import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from per3clock import ml

part5 = pytest.mark.xfail(reason="Part 5", strict=True, raises=NotImplementedError)


@part5
def test_encode_decode_roundtrip():
    h = np.array([0, 3, 6, 11.5, 18, 23.9, -3, 15])
    enc = ml.encode_phase(h)
    assert enc.shape == (8, 2)
    assert np.allclose(ml.decode_phase(enc), h % 24)


@part5
def test_circular_abs_error():
    assert np.allclose(ml.circular_abs_error_h([23, 0, 6], [1, 12, 6]), [2, 12, 0])


@part5
def test_loso_cv_no_leakage_shapes():
    rng = np.random.default_rng(0)
    n_subj, per = 6, 8
    phase = np.tile(np.arange(-12, 12, 3.0), n_subj)
    groups = np.repeat([f"S{i}" for i in range(n_subj)], per)
    X = pd.DataFrame({"g1": np.cos(2 * np.pi * phase / 24), "g2": np.sin(2 * np.pi * phase / 24)})
    X += rng.normal(0, 0.05, X.shape)
    pred = ml.loso_cv_predict(Ridge(alpha=1e-3), X, phase, groups)
    assert pred.shape == (n_subj * per,)
    assert np.median(ml.circular_abs_error_h(phase % 24, pred)) < 1.0


@part5
def test_evaluate_phase_models_has_baselines():
    from per3clock import data

    meta = data.analysis_samples()
    clock = data.load_clock_genes()
    X = clock.pivot(index="sample_id", columns="gene", values="expression").loc[meta.sample_id]
    t = ml.evaluate_phase_models(X, meta)
    assert {"model", "median_abs_error_h"} <= set(t.columns)
    assert t["model"].str.contains("mean").any() and t["model"].str.contains("clock").any()
