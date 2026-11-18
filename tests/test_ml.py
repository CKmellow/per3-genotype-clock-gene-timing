import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from per3clock import ml


def test_encode_decode_roundtrip():
    h = np.array([0, 3, 6, 11.5, 18, 23.9, -3, 15])
    enc = ml.encode_phase(h)
    assert enc.shape == (8, 2)
    assert np.allclose(ml.decode_phase(enc), h % 24)


def test_circular_abs_error():
    assert np.allclose(ml.circular_abs_error_h([23, 0, 6], [1, 12, 6]), [2, 12, 0])


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


def test_evaluate_phase_models_has_baselines():
    from per3clock import data

    meta = data.analysis_samples()
    clock = data.load_clock_genes()
    X = clock.pivot(index="sample_id", columns="gene", values="expression").loc[meta.sample_id]
    t = ml.evaluate_phase_models(X, meta)
    assert {"model", "median_abs_error_h"} <= set(t.columns)
    assert t["model"].str.contains("mean").any() and t["model"].str.contains("clock").any()


def test_baselines_and_selection():
    rng = np.random.default_rng(1)
    n_subj, per = 5, 8
    phase = np.tile(np.arange(0, 24, 3.0), n_subj)
    offsets = np.repeat(rng.normal(0, 0.5, n_subj), per)
    clock = np.mod(phase + 5 + offsets, 24)
    meta = pd.DataFrame({"sample_id": [f"s{i}" for i in range(len(phase))],
                         "subject": np.repeat([f"S{i}" for i in range(n_subj)], per),
                         "per3_genotype": np.repeat(["4/4", "5/5", "4/4", "5/5", "4/4"], per),
                         "clock_time_h": clock, "circadian_phase_h": phase})
    X = pd.DataFrame(rng.normal(size=(len(phase), 30)), columns=[f"g{i}" for i in range(30)])
    X["rhythmic"] = np.cos(2 * np.pi * (phase - 4) / 24) + rng.normal(0, 0.05, len(phase))
    pred = ml.phase_predictions(ml.default_phase_models(X, meta, k=3), meta)
    s = ml.summarise_predictions(pred, n_boot=50).set_index("model")
    assert s.loc["baseline: clock time only", "median_abs_error_h"] < 1.0
    assert s.loc["baseline: circular mean phase", "median_abs_error_h"] > 3.0
    assert set(s["n_subjects"]) == {n_subj}
    freq = ml.fold_selection_frequency(X, phase, meta["subject"], k=1)
    assert freq.iloc[0]["gene"] == "rhythmic" and freq.iloc[0]["frac_folds"] == 1.0
    cmp = ml.paired_subject_comparison(pred, "baseline: circular mean phase",
                                       "baseline: clock time only", n_boot=100)
    assert cmp["mean_diff_h"] < 0 and cmp["n_subjects"] == n_subj
    centred = ml.within_subject_centre(X, meta["subject"])
    assert np.allclose(centred.groupby(meta["subject"].to_numpy()).mean(), 0)
