import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from per3clock import config, data, eda  # noqa: E402


@pytest.fixture(scope="module")
def main_clock():
    clock = data.load_clock_genes()
    return clock[clock.sample_id.isin(data.analysis_samples().sample_id)]


def test_plot_clock_gene_profiles():
    fig = eda.plot_clock_gene_profiles(data.load_clock_genes())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_pca_samples():
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(rng.normal(size=(100, 30)), columns=[f"S{i}" for i in range(30)])
    scores, ev = eda.pca_samples(expr, n_components=3)
    assert scores.shape == (30, 3) and list(scores.columns) == ["PC1", "PC2", "PC3"]
    assert (np.diff(ev.to_numpy()) <= 1e-12).all()


def test_plot_pca_and_correlation():
    meta = data.analysis_samples().head(12)
    rng = np.random.default_rng(1)
    expr = pd.DataFrame(rng.normal(size=(40, 12)), columns=meta.sample_id)
    scores, _ = eda.pca_samples(expr, n_components=2)
    for fig in (eda.plot_pca(scores, meta), eda.plot_sample_correlation(expr, meta)):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# --- time handling -----------------------------------------------------------------------
def test_unwrap_clock_time(meta):
    u = eda.unwrap_clock_time(meta)
    assert u.index.equals(meta.index) and u.name == "clock_time_unwrapped_h"
    days = (u - meta.clock_time_h) / 24  # same clock time, shifted by whole days
    assert np.allclose(days, days.round()) and set(days.round()) <= {0, 1}
    assert u.between(12, 48).all()
    df = meta.assign(u=u).sort_values(["subject", "sleep_condition", "timepoint_n"])
    for _, g in df.groupby(["subject", "sleep_condition"]):
        step = np.diff(g.u) / np.diff(g.timepoint_n)
        assert (step > 1.5).all() and (step < 4.5).all()  # ~3 h per timepoint, never wraps


def test_subject_timing_summary(meta):
    main = data.analysis_samples(metadata=meta)
    t = eda.subject_timing_summary(main)
    assert len(t) == 24 and t.n_samples.sum() == 221
    assert t.wake_clock_h.between(5, 12).all()  # morning wake times
    assert np.isnan(t.loc["AF0182", "phase0_clock_h"])
    assert t.drop(index="AF0182").phase0_clock_h.notna().all()
    # data/README.md: implied phase-0 clock time averages ~04:10 in SE
    assert t.phase0_clock_h.mean() == pytest.approx(4.1, abs=0.3)
    with pytest.raises(ValueError):
        eda.subject_timing_summary(meta)


def test_genotype_profile_table(main_clock):
    t = eda.genotype_profile_table(main_clock, center=True)
    assert list(t.columns) == ["gene", "per3_genotype", "time_bin_h", "mean", "se", "n_subjects"]
    assert set(t.gene) == set(config.GENES_PRIMARY)
    assert t.groupby("per3_genotype").n_subjects.max().to_dict() == {"4/4": 11, "5/5": 13}
    # subject-centred group means (bins with >= 3 subjects, as plotted) are small
    shown = t[t.n_subjects >= eda.MIN_SUBJECTS_PER_BIN]
    assert shown["mean"].abs().max() < 1 and shown["se"].notna().all()
    tp = eda.genotype_profile_table(main_clock, time_col="circadian_phase_h")
    assert tp.groupby("per3_genotype").n_subjects.max().to_dict() == {"4/4": 11, "5/5": 12}
    assert set(tp.time_bin_h) <= set(range(-15, 16, 3))


def test_plot_clock_gene_profiles_rows_and_condition(main_clock):
    fig = eda.plot_clock_gene_profiles(main_clock, genes=("CLOCK", "CRY1"),
                                       time_col=("clock_time_h", "circadian_phase_h"),
                                       center=True, show_subjects=False)
    assert len(fig.axes) == 4
    plt.close(fig)
    with pytest.raises(ValueError, match="no rows"):
        eda.plot_clock_gene_profiles(main_clock, condition="Sleep Restriction")


# --- PCA ---------------------------------------------------------------------------------
def test_pca_samples_rejects_nan_and_is_deterministic():
    rng = np.random.default_rng(2)
    expr = pd.DataFrame(rng.normal(size=(20, 8)))
    a, _ = eda.pca_samples(expr, 3)
    b, _ = eda.pca_samples(expr, 3)
    pd.testing.assert_frame_equal(a, b)
    expr.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        eda.pca_samples(expr)


def test_pca_factor_r2_identifies_subject_effect():
    meta = data.analysis_samples()
    rng = np.random.default_rng(4)
    offset = {s: rng.normal(0, 5, 50) for s in meta.subject.unique()}
    expr = pd.DataFrame({sid: offset[s] + rng.normal(0, 0.5, 50)
                         for sid, s in zip(meta.sample_id, meta.subject, strict=True)})
    scores, ev = eda.pca_samples(expr, 3)
    t = eda.pca_factor_r2(scores, meta, explained=ev)
    assert list(t.index) == ["PC1", "PC2", "PC3", "weighted"]
    assert t.loc["PC1", "subject"] > 0.9
    assert (t.loc["PC1", "per3_genotype"] <= t.loc["PC1", "subject"] + 1e-12)
    assert t.loc["weighted", "explained"] == pytest.approx(ev.sum())


def test_correlation_summary():
    meta = data.analysis_samples()
    meta = meta[meta.subject.isin(["AF0004", "AF0010", "AF0101"])]  # 4/4, 4/4, 5/5
    rng = np.random.default_rng(8)
    prof = {s: rng.normal(0, 1, 100) for s in meta.subject.unique()}
    expr = pd.DataFrame({sid: prof[s] + rng.normal(0, 0.3, 100)
                         for sid, s in zip(meta.sample_id, meta.subject, strict=True)})
    t = eda.correlation_summary(expr, meta)
    assert list(t.index) == ["same subject", "different subject, same genotype",
                             "different genotype"]
    n = len(meta)
    assert t.n_pairs.sum() == n * (n - 1) // 2
    assert t.loc["same subject", "median_r"] > 0.8 > t.loc["different genotype", "median_r"]


def test_plot_pca_variants():
    meta = data.analysis_samples().head(30)
    rng = np.random.default_rng(6)
    scores, ev = eda.pca_samples(pd.DataFrame(rng.normal(size=(40, 30)), columns=meta.sample_id))
    for cb in ("per3_genotype", "timepoint", "circadian_phase_h", "clock_time_h",
               "sleep_condition"):
        fig = eda.plot_pca(scores, meta, cb, explained=ev)
        assert isinstance(fig, plt.Figure)
        assert "% of variance" in fig.axes[0].get_xlabel()
        plt.close(fig)
    fig = eda.plot_pca_grid(scores, meta, ("subject", "per3_genotype", "timepoint"),
                            explained=ev, suptitle="t")
    assert sum(ax.get_visible() for ax in fig.axes if ax.get_label() != "<colorbar>") >= 3
    plt.close(fig)
