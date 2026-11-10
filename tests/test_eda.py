import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from per3clock import data, eda  # noqa: E402

part2 = pytest.mark.xfail(reason="Part 2", strict=True, raises=NotImplementedError)


@part2
def test_plot_clock_gene_profiles():
    fig = eda.plot_clock_gene_profiles(data.load_clock_genes())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@part2
def test_pca_samples():
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(rng.normal(size=(100, 30)), columns=[f"S{i}" for i in range(30)])
    scores, ev = eda.pca_samples(expr, n_components=3)
    assert scores.shape == (30, 3) and list(scores.columns) == ["PC1", "PC2", "PC3"]
    assert (np.diff(ev.to_numpy()) <= 1e-12).all()


@part2
def test_plot_pca_and_correlation():
    meta = data.analysis_samples().head(12)
    rng = np.random.default_rng(1)
    expr = pd.DataFrame(rng.normal(size=(40, 12)), columns=meta.sample_id)
    scores, _ = eda.pca_samples(expr, n_components=2)
    for fig in (eda.plot_pca(scores, meta), eda.plot_sample_correlation(expr, meta)):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
