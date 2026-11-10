import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from per3clock import config, plotting  # noqa: E402


def test_save_fig(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    plotting.set_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = plotting.save_fig(fig, "unit_test")
    assert out == tmp_path / "unit_test.png" and out.stat().st_size > 0
    plt.close(fig)


def test_save_fig_rejects_paths():
    with pytest.raises(ValueError):
        plotting.save_fig(None, "sub/dir")
