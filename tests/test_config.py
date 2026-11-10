import numpy as np

from per3clock import config


def test_paths_are_inside_repo():
    assert (config.ROOT / "pyproject.toml").exists()
    for p in (config.RAW_DIR, config.EXTERNAL_DIR, config.FIGURES_DIR, config.TABLES_DIR):
        assert config.ROOT in p.parents


def test_gene_lists():
    assert config.GENES_PRIMARY == ("PER2", "PER3", "NR1D1")
    assert set(config.GENES_PRIMARY).isdisjoint(config.GENES_SECONDARY)
    assert config.PERIOD_H == 24


def test_set_seeds_reproducible():
    config.set_seeds()
    a = np.random.rand(3)
    config.set_seeds()
    b = np.random.rand(3)
    assert np.allclose(a, b)
