"""Exploratory data analysis helpers (Part 2). Each returns a matplotlib Figure.

Save every figure with ``per3clock.plotting.save_fig(fig, name)``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from . import config


def plot_clock_gene_profiles(
    clock_long: pd.DataFrame,
    genes: tuple[str, ...] = config.GENES_PRIMARY,
    time_col: str = "clock_time_h",
) -> plt.Figure:
    """Expression vs time for each gene (one panel per gene), coloured by genotype.

    Shows individual subjects (thin lines) and the genotype mean (thick line).
    Suggested name: ``eda_clock_gene_profiles``.
    """
    raise NotImplementedError("Part 2: implement plot_clock_gene_profiles")


def pca_samples(
    expr: pd.DataFrame, n_components: int = 5
) -> tuple[pd.DataFrame, pd.Series]:
    """PCA of samples on (filtered, centred) features.

    Returns
    -------
    ``(scores, explained)``: ``scores`` indexed by ``sample_id`` with columns
    ``PC1..PCk``; ``explained`` = explained-variance ratio indexed by ``PC1..PCk``.
    """
    raise NotImplementedError("Part 2: implement pca_samples")


def plot_pca(scores: pd.DataFrame, meta: pd.DataFrame, color_by: str = "subject") -> plt.Figure:
    """Scatter PC1 vs PC2 coloured by a metadata column. Name: ``eda_pca_<color_by>``."""
    raise NotImplementedError("Part 2: implement plot_pca")


def plot_sample_correlation(expr: pd.DataFrame, meta: pd.DataFrame) -> plt.Figure:
    """Sample–sample correlation heatmap ordered by subject/condition/timepoint.

    Name: ``eda_sample_correlation``.
    """
    raise NotImplementedError("Part 2: implement plot_sample_correlation")
