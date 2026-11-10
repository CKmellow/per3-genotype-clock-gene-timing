"""Preprocessing and quality control (Part 2).

Stage 3 must report before/after counts for every step and justify each exclusion.
"""

from __future__ import annotations

import pandas as pd


def filter_low_expression(
    expr: pd.DataFrame, min_log2: float = 6.0, min_frac: float = 0.2
) -> pd.DataFrame:
    """Keep probes/genes expressed above ``min_log2`` in at least ``min_frac`` of samples.

    Parameters
    ----------
    expr : features × samples (log2).

    Returns
    -------
    Filtered ``expr`` (same columns, subset of rows). Thresholds are placeholders.
    Part 2 chooses and justifies them (e.g. from the expression density plot).
    """
    raise NotImplementedError("Part 2: implement filter_low_expression")


def sample_qc_table(expr: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-sample QC metrics.

    Returns
    -------
    DataFrame indexed by ``sample_id`` with at least ``median_log2``, ``iqr_log2``,
    ``corr_to_subject_median`` (Pearson r with the subject's median profile) and
    ``outlier`` (bool), plus the metadata columns.
    """
    raise NotImplementedError("Part 2: implement sample_qc_table")


def apply_qc(
    meta: pd.DataFrame, expr: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the agreed sample-level QC rules.

    Returns
    -------
    ``(meta_kept, exclusions)`` where ``exclusions`` has columns
    ``sample_id, subject, sleep_condition, reason`` and is saved by stage 3 as
    ``results/tables/qc_exclusions.csv``. Must handle the flagged AF0182 SE
    T4/T5 annotation issue (see ``results/tables/sample_anomalies.csv``) and
    decide on the authors' time-series rule (see ``data/README.md``).
    """
    raise NotImplementedError("Part 2: implement apply_qc")


def qc_counts(steps: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Build the before/after table from ``[(step_name, meta_after_step), ...]``.

    Returns
    -------
    DataFrame with columns ``step, samples, subjects, subjects_4_4, subjects_5_5``,
    one row per step, saved as ``results/tables/qc_counts.csv``.
    """
    raise NotImplementedError("Part 2: implement qc_counts")
