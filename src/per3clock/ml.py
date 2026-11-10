"""Predict internal circadian phase from expression (Part 5).

Target: ``circadian_phase_h`` encoded as (sin, cos). Evaluation: leave-one-subject-
out CV (``GroupKFold``/``LeaveOneGroupOut`` with ``groups = subject``), metric =
median absolute circular error in hours. All preprocessing (scaling, feature
selection) must happen **inside** the CV folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def encode_phase(phase_h, period: float = config.PERIOD_H) -> np.ndarray:
    """Hours → array of shape (n, 2) with columns ``[sin, cos]`` of ``2π·h/period``."""
    raise NotImplementedError("Part 5: implement encode_phase")


def decode_phase(sincos: np.ndarray, period: float = config.PERIOD_H) -> np.ndarray:
    """Inverse of :func:`encode_phase` → hours in ``[0, period)`` (via ``arctan2``)."""
    raise NotImplementedError("Part 5: implement decode_phase")


def circular_abs_error_h(true_h, pred_h, period: float = config.PERIOD_H) -> np.ndarray:
    """Absolute circular error in hours, in ``[0, period/2]``. ``(23, 1) → 2``."""
    raise NotImplementedError("Part 5: implement circular_abs_error_h")


def loso_cv_predict(model, X: pd.DataFrame, phase_h, groups, period: float = config.PERIOD_H):
    """Leave-one-subject-out predictions of phase (hours) for every sample.

    ``model`` is an sklearn estimator/pipeline that fits the 2-column sin/cos target
    (e.g. ``make_pipeline(StandardScaler(), Ridge())``). It is cloned for each fold.
    """
    raise NotImplementedError("Part 5: implement loso_cv_predict")


def evaluate_phase_models(
    X: pd.DataFrame, meta: pd.DataFrame, period: float = config.PERIOD_H
) -> pd.DataFrame:
    """Compare the model with baselines under LOSO CV.

    Baselines: (a) predict the circular mean phase of the training fold;
    (b) predict phase from ``clock_time_h`` only. Optional: model + genotype.

    Returns
    -------
    DataFrame with columns ``model, median_abs_error_h, mean_abs_error_h,
    n_samples, n_subjects``, saved as ``results/tables/ml_phase_prediction.csv``.
    """
    raise NotImplementedError("Part 5: implement evaluate_phase_models")
