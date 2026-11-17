"""Predict internal circadian phase from expression (Part 5).

Target: ``circadian_phase_h`` encoded as (sin, cos). Evaluation: leave-one-subject-
out CV (``LeaveOneGroupOut`` with ``groups = subject``), metric = median absolute
circular error in hours. All preprocessing (scaling, feature selection, choice of the
ridge penalty) happens **inside** the CV folds: each fold's pipeline is cloned and fitted
on the training subjects only.

Models compared on the same folds (:func:`default_phase_models`):

* ``baseline: circular mean phase``: ignores the sample and predicts the training
  fold's circular mean phase.
* ``baseline: clock time only``: phase = clock time − the training fold's circular
  mean offset (clock time − phase). Strong by design: the protocol ties clock time to
  phase, so this model is only wrong by each person's own offset.
* ``genes``: ridge regression from expression to (sin, cos) of phase.
* ``clock time + genes``: the same, with (sin, cos) of clock time as two extra features.
  Comparing it with the clock-time baseline asks whether expression adds information.
* ``genes + PER3 genotype`` (optional): genotype as a 0/1 feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

CLOCK_FEATURES: tuple[str, str] = ("clock_sin", "clock_cos")
RIDGE_ALPHAS: np.ndarray = np.logspace(-2, 4, 25)


# ---------------------------------------------------------------------------------------
# Phase encoding and circular error
# ---------------------------------------------------------------------------------------
def encode_phase(phase_h, period: float = config.PERIOD_H) -> np.ndarray:
    """Hours → array of shape (n, 2) with columns ``[sin, cos]`` of ``2π·h/period``."""
    a = 2 * np.pi * np.asarray(phase_h, dtype=float).ravel() / period
    return np.column_stack([np.sin(a), np.cos(a)])


def decode_phase(sincos: np.ndarray, period: float = config.PERIOD_H) -> np.ndarray:
    """Inverse of :func:`encode_phase` → hours in ``[0, period)`` (via ``arctan2``)."""
    sc = np.asarray(sincos, dtype=float).reshape(-1, 2)
    h = np.mod(np.arctan2(sc[:, 0], sc[:, 1]) * period / (2 * np.pi), period)
    h[np.isclose(h, period)] = 0.0  # mod can return period - ε for angles just below 0
    return h


def circular_abs_error_h(true_h, pred_h, period: float = config.PERIOD_H) -> np.ndarray:
    """Absolute circular error in hours, in ``[0, period/2]``. ``(23, 1) → 2``."""
    d = np.mod(np.asarray(pred_h, dtype=float) - np.asarray(true_h, dtype=float), period)
    return np.minimum(d, period - d)


# ---------------------------------------------------------------------------------------
# Estimators (sklearn-compatible, fitted on a 2-column sin/cos target)
# ---------------------------------------------------------------------------------------
class MeanPhaseBaseline(BaseEstimator):
    """Predicts the circular mean of the training targets for every sample."""

    def fit(self, X, Y):
        m = np.asarray(Y, dtype=float).mean(axis=0)
        self.mean_ = m / (np.linalg.norm(m) or 1.0)
        return self

    def predict(self, X):
        return np.tile(self.mean_, (len(X), 1))


class ClockOffsetBaseline(BaseEstimator):
    """Phase = clock time − circular mean offset learned on the training fold.

    ``X`` must contain the columns :data:`CLOCK_FEATURES` (sin, cos of clock time).
    """

    def __init__(self, columns: tuple[str, str] = CLOCK_FEATURES):
        self.columns = columns

    def _clock_angle(self, X) -> np.ndarray:
        c = np.asarray(pd.DataFrame(X)[list(self.columns)], dtype=float)
        return np.arctan2(c[:, 0], c[:, 1])

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=float)
        diff = np.arctan2(Y[:, 0], Y[:, 1]) - self._clock_angle(X)
        self.offset_ = float(np.arctan2(np.sin(diff).mean(), np.cos(diff).mean()))
        return self

    def predict(self, X):
        a = self._clock_angle(X) + self.offset_
        return np.column_stack([np.sin(a), np.cos(a)])


class TopRhythmicFeatures(BaseEstimator, TransformerMixin):
    """Keep the ``k`` features best explained by the (sin, cos) phase target.

    Score = R² of regressing each feature on sin and cos of the **training** phase (a
    cosinor fit on the training fold), so the choice never sees the test subject. The
    first ``keep_first`` columns (e.g. clock time) are always kept.
    """

    def __init__(self, k: int = 50, keep_first: int = 0):
        self.k = k
        self.keep_first = keep_first

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        cand = X[:, self.keep_first:]
        Xc = cand - cand.mean(axis=0)
        D = np.column_stack([np.ones(len(Y)), Y])
        beta, *_ = np.linalg.lstsq(D, cand, rcond=None)
        ss_res = ((cand - D @ beta) ** 2).sum(axis=0)
        ss_tot = (Xc**2).sum(axis=0)
        r2 = np.where(ss_tot > 0, 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1), 0.0)
        k = min(self.k, cand.shape[1])
        top = np.argsort(-r2, kind="stable")[:k] + self.keep_first
        self.scores_ = r2
        self.support_ = np.concatenate([np.arange(self.keep_first), np.sort(top)])
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]


def ridge_pipeline(k: int | None = None, keep_first: int = 0):
    """StandardScaler → (optional in-fold top-``k`` selection) → RidgeCV."""
    steps = [StandardScaler()]
    if k is not None:
        steps = [TopRhythmicFeatures(k=k, keep_first=keep_first), StandardScaler()]
    return make_pipeline(*steps, RidgeCV(alphas=RIDGE_ALPHAS))


# ---------------------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------------------
def loso_cv_predict(model, X: pd.DataFrame, phase_h, groups, period: float = config.PERIOD_H):
    """Leave-one-subject-out predictions of phase (hours) for every sample.

    ``model`` is an sklearn estimator/pipeline that fits the 2-column sin/cos target
    (e.g. ``make_pipeline(StandardScaler(), Ridge())``). It is cloned for each fold.
    """
    Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    Y = encode_phase(phase_h, period)
    groups = np.asarray(groups)
    pred = np.full(len(Y), np.nan)
    for train, test in LeaveOneGroupOut().split(Xdf, Y, groups):
        m = clone(model).fit(Xdf.iloc[train], Y[train])
        pred[test] = decode_phase(m.predict(Xdf.iloc[test]), period)
    return pred


def clock_features(meta: pd.DataFrame, period: float = config.PERIOD_H) -> pd.DataFrame:
    """(sin, cos) of ``clock_time_h`` as a DataFrame indexed like ``meta``."""
    enc = encode_phase(meta["clock_time_h"].to_numpy(), period)
    return pd.DataFrame(enc, index=meta.index, columns=list(CLOCK_FEATURES))


def genotype_feature(meta: pd.DataFrame) -> pd.DataFrame:
    """``per3_55`` = 1 for 5/5, 0 for 4/4."""
    return (meta[["per3_genotype"]] == "5/5").astype(float).set_axis(["per3_55"], axis=1)


def default_phase_models(
    X: pd.DataFrame, meta: pd.DataFrame, *, k: int | None = None,
    with_genotype: bool = True, label: str = "genes",
) -> dict[str, tuple[object, pd.DataFrame]]:
    """``{model name: (estimator, feature matrix)}`` for :func:`phase_predictions`.

    ``X`` (samples × genes) must be row-aligned with ``meta``. ``k`` turns on in-fold
    selection of the top-``k`` rhythmic genes (use it for genome-wide ``X``).
    """
    Xg = pd.DataFrame(np.asarray(X, dtype=float), index=meta.index, columns=X.columns)
    clock = clock_features(meta)
    models = {
        "baseline: circular mean phase": (MeanPhaseBaseline(), clock),
        "baseline: clock time only": (ClockOffsetBaseline(), clock),
        label: (ridge_pipeline(k), Xg),
        f"clock time + {label}": (ridge_pipeline(k, keep_first=2), pd.concat([clock, Xg], axis=1)),
    }
    if with_genotype:
        models[f"{label} + PER3 genotype"] = (
            ridge_pipeline(k, keep_first=1), pd.concat([genotype_feature(meta), Xg], axis=1))
    return models


def phase_predictions(
    models: dict[str, tuple[object, pd.DataFrame]], meta: pd.DataFrame,
    period: float = config.PERIOD_H,
) -> pd.DataFrame:
    """Long table of LOSO predictions: one row per sample × model.

    Columns ``sample_id, subject, per3_genotype, model, true_phase_h, pred_phase_h,
    abs_error_h`` (phases in ``[0, period)``).
    """
    true_h = np.mod(meta["circadian_phase_h"].to_numpy(dtype=float), period)
    rows = []
    for name, (model, Xm) in models.items():
        pred = loso_cv_predict(model, Xm, true_h, meta["subject"].to_numpy(), period)
        rows.append(pd.DataFrame({
            "sample_id": meta["sample_id"].to_numpy(),
            "subject": meta["subject"].to_numpy(),
            "per3_genotype": meta["per3_genotype"].to_numpy(),
            "model": name,
            "true_phase_h": true_h,
            "pred_phase_h": pred,
            "abs_error_h": circular_abs_error_h(true_h, pred, period),
        }))
    return pd.concat(rows, ignore_index=True)


def _bootstrap_median_ci(err: pd.Series, subjects: pd.Series, n_boot: int, rng) -> tuple:
    by_subj = [g.to_numpy() for _, g in err.groupby(subjects.to_numpy())]
    meds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(by_subj), len(by_subj))
        meds[b] = np.median(np.concatenate([by_subj[i] for i in pick]))
    return tuple(np.percentile(meds, [2.5, 97.5]))


def summarise_predictions(pred: pd.DataFrame, n_boot: int = 2000,
                          seed: int = config.SEED) -> pd.DataFrame:
    """One row per model: median/mean abs circular error, subject-bootstrap 95% CI.

    The CI resamples **subjects** (not samples), because samples within a subject are
    not independent.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name, g in pred.groupby("model", sort=False):
        lo, hi = _bootstrap_median_ci(g["abs_error_h"], g["subject"], n_boot, rng)
        rows.append({
            "model": name,
            "median_abs_error_h": float(g["abs_error_h"].median()),
            "median_ci95_low_h": float(lo),
            "median_ci95_high_h": float(hi),
            "mean_abs_error_h": float(g["abs_error_h"].mean()),
            "pct_within_3h": float((g["abs_error_h"] <= 3).mean() * 100),
            "n_samples": int(len(g)),
            "n_subjects": int(g["subject"].nunique()),
        })
    return pd.DataFrame(rows)


def paired_subject_comparison(pred: pd.DataFrame, model_a: str, model_b: str,
                              n_boot: int = 5000, seed: int = config.SEED) -> dict:
    """Paired comparison of two models across subjects on the same LOSO folds.

    Per subject, take each model's median absolute error; ``diff = b − a`` (negative
    means ``b`` is better). Reports the mean and median difference, a subject-bootstrap
    95% CI of the mean difference, the Wilcoxon signed-rank p-value and how many
    subjects improved.
    """
    per = (pred[pred["model"].isin([model_a, model_b])]
           .groupby(["subject", "model"])["abs_error_h"].median().unstack())
    d = (per[model_b] - per[model_a]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    nz = d[d != 0]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 2 else float("nan")
    return {
        "model_a": model_a, "model_b": model_b, "n_subjects": int(len(d)),
        "mean_diff_h": float(d.mean()), "median_diff_h": float(np.median(d)),
        "ci95_low_h": float(np.percentile(boots, 2.5)),
        "ci95_high_h": float(np.percentile(boots, 97.5)),
        "wilcoxon_p": p, "n_better": int((d < 0).sum()), "n_worse": int((d > 0).sum()),
        "n_tied": int((d == 0).sum()),
    }


def evaluate_phase_models(
    X: pd.DataFrame, meta: pd.DataFrame, period: float = config.PERIOD_H
) -> pd.DataFrame:
    """Compare the model with baselines under LOSO CV.

    Baselines: (a) predict the circular mean phase of the training fold;
    (b) predict phase from ``clock_time_h`` only. Optional: model + genotype.

    ``X`` is samples × features (e.g. the clock genes), row-aligned with ``meta``
    (same order). For genome-wide ``X`` use :func:`default_phase_models` with ``k``.

    Returns
    -------
    DataFrame with columns ``model, median_abs_error_h, mean_abs_error_h,
    n_samples, n_subjects`` (plus a 95% CI and % within 3 h), saved as
    ``results/tables/ml_phase_prediction.csv``.
    """
    if len(X) != len(meta):
        raise ValueError("X and meta must have the same rows in the same order")
    meta = meta.reset_index(drop=True)
    pred = phase_predictions(default_phase_models(X, meta), meta, period)
    return summarise_predictions(pred)


def within_subject_centre(X: pd.DataFrame, groups) -> pd.DataFrame:
    """Subtract each subject's own mean expression from that subject's samples.

    Removes stable between-person differences (Stage 4: subject explains ~77 % of the
    leading PCs). It uses **no labels**, only the subject's own expression, so it does
    not leak the target. But it assumes the subject's whole sampled series is
    available when predicting, which is true for this protocol (10 draws over ~30 h)
    and not for a single blood draw.
    """
    g = np.asarray(groups)
    return X - X.groupby(g).transform("mean").to_numpy()


def fold_selection_frequency(X: pd.DataFrame, phase_h, groups, k: int = 50,
                             period: float = config.PERIOD_H) -> pd.DataFrame:
    """How often each gene is picked by :class:`TopRhythmicFeatures` across LOSO folds.

    Returns ``gene, n_folds_selected, frac_folds, mean_r2`` for genes selected at least
    once, sorted by frequency (then mean training R²).
    """
    Y = encode_phase(phase_h, period)
    groups = np.asarray(groups)
    counts = pd.Series(0, index=X.columns, dtype=int)
    r2 = pd.Series(0.0, index=X.columns)
    n = 0
    for train, _ in LeaveOneGroupOut().split(X, Y, groups):
        sel = TopRhythmicFeatures(k=k).fit(X.iloc[train], Y[train])
        counts.iloc[sel.support_] += 1
        r2 += sel.scores_
        n += 1
    out = pd.DataFrame({"n_folds_selected": counts, "frac_folds": counts / n,
                        "mean_r2": r2 / n})
    out = out[out["n_folds_selected"] > 0].sort_values(
        ["n_folds_selected", "mean_r2"], ascending=False)
    return out.rename_axis("gene").reset_index()


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------
def plot_predicted_vs_true(pred: pd.DataFrame, models: list[str], *, ncols: int = 2,
                           period: float = config.PERIOD_H, seed: int = config.SEED):
    """Scatter of predicted vs true phase per model (true phase jittered ±0.4 h)."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.3 * nrows), squeeze=False)
    colours = {"4/4": "#3b6fb6", "5/5": "#d9822b"}
    for ax, name in zip(axes.ravel(), models, strict=False):
        g = pred[pred["model"] == name]
        x = g["true_phase_h"].to_numpy() + rng.uniform(-0.4, 0.4, len(g))
        for gt, col in colours.items():
            m = (g["per3_genotype"] == gt).to_numpy()
            ax.scatter(x[m], g["pred_phase_h"].to_numpy()[m], s=12, alpha=0.6, color=col,
                       label=f"PER3 {gt}")
        ax.plot([0, period], [0, period], color="0.3", lw=1)
        for off in (-period, period):
            ax.plot([0, period], [off, period + off], color="0.3", lw=1)
        ax.set(xlim=(-0.5, period), ylim=(0, period), xticks=range(0, 25, 6),
               yticks=range(0, 25, 6), xlabel="true circadian phase (h, mod 24)",
               ylabel="predicted phase (h)")
        med = g["abs_error_h"].median()
        ax.set_title(f"{name}\nmedian |error| = {med:.2f} h", fontsize=10)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_error_by_model(summary: pd.DataFrame, pred: pd.DataFrame, baseline: str):
    """Left: median error with subject-bootstrap 95% CI. Right: per-subject median
    error of each model minus ``baseline`` (negative = better than the baseline)."""
    import matplotlib.pyplot as plt

    s = summary.iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.45 * len(s) + 1.8),
                                 gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(len(s))
    is_base = s["model"].str.startswith("baseline").to_numpy()
    a1.errorbar(s["median_abs_error_h"], y,
                xerr=[s["median_abs_error_h"] - s["median_ci95_low_h"],
                      s["median_ci95_high_h"] - s["median_abs_error_h"]],
                fmt="none", ecolor="0.5", capsize=3)
    a1.scatter(s["median_abs_error_h"], y, c=np.where(is_base, "0.4", "#3b6fb6"), zorder=3)
    a1.set_yticks(y, s["model"])
    a1.axvline(1.5, ls=":", color="0.6")
    a1.set(xlabel="median absolute circular error (h), 95% CI", xlim=(0, None),
           title="LOSO error by model (dotted: half a 3-h bin)")

    per = pred.groupby(["model", "subject"])["abs_error_h"].median().unstack(0)
    diff = per.sub(per[baseline], axis=0)
    rng = np.random.default_rng(config.SEED)
    for i, name in enumerate(s["model"]):
        d = diff[name].to_numpy()
        a2.scatter(d, i + rng.uniform(-0.15, 0.15, len(d)), s=10, alpha=0.6, color="#3b6fb6")
        a2.plot([np.median(d)] * 2, [i - 0.3, i + 0.3], color="k", lw=2)
    a2.axvline(0, color="0.4", lw=1)
    a2.set_yticks(y, [""] * len(y))
    a2.set(xlabel=f"per-subject median error minus '{baseline}' (h)",
           title="Paired across subjects (bar = median)")
    fig.tight_layout()
    return fig
