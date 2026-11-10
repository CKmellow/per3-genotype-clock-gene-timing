import numpy as np
import pandas as pd
import pytest

from per3clock import data, qc

part2 = pytest.mark.xfail(reason="Part 2", strict=True, raises=NotImplementedError)


@pytest.fixture
def toy():
    meta = data.analysis_samples().head(20)
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(rng.normal(8, 1, (50, 20)), columns=meta.sample_id)
    expr.iloc[:10] = 2.0  # clearly unexpressed
    return meta, expr


@part2
def test_filter_low_expression(toy):
    _, expr = toy
    out = qc.filter_low_expression(expr, min_log2=6, min_frac=0.5)
    assert len(out) == 40 and list(out.columns) == list(expr.columns)


@part2
def test_sample_qc_table(toy):
    meta, expr = toy
    t = qc.sample_qc_table(expr, meta)
    assert {"median_log2", "iqr_log2", "corr_to_subject_median", "outlier"} <= set(t.columns)
    assert len(t) == expr.shape[1]


@part2
def test_apply_qc():
    meta = data.load_metadata()
    kept, excl = qc.apply_qc(meta)
    assert {"sample_id", "subject", "sleep_condition", "reason"} <= set(excl.columns)
    assert len(kept) + excl.sample_id.nunique() == len(meta)


@part2
def test_qc_counts():
    meta = data.load_metadata()
    t = qc.qc_counts([("raw", meta), ("SE", meta[meta.condition_code == "SE"])])
    assert list(t["step"]) == ["raw", "SE"] and t.loc[0, "samples"] == 438
