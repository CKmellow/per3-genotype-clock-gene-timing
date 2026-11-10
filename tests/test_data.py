"""Part 1 data-layer tests. Offline tests use only files committed in data/raw."""

import numpy as np
import pandas as pd
import pytest

from per3clock import config, data

RECOVERED = [f"GSM15264{i}" for i in range(34, 45)]


# --- metadata ---------------------------------------------------------------------------
def test_metadata_shape_and_columns(meta):
    assert len(meta) == 438
    assert list(meta.columns) == data.METADATA_COLUMNS
    assert meta["sample_id"].is_unique


def test_subjects_and_genotypes(meta):
    assert meta["subject"].nunique() == 26
    per_subject = meta.groupby("subject")["per3_genotype"].nunique()
    assert (per_subject == 1).all()
    split = meta.drop_duplicates("subject")["per3_genotype"].value_counts().to_dict()
    assert split == {"5/5": 14, "4/4": 12}


def test_no_blank_subject_or_genotype_after_recovery(meta):
    assert meta[["subject", "per3_genotype"]].notna().all().all()
    assert sorted(meta.loc[meta["is_recovered"], "sample_id"]) == sorted(RECOVERED)


def test_recovered_examples(meta):
    row = meta.set_index("sample_id").loc["GSM1526434"]
    assert (row.subject, row.condition_code, row.timepoint) == ("AF0033", "SE", "T2")
    # genotype comes from AF0033's other samples
    assert row.per3_genotype == meta.query("subject == 'AF0033' and not is_recovered")[
        "per3_genotype"
    ].iloc[0]


def test_recovered_genotype_matches_geo_soft(meta):
    soft = data.load_soft_samples()
    rec = meta.set_index("sample_id").loc[RECOVERED]
    geo_gt = soft.loc[RECOVERED, "genotype"].str.replace("\\", "/", regex=False)
    assert (geo_gt == rec["per3_genotype"]).all()


def test_conditions_timepoints(meta):
    assert set(meta["condition_code"]) == {"SE", "SR"}
    assert meta["timepoint_n"].between(1, 10).all()
    assert not meta.duplicated(["subject", "condition_code", "timepoint"]).any()
    se_only = {"AF0186", "AF0295"}
    sr_only = {"AF0079", "AF0091"}
    conds = meta.groupby("subject")["condition_code"].agg(set)
    assert set(conds[conds == {"SE"}].index) == se_only
    assert set(conds[conds == {"SR"}].index) == sr_only


@pytest.mark.parametrize(
    "s,expected", [("18:15", 18.25), ("0:15", 0.25), ("23:59", 23 + 59 / 60), ("24:00", 0.0)]
)
def test_parse_clock_time_scalar(s, expected):
    assert data.parse_clock_time(s) == pytest.approx(expected)


def test_parse_clock_time_invalid():
    with pytest.raises(ValueError):
        data.parse_clock_time(["12:00", "noon"])


def test_clock_time_column(meta):
    assert meta["clock_time_h"].between(0, 24, inclusive="left").all()
    r = meta.set_index("sample_id").loc["GSM968833"]
    assert r.clock_time == "18:15" and r.clock_time_h == pytest.approx(18.25)


def test_numeric_columns(meta):
    assert meta["hours_awake"].between(7.5, 34.5).all()
    assert set(np.unique(meta["circadian_phase_h"])) <= set(range(-15, 16, 3))


# --- clock genes ------------------------------------------------------------------------
def test_clock_genes_long(clock):
    assert set(clock["gene"]) == set(config.CLOCK_GENES)
    assert clock["sample_id"].nunique() == 438
    assert len(clock) == 438 * 8
    assert clock["expression"].notna().all()
    assert {"subject", "per3_genotype", "clock_time_h", "is_recovered"} <= set(clock.columns)


def test_clock_genes_probe_level():
    p = data.load_clock_genes(collapse=None)
    assert len(p) == 438 * 14 and p["probe_id"].nunique() == 14
    assert p.groupby("gene")["probe_id"].nunique().to_dict()["PER3"] == 3


def test_clock_gene_mean_collapse_consistent(clock):
    p = data.load_clock_genes(collapse=None)
    m = p.groupby(["sample_id", "gene"])["expression"].mean()
    c = clock.set_index(["sample_id", "gene"])["expression"]
    pd.testing.assert_series_equal(m.sort_index(), c.sort_index(), check_names=False)


def test_clock_values_look_log2(clock):
    assert clock["expression"].between(0, 20).all()


# --- annotation -------------------------------------------------------------------------
def test_probe_annotation():
    a = data.load_probe_annotation()
    assert len(a) == 45_220 and a.index.is_unique
    pmap = data.clock_probe_columns(pd.read_csv(config.CLOCK_GENES_CSV, nrows=0).columns)
    for gene, pid in pmap.values():
        assert a.loc[pid, "GENE_SYMBOL"] == gene


def test_collapse_probes_to_genes_small():
    annot = pd.DataFrame(
        {"GENE_SYMBOL": ["A", "A", "B", np.nan]}, index=pd.Index([1, 2, 3, 4], name="ID")
    )
    vals = {"s1": [1.0, 3.0, 5.0, 9.0], "s2": [2.0, 2.0, 6.0, 9.0]}
    expr = pd.DataFrame(vals, index=[1, 2, 3, 4])
    mean = data.collapse_probes_to_genes(expr, annot, "mean")
    assert mean.loc["A"].tolist() == [2.0, 2.0] and "nan" not in mean.index.astype(str)
    mx = data.collapse_probes_to_genes(expr, annot, "max_mean")
    assert mx.loc["A"].tolist() == [3.0, 2.0]
    with pytest.raises(ValueError):
        data.collapse_probes_to_genes(expr, annot, "sum")


# --- analysis set -----------------------------------------------------------------------
def test_analysis_samples_se(meta):
    s = data.analysis_samples(metadata=meta)
    assert set(s["sleep_condition"]) == {"Sleep Extension"}
    assert s["subject"].nunique() == 24 and len(s) == 221
    assert s.drop_duplicates("subject")["per3_genotype"].value_counts().to_dict() == {
        "5/5": 13,
        "4/4": 11,
    }


def test_analysis_samples_sr(meta):
    s = data.analysis_samples("Sleep Restriction", metadata=meta)
    assert s["subject"].nunique() == 24 and len(s) == 217


def test_analysis_samples_bad_condition():
    with pytest.raises(ValueError):
        data.analysis_samples("Nap")


# --- network: series matrix ---------------------------------------------------------------
@pytest.mark.network
def test_series_matrix_matches_clock_csv(meta):
    expr, chars = data.load_expression_matrix(with_characteristics=True)
    assert expr.shape == (config.N_PROBES, config.N_SAMPLES)
    assert set(expr.columns) == set(meta["sample_id"])
    wide = pd.read_csv(config.CLOCK_GENES_CSV).set_index("sample_id")
    pmap = data.clock_probe_columns(wide.columns)
    sub = expr.loc[[pid for _, pid in pmap.values()], wide.index].T
    sub.columns = list(pmap)
    assert float((sub - wide[list(pmap)]).abs().max().max()) < 1e-6
    m = meta.set_index("sample_id").loc[chars.index]
    assert (chars["title"] == m["title"]).all()
    assert (chars["circadianphase"].astype(float) == m["circadian_phase_h"]).all()
    assert (chars["timesampletaken"] == m["clock_time"]).all()
