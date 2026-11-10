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


def test_clock_gene_mean_collapse_consistent():
    p = data.load_clock_genes(collapse=None)
    m = p.groupby(["sample_id", "gene"])["expression"].mean()
    c = data.load_clock_genes(collapse="mean").set_index(["sample_id", "gene"])["expression"]
    pd.testing.assert_series_equal(m.sort_index(), c.sort_index(), check_names=False)


# --- probe selection ----------------------------------------------------------------------
SELECTED = {"ARNTL": 9332, "CLOCK": 26609, "CRY1": 22736, "NR1D1": 33493, "NR1D2": 38683,
            "PER1": 3881, "PER2": 16821, "PER3": 25341}


def test_default_clock_genes_use_one_probe_per_gene(clock):
    assert clock.groupby("gene")["probe_id"].unique().map(lambda a: a[0]).to_dict() == SELECTED
    p = data.load_clock_genes(collapse=None).set_index(["sample_id", "probe_id"])["expression"]
    c = clock.set_index(["sample_id", "probe_id"])["expression"]
    pd.testing.assert_series_equal(c.sort_index(), p.loc[c.index].sort_index(), check_names=False)


def test_per3_low_probes_not_selected():
    sel = data.clock_probe_selection().set_index("probe_id")
    assert sel.loc[25341, "selected"] and not sel.loc[[17478, 43131], "selected"].any()
    assert sel.groupby("gene")["selected"].sum().eq(1).all()
    # selected = highest mean within each gene
    best = sel.reset_index().loc[lambda d: d.groupby("gene")["mean_log2"].idxmax(), "probe_id"]
    assert set(best) == set(SELECTED.values())


def test_select_probes_toy_and_tie_break():
    annot = pd.DataFrame({"GENE_SYMBOL": ["A", "A", "B", "B", np.nan]},
                         index=pd.Index([1, 2, 3, 4, 5], name="ID"))
    expr = pd.DataFrame({"s1": [1.0, 3.0, 5.0, 5.0, 99.0], "s2": [1.0, 3.0, 5.0, 5.0, 99.0]},
                        index=[1, 2, 3, 4, 5])
    sel = data.select_probes(expr, annot)
    assert sel["probe_id"].to_dict() == {"A": 2, "B": 3}  # tie -> lowest ID; no-symbol ignored
    assert sel.loc["A", "n_candidates"] == 2
    with pytest.raises(ValueError):
        data.select_probes(expr, annot, method="most_rhythmic")


def test_select_probes_is_outcome_blind():
    """Selection depends only on per-probe means: shuffling sample order (i.e. any
    relabelling of time, genotype or condition) must not change it."""
    rng = np.random.default_rng(0)
    annot = pd.DataFrame({"GENE_SYMBOL": list("AABBBC")}, index=range(6))
    expr = pd.DataFrame(rng.normal(6, 1, (6, 40)), index=range(6))
    a = data.select_probes(expr, annot)
    b = data.select_probes(expr.sample(frac=1, axis=1, random_state=1), annot)
    pd.testing.assert_frame_equal(a, b)


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
    mx = data.collapse_probes_to_genes(expr, annot)  # default = max_mean
    assert mx.loc["A"].tolist() == [3.0, 2.0] and list(mx.index) == ["A", "B"]
    with pytest.raises(ValueError):
        data.collapse_probes_to_genes(expr, annot, "sum")


# --- analysis set -----------------------------------------------------------------------
def _summary(s):
    gt = s.drop_duplicates("subject")["per3_genotype"].value_counts().to_dict()
    return s["subject"].nunique(), gt.get("4/4", 0), gt.get("5/5", 0), len(s)


def test_analysis_samples_default(meta):
    s = data.analysis_samples(metadata=meta)
    assert set(s["sleep_condition"]) == {"Sleep Extension"}
    assert _summary(s) == (24, 11, 13, 221)
    assert "AF0182" in set(s["subject"])  # real clock times -> kept for clock-time analyses


def test_analysis_samples_strict(meta):
    s, excl = data.analysis_samples(metadata=meta, strict=True, return_exclusions=True)
    assert _summary(s) == (20, 9, 11, 193)
    strict_out = excl[excl["reason"].str.contains("authors' rule")]["subject"]
    assert set(strict_out) == {"AF0112", "AF0182", "AF0186", "AF0267"}


def test_analysis_samples_for_phase_analysis(meta):
    s, excl = data.analysis_samples(metadata=meta, for_phase_analysis=True,
                                    return_exclusions=True)
    assert _summary(s) == (23, 11, 12, 214)
    assert "AF0182" not in set(s["subject"])
    assert excl.set_index("subject").loc["AF0182", "reason"].startswith("circadian_phase_h")
    both = data.analysis_samples(metadata=meta, strict=True, for_phase_analysis=True)
    assert _summary(both) == (20, 9, 11, 193)


def test_phase_exclusion_only_where_labels_are_bad(meta):
    # AF0182's Sleep Restriction phase labels are consistent, so it stays there
    sr = data.analysis_samples("Sleep Restriction", metadata=meta, for_phase_analysis=True)
    assert "AF0182" in set(sr["subject"])


def test_analysis_exclusions_table(meta):
    s, excl = data.analysis_samples(metadata=meta, return_exclusions=True)
    assert list(excl.columns) == ["subject", "per3_genotype", "sleep_condition", "n_timepoints",
                                  "missing_timepoints", "reason"]
    assert set(excl["subject"]) == {"AF0079", "AF0091"}  # no SE samples at all
    assert set(s["subject"]).isdisjoint(excl["subject"])
    assert s["subject"].nunique() + len(excl) == 26


def test_analysis_samples_sr(meta):
    s = data.analysis_samples("Sleep Restriction", metadata=meta)
    assert _summary(s) == (24, 11, 13, 217)
    assert _summary(data.analysis_samples("Sleep Restriction", metadata=meta, strict=True)) == (
        20, 9, 11, 190)


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
