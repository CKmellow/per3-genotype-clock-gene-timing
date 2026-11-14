import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from per3clock import config, data, qc  # noqa: E402


@pytest.fixture
def toy():
    meta = data.analysis_samples().head(20)
    rng = np.random.default_rng(0)
    expr = pd.DataFrame(rng.normal(8, 1, (50, 20)), columns=meta.sample_id)
    expr.iloc[:10] = 2.0  # clearly unexpressed
    return meta, expr


@pytest.fixture
def toy_probes():
    """Probe matrix with 30 negative controls (~N(2, 0.5)) and 70 expressed probes."""
    rng = np.random.default_rng(3)
    ids = np.arange(1, 101)
    names = [f"(-)3xSLv{i}" for i in range(15)] + [f"DarkCorner{i}" for i in range(15)] + [
        f"A_23_P{i}" for i in range(70)]
    annot = pd.DataFrame({"PROBE_NAME": names, "GENE_SYMBOL": [np.nan] * 30 + [
        f"G{i}" for i in range(70)]}, index=pd.Index(ids, name="ID"))
    vals = np.vstack([rng.normal(2, 0.5, (30, 12)), rng.normal(8, 1, (70, 12))])
    expr = pd.DataFrame(vals, index=ids, columns=[f"S{i}" for i in range(12)])
    return expr, annot


@pytest.fixture(scope="module")
def main_clock_wide():
    """genes × samples matrix of the 8 clock genes, main-analysis samples."""
    clock = data.load_clock_genes()
    main = data.analysis_samples()
    clock = clock[clock.sample_id.isin(main.sample_id)]
    return clock.pivot(index="gene", columns="sample_id", values="expression"), main


# --- normalisation and background ------------------------------------------------------
def test_check_log2_normalised_detects_quantile_normalised():
    rng = np.random.default_rng(1)
    base = np.sort(rng.normal(7, 3, 500))
    qn = pd.DataFrame({f"S{i}": rng.permutation(base) for i in range(6)})
    res = qc.check_log2_normalised(qn.clip(lower=0.5))
    assert res["looks_log2"] and res["quantile_normalised"] and res["n_missing"] == 0
    raw = pd.DataFrame(2 ** rng.normal(10, 3, (500, 6)))  # linear scale, not normalised
    res = qc.check_log2_normalised(raw)
    assert not res["looks_log2"] and not res["quantile_normalised"]


def test_negative_controls_and_background(toy_probes):
    expr, annot = toy_probes
    ids = qc.negative_control_ids(annot)
    assert len(ids) == 30 and set(ids) == set(range(1, 31))
    thr = qc.background_threshold(expr, annot)
    expected = np.quantile(expr.loc[1:30].to_numpy(), 0.95) + np.log2(1.1)
    assert thr == pytest.approx(expected)
    p95 = qc.negative_control_p95(expr, annot)
    assert list(p95.index) == list(expr.columns) and p95.name == "neg_ctrl_p95"


def test_background_needs_probe_matrix(toy_probes):
    expr, annot = toy_probes
    with pytest.raises(ValueError, match="negative-control"):
        qc.background_threshold(expr.loc[31:], annot)


# --- filtering -------------------------------------------------------------------------
def test_filter_low_expression(toy):
    _, expr = toy
    out = qc.filter_low_expression(expr, min_log2=6, min_frac=0.5)
    assert len(out) == 40 and list(out.columns) == list(expr.columns)


def test_filter_low_expression_frac_boundary():
    expr = pd.DataFrame([[5, 5, 1, 1, 1], [5, 1, 1, 1, 1]], index=["a", "b"])
    assert list(qc.filter_low_expression(expr, 3, 0.4).index) == ["a"]
    assert list(qc.filter_low_expression(expr, 3, 0.2).index) == ["a", "b"]
    with pytest.raises(ValueError):
        qc.filter_low_expression(expr, 3, 1.5)


def test_defaults_keep_every_clock_gene(main_clock_wide):
    wide, _ = main_clock_wide
    kept = qc.filter_low_expression(wide)  # documented default rule
    assert set(kept.index) == set(config.CLOCK_GENES)
    # the stub's old placeholder (6.0) would have removed PER3: guard against it
    assert "PER3" not in qc.filter_low_expression(wide, 6.0, qc.DEFAULT_MIN_FRAC).index


def test_expressed_genes_pipeline(toy_probes):
    expr, annot = toy_probes
    expr.loc[31:35, ["S0", "S1"]] = 2.0  # 5 genes expressed in only 2 of 12 samples
    genes, thr = qc.expressed_genes(expr, annot, sample_ids=expr.columns, min_frac=0.5)
    assert thr == pytest.approx(qc.background_threshold(expr, annot))
    assert len(genes) == 70 and list(genes.columns) == list(expr.columns)
    genes, _ = qc.expressed_genes(expr, annot, sample_ids=["S0", "S1", "S2"], min_frac=0.5)
    assert len(genes) == 65  # G0-G4 are below background in 2 of the 3 samples


def test_filter_sensitivity(main_clock_wide):
    wide, _ = main_clock_wide
    t = qc.filter_sensitivity(wide, min_log2_values=(qc.DEFAULT_MIN_LOG2, 6.0),
                              min_frac_values=(0.2,))
    assert list(t.columns) == ["min_log2", "min_frac", "kept", "removed", "clock_genes_lost"]
    assert t.loc[0, "clock_genes_lost"] == "" and "PER3" in t.loc[1, "clock_genes_lost"]
    assert (t["kept"] + t["removed"] == len(wide)).all()


# --- sample QC -------------------------------------------------------------------------
def test_sample_qc_table(toy):
    meta, expr = toy
    t = qc.sample_qc_table(expr, meta)
    assert {"median_log2", "iqr_log2", "corr_to_subject_median", "outlier"} <= set(t.columns)
    assert len(t) == expr.shape[1]


def test_sample_qc_table_flags_planted_outlier():
    rng = np.random.default_rng(5)
    meta = data.analysis_samples()
    meta = meta[meta.subject.isin(meta.subject.unique()[:4])]
    subj_profile = {s: rng.normal(8, 2, 300) for s in meta.subject.unique()}
    expr = pd.DataFrame({sid: subj_profile[s] + rng.normal(0, 0.3, 300)
                         for sid, s in zip(meta.sample_id, meta.subject, strict=True)})
    bad = meta.sample_id.iloc[3]
    expr[bad] = rng.normal(8, 2, 300)  # unrelated profile, e.g. a swapped/failed array
    t = qc.sample_qc_table(expr, meta)
    assert t.loc[bad, "outlier"] and t["outlier"].sum() == 1
    assert "own subject" in t.loc[bad, "outlier_reason"]
    assert t.loc[bad, "subject"] == meta.set_index("sample_id").loc[bad, "subject"]


def test_sample_qc_table_single_sample_subject_and_background(toy):
    meta, expr = toy
    bg = pd.Series(np.random.default_rng(7).normal(2.5, 0.3, expr.shape[1]), index=expr.columns)
    bg.iloc[0] = 9.0
    t = qc.sample_qc_table(expr, meta, background=bg)
    lone = meta.subject.value_counts()
    lone_ids = meta[meta.subject.isin(lone[lone == 1].index)].sample_id
    assert t.loc[lone_ids, "corr_to_subject_median"].isna().all()
    assert t["high_background"].iloc[0] and t["high_background"].sum() == 1


def test_sample_qc_table_rejects_unknown_samples(toy):
    meta, expr = toy
    with pytest.raises(ValueError, match="missing from meta"):
        qc.sample_qc_table(expr.rename(columns={expr.columns[0]: "GSM0"}), meta)


def test_infer_sex_consistency():
    meta = data.analysis_samples()
    meta = meta[meta.subject.isin(["AF0004", "AF0010"])]
    xist = np.where(meta.subject == "AF0010", 14.5, 1.5)
    rps = np.where(meta.subject == "AF0010", 1.3, 12.0)
    genes = pd.DataFrame([xist, rps], index=["XIST", "RPS4Y1"], columns=meta.sample_id)
    swapped = meta.sample_id.iloc[0]
    genes.loc[:, swapped] = [14.5, 1.3]
    s = qc.infer_sex(genes, meta)
    assert set(s.subject_sex) == {"female", "male"}
    assert not s.loc[swapped, "consistent"] and s["consistent"].sum() == len(s) - 1
    with pytest.raises(ValueError, match="sex markers"):
        qc.infer_sex(genes.drop(index="XIST"), meta)


def test_clock_gene_background_check(main_clock_wide):
    wide, _ = main_clock_wide
    t = qc.clock_gene_background_check(wide, qc.DEFAULT_MIN_LOG2)
    assert list(t.index) == list(config.CLOCK_GENES)
    assert t["passes_filter"].all() and (t["frac_above_threshold"] > 0.95).all()
    assert t["mean_log2"].idxmin() == "PER3"


def test_pooled_clock_timing_expected_order(clock):
    main = data.analysis_samples()
    t = qc.pooled_clock_timing(clock[clock.sample_id.isin(main.sample_id)])
    # repressor arm (PER, NR1D) peaks in the early morning in blood
    for g in ("PER1", "PER2", "PER3", "NR1D1", "NR1D2"):
        assert 0 <= t.loc[g, "acrophase_h"] < 9, g
    assert (t["n_subjects"] == 24).all() and "p" not in t.columns
    with pytest.raises(ValueError, match="single sleep_condition"):
        qc.pooled_clock_timing(clock)


# --- sample sets and counts --------------------------------------------------------------
def test_apply_qc():
    meta = data.load_metadata()
    kept, excl = qc.apply_qc(meta)
    assert {"sample_id", "subject", "sleep_condition", "reason"} <= set(excl.columns)
    assert len(kept) + excl.sample_id.nunique() == len(meta)


def test_apply_qc_reproduces_shared_sets(meta):
    for kw, n_samples, n_subj in [({}, 221, 24), ({"strict": True}, 193, 20),
                                  ({"for_phase_analysis": True}, 214, 23)]:
        kept, excl = qc.apply_qc(meta, **kw)
        ref = data.analysis_samples(metadata=meta, **kw)
        assert list(kept.sample_id) == list(ref.sample_id)
        assert len(kept) == n_samples and kept.subject.nunique() == n_subj
        assert not set(kept.sample_id) & set(excl.sample_id)
    kept, excl = qc.apply_qc(meta, for_phase_analysis=True)
    af182 = excl[excl.subject.eq("AF0182") & excl.sleep_condition.eq("Sleep Extension")]
    assert len(af182) == 7 and af182.reason.str.contains("circadian_phase_h").all()
    sr = excl[excl.sleep_condition == "Sleep Restriction"]
    assert len(sr) == 217 and sr.reason.str.contains("sensitivity").all()


def test_apply_qc_keeps_af0182_for_clock_time(meta):
    kept, _ = qc.apply_qc(meta)
    assert "AF0182" in set(kept.subject)


def test_apply_qc_drop_outliers_is_opt_in(meta):
    with pytest.raises(ValueError, match="needs expr"):
        qc.apply_qc(meta, drop_outliers=True)
    main = data.analysis_samples(metadata=meta)
    rng = np.random.default_rng(2)
    base = rng.normal(8, 2, 200)
    expr = pd.DataFrame({s: base + rng.normal(0, 0.2, 200) for s in main.sample_id})
    bad = main.sample_id.iloc[0]
    expr[bad] = rng.normal(8, 2, 200)
    kept, excl = qc.apply_qc(meta, expr, drop_outliers=True)
    assert bad not in set(kept.sample_id)
    assert excl.set_index("sample_id").loc[bad, "reason"].startswith("QC outlier")
    assert len(kept) + len(excl) == len(meta)


def test_qc_counts():
    meta = data.load_metadata()
    t = qc.qc_counts([("raw", meta), ("SE", meta[meta.condition_code == "SE"])])
    assert list(t["step"]) == ["raw", "SE"] and t.loc[0, "samples"] == 438


def test_qc_counts_columns_and_features(meta):
    main = data.analysis_samples(metadata=meta)
    t = qc.qc_counts([("raw", meta, 43_758), ("main", main, 17_000), ("no features", main)])
    assert list(t.columns) == ["step", "samples", "subjects", "subjects_4_4", "subjects_5_5",
                               "samples_SE", "samples_SR", "features"]
    assert t.loc[0, ["subjects", "subjects_4_4", "subjects_5_5"]].tolist() == [26, 12, 14]
    assert t.loc[1, ["samples", "subjects_4_4", "subjects_5_5", "samples_SR"]].tolist() == [
        221, 11, 13, 0]
    assert t.loc[0, "features"] == 43_758 and pd.isna(t.loc[2, "features"])
    with pytest.raises(ValueError):
        qc.qc_counts([("bad",)])


# --- figures -----------------------------------------------------------------------------
def test_qc_figures(toy, toy_probes, clock):
    meta, expr = toy
    fig = qc.plot_expression_qc(expr, meta, threshold=3.0, negative_controls=expr.iloc[:10],
                                highlight=(expr.index[20],))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    fig = qc.plot_sample_qc(qc.sample_qc_table(expr, meta))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    main = data.analysis_samples()
    cl = clock[clock.sample_id.isin(main.sample_id)]
    wide = cl.pivot(index="gene", columns="sample_id", values="expression")
    sex_genes = pd.DataFrame(
        [np.full(wide.shape[1], 14.0), np.full(wide.shape[1], 1.5)],
        index=["XIST", "RPS4Y1"], columns=wide.columns)
    fig = qc.plot_biology_checks(cl, qc.pooled_clock_timing(cl), qc.infer_sex(sex_genes, main),
                                 array_median=6.7)
    assert isinstance(fig, plt.Figure) and len(fig.axes) == 3
    plt.close(fig)


# --- network: real series matrix ---------------------------------------------------------
@pytest.mark.network
def test_default_threshold_matches_background():
    expr = data.load_expression_matrix(dtype="float32")
    annot = data.load_probe_annotation()
    main = data.analysis_samples()
    thr = qc.background_threshold(expr[main.sample_id], annot)
    assert thr == pytest.approx(qc.DEFAULT_MIN_LOG2, abs=0.005)
    chk = qc.check_log2_normalised(expr)
    assert chk["looks_log2"] and chk["quantile_normalised"] and chk["n_missing"] == 0
    genes = data.collapse_probes_to_genes(expr, annot)[main.sample_id]
    kept = qc.filter_low_expression(genes, thr)
    assert len(kept) == 17_528 and set(config.CLOCK_GENES) <= set(kept.index)
    sex = qc.infer_sex(genes, main)
    assert sex["consistent"].all()
