import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

from per3clock import interpret

# ---------------------------------------------------------------------------------------
# Gene lists
# ---------------------------------------------------------------------------------------
SCAN = pd.DataFrame(
    {"q": [0.001, 0.01, 0.2, 0.001, 0.04],
     "amplitude": [0.3, 0.25, 0.5, 0.1, 0.4],
     "acrophase_h": [23.0, 9.0, 9.0, 1.0, 2.0]},
    index=pd.Index(["A", "B", "C", "D", "E"], name="gene"),
)


def test_select_rhythmic_genes_filters_and_wraps():
    assert interpret.select_rhythmic_genes(SCAN) == ["A", "B", "E"]
    assert interpret.select_rhythmic_genes(SCAN, min_amplitude=0.0) == ["A", "B", "D", "E"]
    assert interpret.select_rhythmic_genes(SCAN, phase_window=(6, 12)) == ["B"]
    # window crossing midnight
    assert interpret.select_rhythmic_genes(SCAN, phase_window=(22, 3)) == ["A", "E"]
    # also accepts a 'gene' column
    assert interpret.select_rhythmic_genes(SCAN.reset_index()) == ["A", "B", "E"]


# ---------------------------------------------------------------------------------------
# Enrichment (offline: gene sets passed as a dict)
# ---------------------------------------------------------------------------------------
BACKGROUND = [f"G{i}" for i in range(100)]
LIB = {
    "clock": [f"G{i}" for i in range(10)] + ["NOT_ON_ARRAY"],  # 10 in background
    "noise": [f"G{i}" for i in range(50, 60)],
    "tiny": ["G0", "G1"],  # below min_size
}


def test_run_enrichment_offline_matches_hypergeometric():
    genes = [f"G{i}" for i in range(8)] + ["G50", "UNKNOWN"]  # 9 in background
    r = interpret.run_enrichment(genes, {"toy": LIB}, BACKGROUND)
    assert {"gene_set", "term", "overlap", "p", "q", "genes"} <= set(r.columns)
    assert list(r["term"]) == ["clock", "noise"]  # 'tiny' dropped (min_size=5)
    top = r.iloc[0]
    assert top["overlap"] == "8/10" and top["n_genes"] == 9 and top["n_background"] == 100
    assert top["p"] == pytest.approx(hypergeom.sf(7, 100, 10, 9))
    assert top["genes"] == ";".join(sorted(f"G{i}" for i in range(8)))
    # BH within library, sorted by q
    assert r["q"].is_monotonic_increasing and (r["q"] >= r["p"]).all()
    assert top["fold_enrichment"] == pytest.approx((8 / 9) / (10 / 100))


def test_run_enrichment_no_overlap_returns_empty_frame():
    r = interpret.run_enrichment(["UNKNOWN"], {"toy": LIB}, BACKGROUND)
    assert r.empty and {"term", "p", "q"} <= set(r.columns)


def test_gmt_roundtrip(tmp_path):
    interpret._write_gmt(LIB, tmp_path / "x.gmt")
    assert interpret._read_gmt(tmp_path / "x.gmt") == LIB


# ---------------------------------------------------------------------------------------
# STRING (offline: canned API responses)
# ---------------------------------------------------------------------------------------
IDS_TSV = (
    "queryIndex\tqueryItem\tstringId\tncbiTaxonId\ttaxonName\tpreferredName\tannotation\n"
    "0\tPER2\t9606.P2\t9606\tHomo sapiens\tPER2\tx\n"
    "1\tARNTL\t9606.B1\t9606\tHomo sapiens\tBMAL1\tx\n"
    "2\tNR1D1\t9606.N1\t9606\tHomo sapiens\tNR1D1\tx\n"
)
NET_TSV = (
    "stringId_A\tstringId_B\tpreferredName_A\tpreferredName_B\tncbiTaxonId\tscore\tnscore"
    "\tfscore\tpscore\tascore\tescore\tdscore\ttscore\n"
    "9606.B1\t9606.P2\tBMAL1\tPER2\t9606\t0.999\t0\t0\t0\t0.1\t0.99\t0.9\t0.8\n"
    "9606.P2\t9606.B1\tPER2\tBMAL1\t9606\t0.999\t0\t0\t0\t0.1\t0.99\t0.9\t0.8\n"
    "9606.N1\t9606.P2\tNR1D1\tPER2\t9606\t0.95\t0\t0\t0\t0.1\t0.6\t0\t0.9\n"
)


def test_fetch_string_network_parses_and_maps_names(monkeypatch, tmp_path):
    def fake_post(method, params, cache_path, refresh):
        assert "caller_identity" not in params  # added inside _string_post
        return IDS_TSV if method == "get_string_ids" else NET_TSV

    monkeypatch.setattr(interpret, "_string_post", fake_post)
    e = interpret.fetch_string_network(["PER2", "ARNTL", "NR1D1"], cache_dir=tmp_path)
    assert list(e.columns[:3]) == ["source", "target", "score"]
    assert len(e) == 2  # duplicate A-B / B-A collapsed
    pairs = {frozenset(p) for p in zip(e.source, e.target, strict=True)}
    assert pairs == {frozenset({"ARNTL", "PER2"}), frozenset({"NR1D1", "PER2"})}  # our symbols
    assert e["score"].between(0, 1).all()


# ---------------------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------------------
def test_build_network_and_centrality():
    edges = pd.DataFrame(
        {"source": ["NR1D1", "NR1D1", "PER2"], "target": ["PER2", "PER3", "PER3"],
         "score": [0.99, 0.98, 0.9]}
    )
    G = interpret.build_network(edges)
    assert isinstance(G, nx.Graph) and G.number_of_edges() == 3
    assert G["NR1D1"]["PER2"]["weight"] == pytest.approx(0.99)
    c = interpret.network_centrality(G)
    assert {"degree", "betweenness", "closeness"} <= set(c.columns) and len(c) == 3


def test_centrality_star_graph():
    edges = pd.DataFrame({"source": ["HUB"] * 4, "target": list("ABCD"), "score": [0.9] * 4})
    c = interpret.network_centrality(interpret.build_network(edges))
    assert c.index[0] == "HUB" and c.loc["HUB", "degree"] == 4
    assert c.loc["HUB", "betweenness"] == pytest.approx(1.0)
    assert c.loc["A", "betweenness"] == 0 and "community" in c


# ---------------------------------------------------------------------------------------
# Network-wide genotype shift
# ---------------------------------------------------------------------------------------
def _fake_fits(shift_h: float, n_per_group: int = 12, n_genes: int = 6, seed: int = 1):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(2 * n_per_group):
        geno = "5/5" if s < n_per_group else "4/4"
        subj_offset = rng.normal(0, 0.5)
        for g in range(n_genes):
            base = 2.0 * g
            ph = base + subj_offset + rng.normal(0, 0.5) + (shift_h if geno == "5/5" else 0)
            rows.append({"subject": f"S{s}", "per3_genotype": geno, "gene": f"G{g}",
                         "acrophase_h": ph % 24})
    return pd.DataFrame(rows)


def test_network_shift_test_detects_consistent_shift():
    r = interpret.network_shift_test(_fake_fits(-2.0), n_perm=500, n_boot=500)
    assert r["n_genes"] == 6 and r["n_subjects"] == 24
    assert r["mean_shift_h"] == pytest.approx(-2.0, abs=0.6)
    assert r["n_earlier"] == 6 and r["p_mean_shift"] < 0.01
    assert r["ci_lower_h"] < r["mean_shift_h"] < r["ci_upper_h"] < 0


def test_network_shift_test_null():
    r = interpret.network_shift_test(_fake_fits(0.0), n_perm=500, n_boot=500)
    assert abs(r["mean_shift_h"]) < 0.6 and r["p_mean_shift"] > 0.05
    assert r["ci_lower_h"] < 0 < r["ci_upper_h"]


def test_expression_long_shapes():
    expr = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["PER2", "PER3"], columns=["s1", "s2"])
    samples = pd.DataFrame({"sample_id": ["s1", "s2"], "subject": ["A", "A"]})
    long = interpret.expression_long(expr, samples, ["PER3", "MISSING"])
    assert len(long) == 2 and set(long.gene) == {"PER3"} and "subject" in long


# ---------------------------------------------------------------------------------------
# Live APIs
# ---------------------------------------------------------------------------------------
@pytest.mark.network
def test_fetch_string_network():
    e = interpret.fetch_string_network(["PER2", "PER3", "NR1D1"])
    assert {"source", "target", "score"} <= set(e.columns) and len(e) >= 2


@pytest.mark.network
def test_run_enrichment():
    genes = ["PER1", "PER2", "PER3", "CRY1", "CRY2", "ARNTL", "CLOCK", "NR1D1", "NR1D2"]
    r = interpret.run_enrichment(genes)
    assert {"term", "p", "q"} <= set(r.columns)
    assert r["term"].str.contains("ircadian").head(10).any()
