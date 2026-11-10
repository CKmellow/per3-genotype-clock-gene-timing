import networkx as nx
import pandas as pd
import pytest

from per3clock import interpret

part4 = pytest.mark.xfail(reason="Part 4", strict=True, raises=NotImplementedError)


@part4
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


@part4
@pytest.mark.network
def test_fetch_string_network():
    e = interpret.fetch_string_network(["PER2", "PER3", "NR1D1"])
    assert {"source", "target", "score"} <= set(e.columns) and len(e) >= 2


@part4
@pytest.mark.network
def test_run_enrichment():
    genes = ["PER1", "PER2", "PER3", "CRY1", "CRY2", "ARNTL", "CLOCK", "NR1D1", "NR1D2"]
    r = interpret.run_enrichment(genes)
    assert {"term", "p", "q"} <= set(r.columns)
    assert r["term"].str.contains("ircadian").head(10).any()
