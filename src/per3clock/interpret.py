"""Biological interpretation: enrichment and networks (Part 4).

Network calls (Enrichr, STRING) must cache responses under
``config.EXTERNAL_DIR`` and their tests must be marked ``@pytest.mark.network``.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def run_enrichment(
    genes: list[str],
    gene_sets: tuple[str, ...] = ("GO_Biological_Process_2023", "KEGG_2021_Human"),
    background: list[str] | None = None,
) -> pd.DataFrame:
    """Over-representation analysis (e.g. ``gseapy.enrichr``).

    Parameters
    ----------
    genes : HGNC symbols (e.g. rhythmic genes with q < 0.05 from Part 3).
    gene_sets : Enrichr library names. Check the current names with
        ``gseapy.get_library_name()``.
    background : the expressed genes on the array (recommended).

    Returns
    -------
    DataFrame with at least ``gene_set, term, overlap, p, q, genes``, sorted by ``q``.
    Saved as ``results/tables/enrichment_rhythmic.csv``.
    """
    raise NotImplementedError("Part 4: implement run_enrichment")


def fetch_string_network(
    genes: list[str], species: int = 9606, required_score: int = 400
) -> pd.DataFrame:
    """Fetch STRING interactions among ``genes`` (STRING API ``/api/tsv/network``).

    Returns
    -------
    Edge table with columns ``source, target, score`` (score 0–1), saved as
    ``results/tables/string_edges.csv``.
    """
    raise NotImplementedError("Part 4: implement fetch_string_network")


def build_network(edges: pd.DataFrame) -> nx.Graph:
    """Undirected NetworkX graph from an edge table (edge attribute ``weight`` = score)."""
    raise NotImplementedError("Part 4: implement build_network")


def network_centrality(G: nx.Graph) -> pd.DataFrame:
    """Node table with ``degree``, ``betweenness``, ``closeness``, sorted by degree."""
    raise NotImplementedError("Part 4: implement network_centrality")
