"""Biological interpretation: enrichment and networks (Part 4).

Two interpretation methods, both linked back to the research question
(does PER3 genotype shift the timing of PER2 / PER3 / NR1D1 peaks?):

* **Over-representation analysis** (:func:`run_enrichment`): which biological
  processes are over-represented among the genes with a shared daily rhythm,
  tested against the genes actually expressed on the array (hypergeometric test,
  Benjamini–Hochberg FDR within each library). Gene-set libraries are the
  Enrichr libraries, downloaded once with ``gseapy.get_library`` and cached as GMT
  files in ``config.EXTERNAL_DIR``.
* **Protein-interaction network** (:func:`fetch_string_network`,
  :func:`build_network`, :func:`network_centrality`): where PER3 sits in the clock
  network (STRING), with NetworkX centralities and Louvain communities.

Network calls (Enrichr, STRING) cache their responses under
``config.EXTERNAL_DIR`` and their tests are marked ``@pytest.mark.network``.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from . import config

DEFAULT_LIBRARIES: tuple[str, ...] = ("GO_Biological_Process_2023", "KEGG_2021_Human")

STRING_API: str = "https://string-db.org/api"
STRING_CALLER: str = "per3clock"  # generic client name (no personal details)

# Core clock + well-known clock-controlled transcription factors and kinases used as the
# seed of the STRING network (Takahashi 2017). Genes not expressed on the array are
# dropped in the notebook.
CLOCK_NETWORK_GENES: tuple[str, ...] = config.CLOCK_GENES + (
    "CRY2", "NPAS2", "DBP", "TEF", "HLF", "NFIL3", "BHLHE40", "BHLHE41",
    "RORA", "RORC", "CIART", "CSNK1D", "CSNK1E",
)


# ---------------------------------------------------------------------------------------
# Gene lists
# ---------------------------------------------------------------------------------------
def select_rhythmic_genes(
    scan: pd.DataFrame,
    *,
    q_max: float = 0.05,
    min_amplitude: float = 0.2,
    phase_window: tuple[float, float] | None = None,
) -> list[str]:
    """Genes from the genome-wide scan (``rhythmic_genes_SE.csv``) passing the filters.

    Parameters
    ----------
    scan : table indexed by gene (or with a ``gene`` column) with ``q``, ``amplitude``
        and ``acrophase_h``.
    q_max : BH q-value threshold (rhythmicity).
    min_amplitude : minimum population amplitude in log2 units (0.2 = peak-to-trough
        of 0.4 log2, i.e. ~32 %).
    phase_window : optional ``(start, end)`` in hours; keeps genes whose acrophase lies
        in ``[start, end)``. Wraps around 24 h when ``start > end``.
    """
    s = scan.set_index("gene") if "gene" in scan.columns else scan
    keep = (s["q"] < q_max) & (s["amplitude"] >= min_amplitude)
    if phase_window is not None:
        a, b = phase_window
        ph = s["acrophase_h"] % config.PERIOD_H
        keep &= ((ph >= a) & (ph < b)) if a <= b else ((ph >= a) | (ph < b))
    return sorted(s.index[keep].astype(str))


# ---------------------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------------------
def _read_gmt(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                out[parts[0]] = [g for g in parts[2:] if g]
    return out


def _write_gmt(library: Mapping[str, Iterable[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for term, genes in library.items():
            fh.write("\t".join([term, "", *genes]) + "\n")


def load_gene_set_library(
    name: str, *, cache_dir: Path | None = None, refresh: bool = False
) -> dict[str, list[str]]:
    """Enrichr gene-set library ``name`` as ``{term: [genes]}``, cached as GMT.

    The first call downloads it with ``gseapy.get_library`` (needs internet) and
    writes ``<cache_dir>/enrichr_<name>.gmt``; later calls read the file.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else config.EXTERNAL_DIR
    path = cache_dir / f"enrichr_{name}.gmt"
    if path.exists() and not refresh:
        return _read_gmt(path)
    import gseapy  # local import: heavy, and only needed for the download

    lib = gseapy.get_library(name=name, organism="Human")
    _write_gmt(lib, path)
    return _read_gmt(path)


def run_enrichment(
    genes: list[str],
    gene_sets: tuple[str, ...] | Mapping[str, Mapping[str, Iterable[str]]] = DEFAULT_LIBRARIES,
    background: list[str] | None = None,
    *,
    min_size: int = 5,
    max_size: int = 500,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Over-representation analysis (one-sided hypergeometric test).

    Parameters
    ----------
    genes : HGNC symbols (e.g. rhythmic genes with q < 0.05 from Part 3).
    gene_sets : Enrichr library names (downloaded and cached with
        :func:`load_gene_set_library`; check names with ``gseapy.get_library_name()``),
        or a mapping ``{library_name: {term: genes}}`` (offline use / tests).
    background : the expressed genes on the array (recommended). If ``None``, the
        union of all genes in each library is used (Enrichr's default behaviour).
    min_size, max_size : keep terms with this many background genes.

    Returns
    -------
    DataFrame with ``gene_set, term, overlap, n_overlap, term_size, n_genes,
    n_background, fold_enrichment, p, q, genes``, sorted by ``q`` then ``p``. ``q`` is
    Benjamini–Hochberg within each library; ``overlap`` is ``"k/K"`` (Enrichr style);
    ``genes`` is ``;``-joined. Saved as ``results/tables/enrichment_rhythmic.csv``.
    """
    if isinstance(gene_sets, Mapping):
        libraries = {name: dict(lib) for name, lib in gene_sets.items()}
    else:
        libraries = {name: load_gene_set_library(name, cache_dir=cache_dir) for name in gene_sets}

    query = {str(g).upper() for g in genes}
    rows = []
    for lib_name, lib in libraries.items():
        lib_sets = {t: {str(g).upper() for g in gs} for t, gs in lib.items()}
        universe = (
            {str(g).upper() for g in background}
            if background is not None
            else set().union(*lib_sets.values()) if lib_sets else set()
        )
        hits = query & universe
        N, n = len(universe), len(hits)
        lib_rows = []
        for term, members in lib_sets.items():
            members = members & universe
            K = len(members)
            if K < min_size or K > max_size:
                continue
            ov = hits & members
            k = len(ov)
            if k == 0:
                continue
            lib_rows.append({
                "gene_set": lib_name,
                "term": term,
                "overlap": f"{k}/{K}",
                "n_overlap": k,
                "term_size": K,
                "n_genes": n,
                "n_background": N,
                "fold_enrichment": (k / n) / (K / N) if n and N else np.nan,
                "p": float(hypergeom.sf(k - 1, N, K, n)),
                "genes": ";".join(sorted(ov)),
            })
        if lib_rows:
            df = pd.DataFrame(lib_rows)
            df["q"] = multipletests(df["p"], method="fdr_bh")[1]
            rows.append(df)

    cols = ["gene_set", "term", "overlap", "n_overlap", "term_size", "n_genes",
            "n_background", "fold_enrichment", "p", "q", "genes"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.concat(rows, ignore_index=True)[cols]
    return out.sort_values(["q", "p", "fold_enrichment"], ascending=[True, True, False],
                           ignore_index=True)


# ---------------------------------------------------------------------------------------
# STRING network
# ---------------------------------------------------------------------------------------
def _string_post(method: str, params: dict, cache_path: Path, refresh: bool) -> str:
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")
    import requests

    r = requests.post(f"{STRING_API}/tsv/{method}",
                      data={**params, "caller_identity": STRING_CALLER}, timeout=60)
    r.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(r.text, encoding="utf-8")
    return r.text


def fetch_string_network(
    genes: list[str],
    species: int = 9606,
    required_score: int = 400,
    *,
    add_nodes: int = 0,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch STRING interactions among ``genes`` (STRING API ``/api/tsv/network``).

    Gene symbols are first mapped to STRING identifiers (``/api/tsv/get_string_ids``)
    so that edges are reported under **our** symbols even when STRING's preferred
    name differs (e.g. ARNTL / BMAL1). Raw responses are cached in ``cache_dir``
    (default ``config.EXTERNAL_DIR``) keyed by the query.

    Returns
    -------
    Edge table with columns ``source, target, score`` (combined score, 0–1) plus the
    evidence channels ``experimental, database, textmining, coexpression`` (0–1), one
    row per unordered pair, sorted by score. Saved as ``results/tables/string_edges.csv``.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else config.EXTERNAL_DIR
    genes = list(dict.fromkeys(str(g) for g in genes))
    key = hashlib.sha1(json.dumps([sorted(genes), species, required_score, add_nodes])
                       .encode()).hexdigest()[:12]

    ids_txt = _string_post(
        "get_string_ids",
        {"identifiers": "\r".join(genes), "species": species, "limit": 1, "echo_query": 1},
        cache_dir / f"string_ids_{key}.tsv", refresh,
    )
    ids = pd.read_csv(io.StringIO(ids_txt), sep="\t")
    to_query = dict(zip(ids["stringId"], ids["queryItem"], strict=False))

    net_txt = _string_post(
        "network",
        {"identifiers": "\r".join(ids["stringId"]), "species": species,
         "required_score": required_score, "add_nodes": add_nodes},
        cache_dir / f"string_network_{key}.tsv", refresh,
    )
    cols = ["source", "target", "score", "experimental", "database", "textmining",
            "coexpression"]
    if not net_txt.strip() or "stringId_A" not in net_txt.splitlines()[0]:
        return pd.DataFrame(columns=cols)
    raw = pd.read_csv(io.StringIO(net_txt), sep="\t")
    name_a = raw["stringId_A"].map(to_query).fillna(raw["preferredName_A"])
    name_b = raw["stringId_B"].map(to_query).fillna(raw["preferredName_B"])
    edges = pd.DataFrame({
        "source": np.where(name_a <= name_b, name_a, name_b),
        "target": np.where(name_a <= name_b, name_b, name_a),
        "score": raw["score"].astype(float),
        "experimental": raw["escore"].astype(float),
        "database": raw["dscore"].astype(float),
        "textmining": raw["tscore"].astype(float),
        "coexpression": raw["ascore"].astype(float),
    })
    edges = edges[edges.source != edges.target]
    edges = (edges.sort_values("score", ascending=False)
             .drop_duplicates(["source", "target"]).reset_index(drop=True))
    return edges[cols]


def build_network(edges: pd.DataFrame) -> nx.Graph:
    """Undirected NetworkX graph from an edge table (edge attribute ``weight`` = score).

    Any other numeric edge columns (e.g. evidence channels) are kept as attributes.
    """
    extra = [c for c in edges.columns if c not in ("source", "target", "score")]
    G = nx.from_pandas_edgelist(
        edges.rename(columns={"score": "weight"}), "source", "target",
        edge_attr=["weight", *extra],
    )
    return G


def network_centrality(G: nx.Graph, *, seed: int = config.SEED) -> pd.DataFrame:
    """Node table with ``degree``, ``betweenness``, ``closeness``, sorted by degree.

    Also returns ``strength`` (sum of edge weights) and ``community`` (Louvain,
    weighted, fixed seed). Betweenness and closeness use unweighted shortest paths
    (STRING scores are confidences, not distances). Index: gene.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["degree", "strength", "betweenness", "closeness",
                                     "community"])
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    comm_of = {n: i for i, c in enumerate(sorted(comms, key=len, reverse=True)) for n in c}
    df = pd.DataFrame({
        "degree": dict(G.degree()),
        "strength": dict(G.degree(weight="weight")),
        "betweenness": nx.betweenness_centrality(G, normalized=True),
        "closeness": nx.closeness_centrality(G),
        "community": comm_of,
    })
    df.index.name = "gene"
    return df.sort_values(["degree", "betweenness"], ascending=False)


# ---------------------------------------------------------------------------------------
# Linking the network back to the question: is there a network-wide genotype shift?
# ---------------------------------------------------------------------------------------
def expression_long(
    expr_genes: pd.DataFrame, samples: pd.DataFrame, genes: Iterable[str]
) -> pd.DataFrame:
    """Genes × samples matrix → long table merged with sample metadata.

    ``samples`` is a metadata table such as ``data.analysis_samples()`` (one row per
    sample, with ``sample_id``). The result has the columns expected by
    :func:`per3clock.rhythm.fit_subject_cosinors` (``gene``, ``expression`` plus the
    metadata columns).
    """
    genes = [g for g in genes if g in expr_genes.index]
    cols = [s for s in samples["sample_id"] if s in expr_genes.columns]
    long = (expr_genes.loc[genes, cols].copy().rename_axis("gene").reset_index()
            .melt(id_vars="gene", var_name="sample_id", value_name="expression"))
    return long.merge(samples, on="sample_id", how="inner")


def _genotype_shifts(z: np.ndarray, is55: np.ndarray, period: float) -> np.ndarray:
    """Per-gene circular difference of mean acrophases (5/5 − 4/4), in hours."""
    s55 = z[is55].sum(axis=0)
    s44 = z[~is55].sum(axis=0)
    return np.angle(s55 * np.conj(s44)) * period / (2 * np.pi)


def network_shift_test(
    fits: pd.DataFrame,
    *,
    n_perm: int = 10_000,
    n_boot: int = 10_000,
    period: float = config.PERIOD_H,
    seed: int = config.SEED,
) -> dict:
    """Is there a **consistent** genotype shift across many clock-network genes?

    Per-gene tests (``circstats.genotype_shift_table``) ask about one gene at a time.
    Here the question is whether the genes, taken together, move in the same
    direction. Two statistics are computed from the per-subject acrophases in
    ``fits`` (output of ``rhythm.fit_subject_cosinors``):

    * ``mean_shift_h``: mean over genes of the circular difference of genotype mean
      acrophases (5/5 − 4/4; negative = 5/5 earlier);
    * ``n_earlier``: number of genes whose 5/5 mean peaks earlier.

    The null distribution comes from permuting genotype labels **between subjects**
    (all genes of a subject move together), which keeps the correlation between
    genes, so genes are not treated as independent evidence. Two-sided p-values.
    The 95 % CI of ``mean_shift_h`` is a bootstrap over subjects within genotype.

    Returns
    -------
    dict with ``n_genes, n_subjects, mean_shift_h, ci_lower_h, ci_upper_h, n_earlier,
    p_mean_shift, p_n_earlier, n_perm`` and ``per_gene`` (Series of shifts in hours).
    """
    wide = fits.pivot_table(index="subject", columns="gene", values="acrophase_h",
                            aggfunc="first")
    geno = fits.groupby("subject")["per3_genotype"].first().reindex(wide.index)
    is55 = (geno == config.GENOTYPES[1]).to_numpy()
    ang = wide.to_numpy(dtype=float) * 2 * np.pi / period
    z = np.where(np.isfinite(ang), np.exp(1j * np.nan_to_num(ang)), 0)

    obs = _genotype_shifts(z, is55, period)
    n_genes = obs.size
    obs_mean, obs_n = float(obs.mean()), int((obs < 0).sum())

    rng = np.random.default_rng(seed)
    perm_mean = np.empty(n_perm)
    perm_n = np.empty(n_perm)
    for i in range(n_perm):
        d = _genotype_shifts(z, rng.permutation(is55), period)
        perm_mean[i] = d.mean()
        perm_n[i] = (d < 0).sum()
    p_mean = (1 + np.sum(np.abs(perm_mean) >= abs(obs_mean) - 1e-12)) / (n_perm + 1)
    half = n_genes / 2
    p_n = (1 + np.sum(np.abs(perm_n - half) >= abs(obs_n - half))) / (n_perm + 1)

    # 95 % CI of the mean shift: resample subjects within each genotype
    i55, i44 = np.flatnonzero(is55), np.flatnonzero(~is55)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(i55, i55.size), rng.choice(i44, i44.size)])
        boot[i] = _genotype_shifts(z[idx], is55[idx], period).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_genes": n_genes,
        "n_subjects": int(len(wide)),
        "mean_shift_h": obs_mean,
        "ci_lower_h": float(lo),
        "ci_upper_h": float(hi),
        "n_earlier": obs_n,
        "p_mean_shift": float(p_mean),
        "p_n_earlier": float(p_n),
        "n_perm": n_perm,
        "per_gene": pd.Series(obs, index=wide.columns, name="shift_h"),
    }


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------
def plot_top_terms(
    tables: Mapping[str, pd.DataFrame], *, n: int = 10, q_max: float = 0.05,
    max_label: int = 60,
) -> plt.Figure:
    """Horizontal bar charts (−log10 q) of the top ``n`` terms of each enrichment table.

    ``tables`` maps a panel title to a :func:`run_enrichment` result.
    """
    fig, axes = plt.subplots(len(tables), 1, figsize=(8.5, 0.32 * n * len(tables) + 1.2),
                             squeeze=False)
    for ax, (title, t) in zip(axes[:, 0], tables.items(), strict=True):
        top = t.nsmallest(n, ["q", "p"]).iloc[::-1]
        lib_short = top["gene_set"].str.replace(r"_\d{4}.*$", "", regex=True)
        labels = [
            (term if len(term) <= max_label else term[: max_label - 1] + "…") + f"  [{lib}]"
            for term, lib in zip(top["term"].str.replace(r"\s*(\(GO:\d+\)|CL\d+)", "", regex=True),
                                 lib_short, strict=True)
        ]
        y = -np.log10(top["q"].clip(lower=1e-300))
        colors = np.where(top["q"] < q_max, "#0072B2", "#BBBBBB")
        ax.barh(range(len(top)), y, color=colors)
        ax.set_yticks(range(len(top)), labels, fontsize=8)
        for i, (val, ov) in enumerate(zip(y, top["overlap"], strict=True)):
            ax.text(val, i, f" {ov}", va="center", fontsize=7)
        ax.axvline(-np.log10(q_max), color="k", lw=0.8, ls="--")
        ax.set_xlabel("−log10 q (BH)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_string_network(
    G: nx.Graph,
    node_info: pd.DataFrame,
    *,
    highlight: Iterable[str] = config.GENES_PRIMARY,
    seed: int = config.SEED,
    title: str | None = None,
) -> plt.Figure:
    """Draw the STRING network.

    ``node_info`` (index: gene) needs ``acrophase_h``, ``amplitude`` and ``q`` from the
    rhythmicity scan. Node colour = acrophase on a cyclic map (grey when not rhythmic
    at q < 0.05), node size = amplitude, edge width = STRING score; ``highlight`` genes
    get a thick black outline.
    """
    pos = nx.spring_layout(G, weight="weight", seed=seed, k=1.2 / np.sqrt(max(len(G), 1)))
    fig, ax = plt.subplots(figsize=(8, 7))
    widths = [0.5 + 3 * (d["weight"] - 0.4) / 0.6 for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#999999", alpha=0.6)

    cmap = plt.get_cmap("twilight_shifted")
    info = node_info.reindex(list(G.nodes))
    rhythmic = info["q"] < 0.05
    colors = [cmap((ph % 24) / 24) if r else "#DDDDDD"
              for ph, r in zip(info["acrophase_h"].fillna(0), rhythmic, strict=True)]
    sizes = 250 + 2500 * info["amplitude"].fillna(0).clip(upper=0.5)
    hl = set(highlight)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=sizes,
        edgecolors=["black" if n in hl else "#666666" for n in G.nodes],
        linewidths=[2.5 if n in hl else 0.6 for n in G.nodes],
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                            font_weight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 24))
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, ticks=[0, 6, 12, 18, 24])
    cb.set_label("Acrophase (circadian phase, h; grey = not rhythmic, q ≥ 0.05)")
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left")
    return fig
