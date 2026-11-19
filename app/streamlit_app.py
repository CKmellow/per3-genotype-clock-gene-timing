"""PER3 genotype and clock-gene timing: interactive dashboard.

Run from the repo root:
    pip install -e . streamlit plotly
    streamlit run app/streamlit_app.py

Reads the saved tables in results/tables/ and the committed clock-gene data in data/raw/,
so it needs no download and no re-analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from per3clock import config, data

TABLES = Path(config.TABLES_DIR)
C44, C55 = "#D55E00", "#0072B2"
COLORS = {"4/4": C44, "5/5": C55}
LABELS = {"4/4": "4/4 · night owl", "5/5": "5/5 · early bird"}
MAIN_GENES = ["NR1D1", "PER2", "PER3"]
TIME_BASES = {"Clock time": "clock_time", "Melatonin time": "circadian_phase"}
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"

st.set_page_config(
    page_title="PER3 Clock Dashboard", page_icon=":material/schedule:", layout="wide"
)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: {FONT}; }}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
.hero {{
  background: linear-gradient(120deg, #0b1f3a 0%, #12355b 55%, #1d4e7a 100%);
  color: #f8fafc; border-radius: 18px; padding: 28px 32px; margin-bottom: 18px;
  position: relative; overflow: hidden;
}}
.hero:after {{
  content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
  border-radius: 50%; border: 28px solid rgba(213,94,0,.35);
  box-shadow: 0 0 0 26px rgba(0,114,178,.30);
}}
.hero .eyebrow {{ font-size: .78rem; letter-spacing: .12em; text-transform: uppercase;
  color: #93c5fd; font-weight: 600; }}
.hero h1 {{ font-size: 2.1rem; margin: 6px 0 8px; color: #fff; font-weight: 700; }}
.hero p {{ max-width: 820px; color: #dbe7f5; font-size: 1.02rem; line-height: 1.55; margin: 0; }}
.pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem;
  font-weight: 600; margin-right: 6px; }}
.p44 {{ background: rgba(213,94,0,.18); color: #ffb98a; }}
.p55 {{ background: rgba(0,114,178,.25); color: #9fd4ff; }}
.kpi {{ background: #fff; border: 1px solid {GRID}; border-radius: 14px; padding: 16px 18px;
  height: 100%; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
.kpi .label {{ color: {MUTED}; font-size: .8rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .05em; }}
.kpi .value {{ color: {INK}; font-size: 1.9rem; font-weight: 700; margin: 4px 0 2px; }}
.kpi .sub {{ color: {MUTED}; font-size: .86rem; }}
.kpi.accent {{ border-top: 4px solid {C55}; }}
.kpi.warn {{ border-top: 4px solid {C44}; }}
.kpi.ok {{ border-top: 4px solid #009E73; }}
.kpi.neutral {{ border-top: 4px solid #6b7280; }}
.note {{ background: #f8fafc; border-left: 4px solid {C55}; border-radius: 8px;
  padding: 12px 16px; color: {INK}; font-size: .93rem; }}
.section {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin: 8px 0 2px; }}
.subtle {{ color: {MUTED}; font-size: .88rem; margin-bottom: 6px; }}
div[data-testid="stTabs"] button p {{ font-size: 1rem; font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data
@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


@st.cache_data
def samples() -> pd.DataFrame:
    """Clock-gene expression for the main set, clock time unwrapped past midnight."""
    main = data.analysis_samples()
    clock = data.load_clock_genes()
    s = clock[clock.sample_id.isin(main.sample_id)]
    s = s.sort_values(["subject", "gene", "timepoint_n"]).copy()

    def unwrap(t: pd.Series) -> np.ndarray:
        t = t.to_numpy().copy()
        t[1:] += 24 * np.cumsum(np.diff(t) < 0)
        return t

    s["clock_time_unwrapped_h"] = s.groupby(["subject", "gene"])["clock_time_h"].transform(unwrap)
    s["centred"] = s.expression - s.groupby(["subject", "gene"]).expression.transform("mean")
    return s


def hhmm(h: float) -> str:
    if pd.isna(h):
        return "n/a"
    h = h % 24
    m = int(round((h % 1) * 60))
    return f"{(int(h) + m // 60) % 24:02d}:{m % 60:02d}"


def signed(x: float, d: int = 2) -> str:
    return f"{x:+.{d}f}".replace("-", "−")


def style(fig: go.Figure, height: int = 420, **kw) -> go.Figure:
    layout = dict(
        template="plotly_white",
        height=height,
        font=dict(family=FONT, color=INK, size=13),
        margin=dict(t=30, b=40, l=10, r=10),
        hoverlabel=dict(font_family=FONT, bgcolor="white"),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    layout.update(kw)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def kpi(col, label: str, value: str, sub: str, kind: str = "accent") -> None:
    col.markdown(
        f'<div class="kpi {kind}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="subtle">{sub}</div>', unsafe_allow_html=True)


peaks = table("acrophase_by_subject.csv")
peaks = peaks[peaks.sample_set == "Primary"]
tests = table("acrophase_genotype_tests.csv")
sens = table("acrophase_sensitivity_tests.csv")
network = table("network_shift_test.csv")
ml = table("ml_phase_prediction.csv")
paired = table("ml_paired_comparisons.csv")
mlk = table("ml_sensitivity_k.csv")
mlgenes = table("ml_selected_genes.csv")
qc = table("qc_counts.csv")
rhythmic = table("rhythmic_genes_SE.csv")
profiles = table("eda_clock_gene_profiles.csv")
pca_r2 = table("eda_pca_factor_r2.csv")
cells = table("enrichment_cell_types.csv")
timing = table("eda_subject_timing.csv")
S = samples()
people = S.drop_duplicates("subject").sort_values("subject").set_index("subject").per3_genotype

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### :material/tune: Controls")
    gene = st.selectbox("Gene", MAIN_GENES + sorted(set(S.gene) - set(MAIN_GENES)), key="gene")
    tb_label = st.segmented_control(
        "Time measured against", list(TIME_BASES), default="Clock time", key="tb"
    )
    tb_label = tb_label or "Clock time"
    tb = TIME_BASES[tb_label]
    person = st.selectbox(
        "Highlight a person",
        people.index,
        index=list(people.index).index("AF0295") if "AF0295" in people.index else 0,
        format_func=lambda x: f"{x} · {people[x]}",
        key="person",
    )
    st.divider()
    st.markdown(
        f'<span class="pill" style="color:{C44};background:rgba(213,94,0,.12)">'
        f'● 4/4 night owl</span><span class="pill" style="color:{C55};'
        f'background:rgba(0,114,178,.12)">● 5/5 early bird</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Data: GEO GSE39445 (Möller-Levet et al. 2013, PNAS). Whole blood, Agilent 44k "
        "array, samples every 3 h over ~40 h awake. Main set: sufficient-sleep week, "
        "24 people, 221 samples."
    )

# ------------------------------------------------------------------ hero + KPIs
st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Bioinformatics group project · PER3 VNTR · human blood transcriptome</div>
  <h1>Does PER3 genotype shift the timing of clock genes?</h1>
  <p><span class="pill p44">4/4 night owls</span><span class="pill p55">5/5 early birds</span>
  <br><br>No detectable shift. 5/5 carriers peak 0.5–1.4 h earlier on the wall clock, but every
  95% interval includes zero, and the gap disappears when time is measured against each person's
  own melatonin rhythm. With 11 vs 13 people the result is inconclusive, not proof of no effect.</p>
</div>
""",
    unsafe_allow_html=True,
)

nr = tests[(tests.gene == "NR1D1") & (tests.time_base == "clock_time")].iloc[0]
best = ml.loc[ml.median_abs_error_h.idxmin()]
base = ml[ml.model == "baseline: clock time only"].iloc[0]
pair = paired[(paired.model_a == "baseline: clock time only") & (paired.model_b == best.model)]
k1, k2, k3, k4 = st.columns(4)
kpi(
    k1,
    "People",
    f"{len(people)}",
    f"{(people == '4/4').sum()} × 4/4 · {(people == '5/5').sum()} × 5/5 · 221 samples",
    "neutral",
)
kpi(k2, "Rhythmic genes", "2,007", f"of {len(rhythmic):,} expressed (q < 0.05, amp ≥ 0.2)", "ok")
kpi(
    k3,
    "NR1D1 shift 5/5 − 4/4",
    f"{signed(nr.shift_h)} h",
    f"95% CI {signed(nr.ci_lower_h, 1)} to {signed(nr.ci_upper_h, 1)} · q = "
    f"{nr.permutation_q:.2f} · not significant",
    "warn",
)
better = (
    f" · {int(pair.iloc[0].n_better)}/{int(pair.iloc[0].n_subjects)} people better"
    if len(pair)
    else ""
)
kpi(
    k4,
    "Phase prediction error",
    f"{best.median_abs_error_h:.2f} h",
    f"vs {base.median_abs_error_h:.2f} h with clock time alone{better}",
    "accent",
)
st.write("")

tabs = st.tabs(
    [
        ":material/insights: Overview",
        ":material/person: One person",
        ":material/compare_arrows: Early birds vs night owls",
        ":material/model_training: Machine learning",
        ":material/biotech: Data & biology",
    ]
)

# ------------------------------------------------------------------ overview
with tabs[0]:
    left, right = st.columns([1.15, 1])
    with left:
        section(
            f"{gene}: average daily profile by genotype",
            f"Expression centred within each person · mean ± SE · {tb_label.lower()} "
            "(change gene and time base in the sidebar)",
        )
        axis = "clock_time_h" if tb == "clock_time" else "circadian_phase_h"
        pr = profiles[
            (profiles.gene == gene) & (profiles.time_axis == axis) & (profiles.n_subjects >= 3)
        ]
        fig = go.Figure()
        for gt in ["4/4", "5/5"]:
            d = pr[pr.per3_genotype == gt].sort_values("time_bin_h")
            rgb = tuple(int(COLORS[gt][i : i + 2], 16) for i in (1, 3, 5))
            fig.add_scatter(
                x=pd.concat([d.time_bin_h, d.time_bin_h[::-1]]),
                y=pd.concat([d["mean"] + d.se, (d["mean"] - d.se)[::-1]]),
                fill="toself",
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
            fig.add_scatter(
                x=d.time_bin_h,
                y=d["mean"],
                mode="lines+markers",
                name=LABELS[gt],
                line=dict(color=COLORS[gt], width=3, shape="spline"),
                marker=dict(size=7),
                customdata=np.c_[d.se, d.n_subjects],
                hovertemplate="%{y:+.2f} ± %{customdata[0]:.2f} log2 · "
                "n = %{customdata[1]}<extra>" + gt + "</extra>",
            )
        if tb == "clock_time":
            ticks = np.arange(12, 46, 6)
            fig.update_xaxes(
                tickvals=ticks,
                ticktext=[hhmm(t) for t in ticks],
                title="Clock time (across ~40 h awake)",
            )
        else:
            fig.update_xaxes(title="Hours from melatonin phase 0", dtick=3)
        fig.update_yaxes(title="Centred expression (log2)")
        st.plotly_chart(style(fig, 400), width="stretch")
    with right:
        section(
            "All genotype shifts at a glance",
            "Peak-time difference 5/5 − 4/4 with 95% bootstrap CI · negative = 5/5 earlier",
        )
        fig = go.Figure()
        for i, (lab, t) in enumerate(TIME_BASES.items()):
            d = tests[tests.time_base == t].set_index("gene").loc[MAIN_GENES[::-1]]
            y = np.arange(3) + (0.15 if i == 0 else -0.15)
            fig.add_scatter(
                x=d.shift_h,
                y=y,
                mode="markers",
                name=lab,
                marker=dict(
                    size=13,
                    color=C55 if i == 0 else "#009E73",
                    symbol="circle" if i == 0 else "diamond",
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=d.ci_upper_h - d.shift_h,
                    arrayminus=d.shift_h - d.ci_lower_h,
                    thickness=2.5,
                    width=0,
                ),
                customdata=np.c_[d.index, d.ci_lower_h, d.ci_upper_h, d.permutation_q],
                hovertemplate="<b>%{customdata[0]}</b> · "
                + lab
                + "<br>shift %{x:+.2f} h [%{customdata[1]:+.2f}, "
                "%{customdata[2]:+.2f}]<br>q = %{customdata[3]:.2f}<extra></extra>",
            )
        fig.add_vrect(x0=-1, x1=1, fillcolor="#f3f4f6", line_width=0, layer="below")
        fig.add_vline(x=0, line_color=MUTED, line_dash="dot")
        fig.update_yaxes(tickvals=[0, 1, 2], ticktext=MAIN_GENES[::-1])
        fig.update_xaxes(title="Shift in peak time (h)")
        st.plotly_chart(style(fig, 400), width="stretch")
    st.markdown(
        '<div class="note"><b>How to read it.</b> On clock time the 5/5 group looks a little '
        "earlier, but every interval includes zero (grey band = ±1 h). 5/5 people also woke "
        "~1.2 h earlier and their melatonin ran ~1.25 h earlier, so the whole system shifted: "
        "measured against melatonin, the clock-gene gap vanishes.</div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ one person
with tabs[1]:
    g = people[person]
    a, b = st.columns([2, 1])
    genes_sel = a.multiselect(
        "Genes to show for this person",
        sorted(S.gene.unique()),
        default=[gene] if gene in MAIN_GENES else [gene, "PER2"],
    )
    centre = b.toggle("Centre each gene (compare shapes)", value=len(genes_sel) > 1)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
    fig = go.Figure()
    for i, gname in enumerate(genes_sel):
        d = S[(S.subject == person) & (S.gene == gname)]
        col = COLORS[g] if len(genes_sel) == 1 else palette[i % len(palette)]
        yv = d.centred if centre else d.expression
        fig.add_scatter(
            x=d.clock_time_unwrapped_h,
            y=yv,
            mode="markers",
            name=gname,
            legendgroup=gname,
            marker=dict(size=10, color=col, line=dict(width=1, color="white")),
            customdata=np.c_[d.clock_time_h.map(hhmm), d.hours_awake],
            hovertemplate=f"<b>{gname}</b> · %{{customdata[0]}} · "
            "%{customdata[1]} h awake<br>%{y:.2f} log2<extra></extra>",
        )
        f = peaks[
            (peaks.subject == person) & (peaks.gene == gname) & (peaks.time_base == "clock_time")
        ]
        if len(f) and len(d):
            f = f.iloc[0]
            t = np.linspace(d.clock_time_unwrapped_h.min(), d.clock_time_unwrapped_h.max(), 300)
            yfit = f.amplitude * np.cos(2 * np.pi * (t - f.acrophase_h) / 24)
            yfit = yfit + (0 if centre else f.mesor)
            fig.add_scatter(
                x=t,
                y=yfit,
                mode="lines",
                line=dict(color=col, width=2.5),
                legendgroup=gname,
                showlegend=False,
                hoverinfo="skip",
            )
    d0 = S[S.subject == person]
    ticks = np.arange(
        np.floor(d0.clock_time_unwrapped_h.min() / 3) * 3, d0.clock_time_unwrapped_h.max() + 3, 3
    )
    fig.update_xaxes(tickvals=ticks, ticktext=[hhmm(x) for x in ticks], title="Clock time")
    fig.update_yaxes(title="Centred expression (log2)" if centre else "Expression (log2)")
    section(
        f"{person} · {LABELS[g]}",
        "Dots are blood samples every 3 h during ~40 h awake; lines are the fitted "
        "24-hour cosine (cosinor). Pick the person in the sidebar.",
    )
    st.plotly_chart(style(fig, 420), width="stretch")
    fits = peaks[
        (peaks.subject == person) & (peaks.time_base == "clock_time") & peaks.gene.isin(genes_sel)
    ]
    if len(fits):
        cols = st.columns(min(len(fits), 4))
        for c, (_, f) in zip(cols, fits.iterrows(), strict=False):
            kind = "ok" if f.p < 0.05 else "neutral"
            kpi(
                c,
                f"{f.gene} peak",
                hhmm(f.acrophase_h),
                f"amplitude {f.amplitude:.2f} · R² {f.r2:.2f} · p {f.p:.3f}",
                kind,
            )

    st.write("")
    section(
        f"{gene}: {person} against everyone else",
        "Each person's fitted wave, centred · thick dark line = highlighted person",
    )
    pk = peaks[(peaks.gene == gene) & (peaks.time_base == "clock_time")]
    t = np.linspace(12, 48, 300)
    fig = go.Figure()
    for gt in ["4/4", "5/5"]:
        first = True
        for _, f in pk[(pk.per3_genotype == gt) & (pk.subject != person)].iterrows():
            fig.add_scatter(
                x=t,
                y=f.amplitude * np.cos(2 * np.pi * (t - f.acrophase_h) / 24),
                mode="lines",
                line=dict(color=COLORS[gt], width=1),
                opacity=0.35,
                name=LABELS[gt],
                legendgroup=gt,
                showlegend=first,
                hovertemplate=f"{f.subject}: peak {hhmm(f.acrophase_h)}<extra></extra>",
            )
            first = False
    me = pk[pk.subject == person]
    if len(me):
        f = me.iloc[0]
        fig.add_scatter(
            x=t,
            y=f.amplitude * np.cos(2 * np.pi * (t - f.acrophase_h) / 24),
            mode="lines",
            line=dict(color=INK, width=4),
            name=person,
            hovertemplate=f"{person}: peak {hhmm(f.acrophase_h)}<extra></extra>",
        )
    ticks = np.arange(12, 49, 6)
    fig.update_xaxes(tickvals=ticks, ticktext=[hhmm(x) for x in ticks], title="Clock time")
    fig.update_yaxes(title="Fitted rhythm (log2, centred)")
    st.plotly_chart(style(fig, 380), width="stretch")

# ------------------------------------------------------------------ genotype
with tabs[2]:
    left, right = st.columns([1, 1.15])
    pk = peaks[(peaks.gene == gene) & (peaks.time_base == tb)]
    r = tests[(tests.gene == gene) & (tests.time_base == tb)]
    with left:
        section(
            f"{gene} peak times on a 24-hour dial",
            f"{tb_label} · one dot per person · arrows = group mean direction "
            "(longer = more agreement)",
        )
        fig = go.Figure()
        for gt, rad in [("4/4", 1.0), ("5/5", 0.82)]:
            x = pk[pk.per3_genotype == gt]
            fig.add_scatterpolar(
                r=np.full(len(x), rad),
                theta=x.acrophase_h * 15,
                mode="markers",
                name=LABELS[gt],
                marker=dict(
                    size=[16 if s == person else 11 for s in x.subject],
                    color=COLORS[gt],
                    line=dict(
                        width=[3 if s == person else 1 for s in x.subject],
                        color=[INK if s == person else "white" for s in x.subject],
                    ),
                ),
                customdata=np.c_[x.subject, x.acrophase_h.map(hhmm), x.r2],
                hovertemplate="<b>%{customdata[0]}</b><br>peak %{customdata[1]}"
                "<br>R² %{customdata[2]:.2f}<extra></extra>",
            )
            z = np.exp(1j * x.acrophase_h * 2 * np.pi / 24).mean()
            fig.add_scatterpolar(
                r=[0, abs(z) * 0.9],
                theta=[0, np.degrees(np.angle(z)) % 360],
                mode="lines+markers",
                line=dict(color=COLORS[gt], width=5),
                marker=dict(size=[0, 9], color=COLORS[gt]),
                showlegend=False,
                hoverinfo="skip",
            )
        tick = [f"{h:02d}:00" if tb == "clock_time" else f"{h}h" for h in range(0, 24, 3)]
        fig.update_layout(
            polar=dict(
                bgcolor="#fbfcfe",
                radialaxis=dict(visible=False, range=[0, 1.12]),
                angularaxis=dict(
                    direction="clockwise",
                    rotation=90,
                    gridcolor=GRID,
                    tickvals=list(range(0, 360, 45)),
                    ticktext=tick,
                ),
            )
        )
        st.plotly_chart(
            style(
                fig,
                470,
                margin=dict(t=30, b=50, l=55, r=55),
                legend=dict(orientation="h", y=-0.06, x=0.15),
            ),
            width="stretch",
        )
        if len(r):
            r = r.iloc[0]
            c1, c2 = st.columns(2)
            kpi(
                c1,
                "Shift 5/5 − 4/4",
                f"{signed(r.shift_h)} h",
                f"95% CI {signed(r.ci_lower_h, 1)} to {signed(r.ci_upper_h, 1)}",
                "warn",
            )
            kpi(
                c2,
                "Permutation test",
                f"p = {r.permutation_p:.2f}",
                f"FDR q = {r.permutation_q:.2f} · Watson–Williams p = {r.watson_williams_p:.2f}",
                "neutral",
            )
        else:
            st.info("Genotype tests were run for NR1D1, PER2 and PER3 only.")
    with right:
        section("Is the result robust?", "Main analysis vs three sensitivity analyses")
        rows = tests[tests.time_base == tb].assign(set="Main analysis")
        srows = sens[sens.time_base == tb].assign(
            set=lambda d: d.sample_set.map(
                {
                    "Strict_SE": "Authors' stricter filter",
                    "Sleep_Restriction": "Insufficient-sleep week",
                    "Probe_Mean_SE": "All probes averaged",
                }
            )
        )
        allr = pd.concat([rows, srows])
        sets = [
            "Main analysis",
            "Authors' stricter filter",
            "Insufficient-sleep week",
            "All probes averaged",
        ]
        offs = dict(zip(sets, [0.27, 0.09, -0.09, -0.27], strict=True))
        scol = dict(zip(sets, [C55, "#6b7280", "#E69F00", "#CC79A7"], strict=True))
        gpos = {gg: i for i, gg in enumerate(MAIN_GENES[::-1])}
        fig = go.Figure()
        for sname in sets:
            d = allr[allr.set == sname]
            main_set = sname == "Main analysis"
            fig.add_scatter(
                x=d.shift_h,
                y=d.gene.map(gpos) + offs[sname],
                mode="markers",
                name=sname,
                marker=dict(
                    size=12 if main_set else 9,
                    color=scol[sname],
                    symbol="circle" if main_set else "circle-open",
                    line=dict(width=2, color=scol[sname]),
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=d.ci_upper_h - d.shift_h,
                    arrayminus=d.shift_h - d.ci_lower_h,
                    color=scol[sname],
                    thickness=2.5 if main_set else 1.2,
                    width=0,
                ),
                customdata=np.c_[d.gene, d.ci_lower_h, d.ci_upper_h, d.permutation_p],
                hovertemplate="<b>%{customdata[0]}</b> · "
                + sname
                + "<br>shift %{x:+.2f} h [%{customdata[1]:+.2f}, "
                "%{customdata[2]:+.2f}]<br>p = %{customdata[3]:.2f}<extra></extra>",
            )
        fig.add_vline(x=0, line_dash="dot", line_color=MUTED)
        fig.update_yaxes(tickvals=list(gpos.values()), ticktext=list(gpos.keys()))
        fig.update_xaxes(title=f"Shift 5/5 − 4/4 (h, {tb_label.lower()})")
        st.plotly_chart(style(fig, 440, legend=dict(orientation="h", y=-0.22)), width="stretch")
        n = network.set_index("time_base")
        nc, nph = n.loc["clock_time"], n.loc["circadian_phase"]
        st.markdown(
            f'<div class="note"><b>Whole clock network (14 genes):</b> '
            f"{signed(nc.mean_shift_h)} h on clock time (p = {nc.p_mean_shift:.2f}) but "
            f"{signed(nph.mean_shift_h)} h on melatonin time (p = {nph.p_mean_shift:.2f}). "
            "The earlier clock-time peaks follow the earlier sleep–wake schedule.</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        section(
            "Why no difference? Wake time and melatonin moved together",
            "Each person's wake time vs the clock time of their melatonin phase 0",
        )
        tm = timing[timing.sleep_condition == "Sleep Extension"]
        fig = go.Figure()
        for gt in ["4/4", "5/5"]:
            d = tm[tm.per3_genotype == gt]
            fig.add_scatter(
                x=d.wake_clock_h,
                y=d.phase0_clock_h,
                mode="markers",
                name=LABELS[gt],
                marker=dict(
                    size=[16 if s == person else 11 for s in d.subject],
                    color=COLORS[gt],
                    line=dict(width=1, color="white"),
                ),
                customdata=np.c_[d.subject, d.wake_clock_h.map(hhmm), d.phase0_clock_h.map(hhmm)],
                hovertemplate="<b>%{customdata[0]}</b><br>woke %{customdata[1]}"
                "<br>melatonin phase 0 %{customdata[2]}<extra></extra>",
            )
        fig.update_xaxes(title="Wake time (clock h)")
        fig.update_yaxes(title="Melatonin phase 0 (clock h)")
        st.plotly_chart(style(fig, 340), width="stretch")
    with c2:
        section(
            "How many people would we need?",
            "Rough minimum detectable shift (80% power, α = 0.05), scaled from our CI widths",
        )
        nper = st.slider("People per genotype", 10, 200, 12, step=2, key="nper")
        for gg in MAIN_GENES:
            rr = tests[(tests.gene == gg) & (tests.time_base == "clock_time")].iloc[0]
            se = (rr.ci_upper_h - rr.ci_lower_h) / (2 * 1.96)
            scale = np.sqrt((2 / nper) / (1 / rr.n_44 + 1 / rr.n_55))
            mde = 2.8 * se * scale
            st.progress(min(1.0, 1 / mde), text=f"{gg}: can detect shifts ≥ {mde:.1f} h")
        st.caption(
            "A bar is full once a 1-hour shift becomes detectable. Simple scaling, "
            "not a formal circular power analysis."
        )

# ------------------------------------------------------------------ ML
with tabs[3]:
    left, right = st.columns([1.2, 1])
    with left:
        section(
            "Predicting a new person's internal time",
            "Leave-one-subject-out CV (23 folds) · median absolute circular error, "
            "95% CI by resampling people",
        )
        m = ml.sort_values("median_abs_error_h", ascending=False)
        kinds = [
            "#9ca3af" if x.startswith("baseline") else (C55 if x == best.model else "#93c5fd")
            for x in m.model
        ]
        fig = go.Figure(
            go.Bar(
                x=m.median_abs_error_h,
                y=m.model,
                orientation="h",
                marker_color=kinds,
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=m.median_ci95_high_h - m.median_abs_error_h,
                    arrayminus=m.median_abs_error_h - m.median_ci95_low_h,
                    color=MUTED,
                    thickness=1.5,
                ),
                text=m.median_abs_error_h.map(lambda v: f"{v:.2f} h"),
                textposition="inside",
                insidetextanchor="start",
                customdata=np.c_[m.features, m.pct_within_3h],
                hovertemplate="<b>%{y}</b><br>median error %{x:.2f} h<br>features: "
                "%{customdata[0]}<br>%{customdata[1]:.0f}% within 3 h<extra></extra>",
            )
        )
        fig.update_xaxes(title="Median error (h) · grey = baselines, dark blue = best")
        st.plotly_chart(style(fig, 450, showlegend=False), width="stretch")
    with right:
        section("Head-to-head comparison", "Same folds, paired per person")
        pairs = [f"{r.model_b}  vs  {r.model_a}" for r in paired.itertuples()]
        default = next(
            (
                i
                for i, r in enumerate(paired.itertuples())
                if r.model_b == best.model and r.model_a == "baseline: clock time only"
            ),
            0,
        )
        choice = st.selectbox(
            "Comparison", range(len(pairs)), index=default, format_func=lambda i: pairs[i]
        )
        p = paired.iloc[choice]
        c1, c2 = st.columns(2)
        kpi(
            c1,
            "Mean change per person",
            f"{signed(p.mean_diff_h)} h",
            f"95% CI {signed(p.ci95_low_h)} to {signed(p.ci95_high_h)}",
            "ok" if p.ci95_high_h < 0 else "neutral",
        )
        kpi(
            c2,
            "People improved",
            f"{int(p.n_better)} / {int(p.n_subjects)}",
            f"Wilcoxon p = {p.wilcoxon_p:.4f}",
            "ok" if p.wilcoxon_p < 0.05 else "neutral",
        )
        st.write("")
        fig = go.Figure(
            go.Bar(
                x=[int(p.n_better), int(p.n_tied), int(p.n_worse)],
                y=["better", "tied", "worse"],
                orientation="h",
                marker_color=["#009E73", "#9ca3af", C44],
                text=[int(p.n_better), int(p.n_tied), int(p.n_worse)],
                textposition="auto",
            )
        )
        st.plotly_chart(style(fig, 190, showlegend=False), width="stretch")
        st.caption("Negative change = the first model has lower error than the second.")

    left, right = st.columns(2)
    with left:
        section("How many genes?", "Sensitivity to k, the number of genes chosen in each fold")
        kk = pd.concat([mlk, ml[ml.model.str.contains("centred genome top-50")].assign(k=50)])
        fig = go.Figure()
        for lab, mask, col in [
            ("centred genes only", ~kk.model.str.startswith("clock"), "#93c5fd"),
            ("clock time + centred genes", kk.model.str.startswith("clock"), C55),
        ]:
            d = kk[mask].sort_values("k")
            fig.add_scatter(
                x=d.k,
                y=d.median_abs_error_h,
                mode="lines+markers",
                name=lab,
                line=dict(color=col, width=3),
                marker=dict(size=10),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=d.median_ci95_high_h - d.median_abs_error_h,
                    arrayminus=d.median_abs_error_h - d.median_ci95_low_h,
                ),
            )
        fig.add_hline(
            y=base.median_abs_error_h,
            line_dash="dot",
            line_color=MUTED,
            annotation_text="clock time only",
            annotation_position="top left",
        )
        fig.update_xaxes(type="log", tickvals=[20, 50, 200], title="Genes selected (k)")
        fig.update_yaxes(title="Median error (h)")
        st.plotly_chart(
            style(fig, 380, legend=dict(orientation="h", y=-0.25, x=0)), width="stretch"
        )
    with right:
        section(
            "Which genes carry the signal?",
            "Genes picked in all 23 CV folds, ranked by rhythm strength (R²) · colour = peak phase",
        )
        top = mlgenes.sort_values(["n_folds_selected", "mean_r2"], ascending=False).head(20)
        fig = go.Figure(
            go.Bar(
                x=top.mean_r2,
                y=top.gene,
                orientation="h",
                marker=dict(
                    color=top.acrophase_h % 24,
                    colorscale="Twilight",
                    cmin=0,
                    cmax=24,
                    colorbar=dict(title="peak (h)", thickness=10),
                ),
                customdata=np.c_[top.mean_r2, top.acrophase_h % 24, top.amplitude],
                hovertemplate="<b>%{y}</b><br>chosen in %{x}/23 folds<br>R² "
                "%{customdata[0]:.2f}<br>peak phase %{customdata[1]:.1f} h<br>amplitude "
                "%{customdata[2]:.2f}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Mean cosinor R² in training folds")
        fig.update_yaxes(dtick=1)
        st.plotly_chart(style(fig, 360, showlegend=False), width="stretch")

# ------------------------------------------------------------------ data & biology
with tabs[4]:
    left, right = st.columns(2)
    with left:
        section("From GEO to the analysis set", "Genes or probes kept at each step")
        q = qc[~qc.step.str.startswith("sensitivity")]
        fig = go.Figure(
            go.Funnel(
                y=q.step.str.replace(r"^\d+ ", "", regex=True),
                x=q.features,
                texttemplate="%{value:,}",
                marker_color=["#1d4e7a", "#24609a", C55, "#3b8fd0", "#56B4E9", "#86cbee"][: len(q)],
                customdata=np.c_[q.samples, q.subjects],
                hovertemplate="%{y}<br>%{x:,} features · %{customdata[0]} samples · "
                "%{customdata[1]} people<extra></extra>",
            )
        )
        st.plotly_chart(style(fig, 380, showlegend=False), width="stretch")
    with right:
        section(
            "What drives variation in expression?",
            "PCA: share of each component explained by each factor (η²)",
        )
        d = pca_r2.rename(columns={pca_r2.columns[0]: "PC"})
        d = d[d.PC.str.startswith("PC")]
        fig = go.Figure()
        for col, lab, c in [
            ("subject", "Person", "#1d4e7a"),
            ("timepoint", "Time point", C44),
            ("circadian_phase_h", "Melatonin phase", "#E69F00"),
            ("per3_genotype", "Genotype", "#009E73"),
        ]:
            fig.add_bar(
                x=d.PC,
                y=d[col],
                name=lab,
                marker_color=c,
                hovertemplate="%{x} · " + lab + ": %{y:.0%}<extra></extra>",
            )
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="η² (share of PC variance)", tickformat=".0%")
        st.plotly_chart(style(fig, 380), width="stretch")

    section(
        "Rhythmic genes around the 24-hour clock",
        "Genome-wide cosinor scan · angle = peak (melatonin phase) · radius = amplitude · "
        "search to highlight genes",
    )
    c1, c2 = st.columns([1, 2])
    search = c1.text_input("Find genes (comma-separated)", "PER2, PER3, NR1D1, MAL, CCR7")
    amin = c1.slider("Minimum amplitude (log2)", 0.2, 0.6, 0.2, 0.05)
    hits = [s.strip().upper() for s in search.split(",") if s.strip()]
    rh = rhythmic[(rhythmic.q < 0.05) & (rhythmic.amplitude >= amin)]
    c1.metric("Rhythmic genes shown", f"{len(rh):,}")
    found = rhythmic[rhythmic.gene.isin(hits)]
    if len(found):
        c1.dataframe(
            found[["gene", "acrophase_h", "amplitude", "q"]]
            .assign(
                acrophase_h=lambda d: d.acrophase_h.round(2),
                amplitude=lambda d: d.amplitude.round(3),
                q=lambda d: d.q.map(lambda v: f"{v:.1e}"),
            )
            .rename(columns={"acrophase_h": "peak (phase h)"}),
            hide_index=True,
            width="stretch",
        )
    fig = go.Figure()
    fig.add_scatterpolar(
        r=rh.amplitude,
        theta=rh.acrophase_h % 24 * 15,
        mode="markers",
        marker=dict(
            size=5,
            color=rh.acrophase_h % 24,
            colorscale="Twilight",
            cmin=0,
            cmax=24,
            opacity=0.6,
        ),
        text=rh.gene,
        hovertemplate="<b>%{text}</b><br>amplitude %{r:.2f}<extra></extra>",
    )
    if len(found):
        fig.add_scatterpolar(
            r=found.amplitude,
            theta=found.acrophase_h % 24 * 15,
            mode="markers+text",
            text=found.gene,
            textposition="top center",
            marker=dict(size=13, color=INK, symbol="star"),
            hovertemplate="<b>%{text}</b><br>amplitude %{r:.2f}<extra></extra>",
        )
    fig.update_layout(
        polar=dict(
            bgcolor="#fbfcfe",
            radialaxis=dict(gridcolor=GRID, angle=90, tickfont=dict(size=10)),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                gridcolor=GRID,
                tickvals=list(range(0, 360, 45)),
                ticktext=[f"{h}h" for h in range(0, 24, 3)],
            ),
        )
    )
    c2.plotly_chart(
        style(fig, 520, showlegend=False, margin=dict(t=30, b=30, l=50, r=50)), width="stretch"
    )

    section(
        "Which immune cells drive the rhythm?",
        "Cell-type enrichment of rhythmic genes (Azimuth), split by when the genes peak",
    )
    ce = cells[cells.gene_set.str.startswith("Azimuth") & (cells.q < 0.05)].copy()
    ce["group"] = ce.gene_list.str.replace(r"\s*\(.*", "", regex=True).str.strip()
    ce = ce[~ce.group.str.startswith("All rhythmic")].sort_values("fold_enrichment")
    ce = ce.groupby("group").tail(6)
    ce["term_short"] = ce.term.str.replace(r" CL\d+", "", regex=True)
    fig = go.Figure()
    for grp, col in zip(ce.group.unique(), ["#E69F00", "#5B4B8A", "#009E73"], strict=False):
        d = ce[ce.group == grp]
        fig.add_bar(
            x=d.fold_enrichment,
            y=d.term_short,
            orientation="h",
            name=grp,
            marker_color=col,
            customdata=np.c_[d.overlap, d.q],
            hovertemplate="<b>%{y}</b><br>%{x:.1f}× enriched · %{customdata[0]} genes"
            "<br>q = %{customdata[1]:.1e}<extra></extra>",
        )
    fig.update_xaxes(title="Fold enrichment")
    st.plotly_chart(style(fig, 400), width="stretch")

    with st.expander(":material/info: Caveats and definitions"):
        st.markdown(
            "- **Cosinor:** y = M + A·cos(2π(t − φ)/24); φ is the peak time (acrophase).\n"
            "- **Melatonin time:** hours from each person's melatonin phase 0, in 3-h bins.\n"
            "- In this dataset melatonin phase is always 19.5 h after waking, so internal time "
            "and time awake cannot be separated.\n"
            "- Whole blood mixes the cell clock with immune cells moving in and out.\n"
            "- All participants are homozygous (no 4/5), sampled during ~40 h awake."
        )
