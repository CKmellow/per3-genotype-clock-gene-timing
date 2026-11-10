# PER3 genotype and clock-gene timing in human blood

[![CI](https://github.com/CKmellow/per3-genotype-clock-gene-timing/actions/workflows/ci.yml/badge.svg)](https://github.com/CKmellow/per3-genotype-clock-gene-timing/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CKmellow/per3-genotype-clock-gene-timing/blob/main/notebooks/PER3_clock_timing.ipynb)

University bioinformatics group project (deadline **18 Nov 2026**).

## Research question
> Does PER3 genotype (4/4 vs 5/5) shift the time of day at which the clock genes PER2, PER3 and NR1D1 reach peak expression in human blood?

**Hypothesis.** PER3 5/5 ("early bird") carriers reach peak clock-gene expression earlier in the day than 4/4 ("night owl") carriers.

**Objectives**
1. To estimate the time of peak expression of PER2, PER3 and NR1D1 in each participant.
2. To compare peak timing between the PER3 4/4 and 5/5 genotype groups and estimate the size of any shift.
3. To determine whether any shift reflects a change in the whole internal clock, using melatonin-based circadian phase.
4. To determine whether the results support the hypothesis.

**Data.** GEO [GSE39445](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39445) (Möller-Levet et al. 2013, *PNAS*): whole-blood transcriptome (Agilent GPL15331), 26 participants (12 × PER3 4/4, 14 × 5/5), ~10 samples every 3 h across ~40 h of wakefulness, after a week of sufficient sleep ("Sleep Extension", **main analysis**) and of insufficient sleep ("Sleep Restriction", **sensitivity analysis only**). See [`data/README.md`](data/README.md).

## Deliverable
One notebook, [`notebooks/PER3_clock_timing.ipynb`](notebooks/PER3_clock_timing.ipynb), that runs top-to-bottom in Google Colab with outputs saved. It is **generated** from the stage notebooks in `notebooks/stages/` by `scripts/build_notebook.py`. Do not edit it by hand.

| Stage | Notebook | Part | Owner |
|---|---|---|---|
| 1 Question & objectives | `01_question.ipynb` | 1 | Cyprian (@CKmellow) |
| 2 Data acquisition | `02_data_acquisition.ipynb` | 1 | Cyprian (@CKmellow) |
| 3 Preprocessing & QC | `03_preprocessing_qc.ipynb` | 2 | assigned after group poll |
| 4 EDA | `04_eda.ipynb` | 2 | assigned after group poll |
| 5 Core analysis | `05_core_analysis.ipynb` | 3 | assigned after group poll |
| 6 Biological interpretation | `06_interpretation.ipynb` | 4 | assigned after group poll |
| 7 Machine learning | `07_machine_learning.ipynb` | 5 | assigned after group poll |
| 8 Conclusions | `08_conclusions.ipynb` | 1 | Cyprian (@CKmellow) |

Work is tracked in [issues](https://github.com/CKmellow/per3-genotype-clock-gene-timing/issues), milestones (M1 Foundation 12 Nov · M2 Parts 2–5 16 Nov · M3 Integration & submission 18 Nov) and the project board **PER3 Clock Timing**.

## Run locally
```bash
git clone git@github.com:CKmellow/per3-genotype-clock-gene-timing.git
cd per3-genotype-clock-gene-timing
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -m "not network"          # fast offline tests (<30 s)
pytest -m network                # downloads the ~100 MB GEO series matrix once, then verifies it
jupyter lab notebooks/stages/    # or open in VS Code
```
The whole-transcriptome series matrix is downloaded on first use to `data/external/` (gitignored) by `per3clock.data.download_series_matrix()`.

## Run in Google Colab
Open any notebook via `https://colab.research.google.com/github/CKmellow/per3-genotype-clock-gene-timing/blob/main/<path-to-notebook>` (or the badge above). The **first cell** of every notebook is a bootstrap cell: in Colab it clones this repo, runs `pip install -e .` and `cd`s into it; locally it does nothing. Run it, then *Runtime → Run all*. If Colab asks to restart the runtime after installing, restart and run all again.

To test a branch in Colab, change `BRANCH = "main"` in the bootstrap cell.

## Repository layout
```
data/raw/          small source files from GEO (committed)
data/external/     large downloads, e.g. series matrix (gitignored)
data/processed/    small derived tables (<5 MB)
docs/              minutes, references.bib
notebooks/stages/  one notebook per stage (edit these)
notebooks/PER3_clock_timing.ipynb   generated final notebook
results/figures/, results/tables/   saved outputs
scripts/           build_notebook.py, verify_data.py
src/per3clock/     shared Python package (one module per part)
tests/             pytest suite (contracts for every part)
```

## Workflow
Branch from `main`, work on one issue, open a PR, merge when CI is green and one teammate has approved. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Key references
- Möller-Levet CS, Archer SN, Bucca G, et al. (2013) Effects of insufficient sleep on circadian rhythmicity and expression amplitude of the human blood transcriptome. *PNAS* 110(12):E1132–E1141. doi:10.1073/pnas.1217154110 (PMID 23440187).
- Archer SN, Robilliard DL, Skene DJ, et al. (2003) A length polymorphism in the circadian clock gene Per3 is linked to delayed sleep phase syndrome and extreme diurnal preference. *Sleep* 26(4):413–415. doi:10.1093/sleep/26.4.413 (PMID 12841365).
