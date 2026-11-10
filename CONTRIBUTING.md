# Contributing

Everyone works on **their own part**, on a **branch**, and merges into `main` through a **pull request**. `main` is protected: a PR needs green CI and one approving review.

## 1. One-time setup
```bash
git clone git@github.com:CKmellow/per3-genotype-clock-gene-timing.git   # or the https URL
cd per3-genotype-clock-gene-timing
python -m venv .venv && source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -m "not network"        # should pass (xfail tests = other parts' unfinished work)
```
Optional, once: `pytest -m network` downloads the ~100 MB GEO series matrix to `data/external/` and checks it.

## 2. Start your part
1. Find your issue (Part 2–5) and assign yourself.
2. Update `main`, then create your branch, named `feat/part-N-short-name`:
   ```bash
   git switch main && git pull
   git switch -c feat/part-3-cosinor
   ```
3. Work **only** in:
   - your stage notebook(s) in `notebooks/stages/`
   - your module in `src/per3clock/` (`qc.py`/`eda.py` = Part 2, `rhythm.py`/`circstats.py` = Part 3, `interpret.py` = Part 4, `ml.py` = Part 5)
   - your tests in `tests/`
   **Never edit another part's notebook or module.** If you need a change in someone else's code (or in `data.py`), ask in their issue or open a small separate PR and tag them.
4. Load data **only** through `per3clock.data` (e.g. `load_clock_genes()`, `analysis_samples()`, `load_expression_matrix()`). Use paths from `per3clock.config`. Never hard-code `C:\Users\...` or `/content/...`.
5. Implement the functions stubbed in your module. Their docstrings are the contract (inputs, outputs, column names). When a function works, **remove the `@pytest.mark.xfail(reason="Part N")` marker** from its test so it counts.
6. Save outputs with `per3clock.plotting.save_fig(fig, "name")` and `df.to_csv(config.TABLES_DIR / "name.csv")`, using the names listed in your stage notebook's *Contract* cell.

## 3. Before every push
```bash
ruff check .                 # lint (ruff check . --fix fixes most issues)
pytest -m "not network"      # must pass
```
**Clear outputs in stage notebooks before committing** (Jupyter: *Edit → Clear All Outputs*; VS Code: *Clear All Outputs*; or `jupyter nbconvert --clear-output --inplace notebooks/stages/0N_*.ipynb`). Outputs are kept only in the final generated notebook. Never commit large files or anything in `data/external/`.

Commit messages use [conventional commits](https://www.conventionalcommits.org/): `feat: add per-subject cosinor fits`, `fix: ...`, `docs: ...`, `test: ...`, `chore: ...`.

## 4. Open a pull request
```bash
git push -u origin feat/part-3-cosinor
gh pr create --base main --fill      # or use the GitHub website
```
- Write `Closes #<your issue number>` in the PR description and fill in the checklist.
- Wait for CI to go green and ask a teammate to review. Merge with **Squash and merge**, then delete the branch.
- Keep PRs small. Several PRs per part are fine.
- If `main` moved on, update your branch: `git switch main && git pull && git switch - && git merge main`.

## 5. Running in Google Colab
Open `https://colab.research.google.com/github/CKmellow/per3-genotype-clock-gene-timing/blob/<branch>/notebooks/stages/<your notebook>.ipynb`. The first cell is the **bootstrap cell**. Set `BRANCH` in it to your branch name, then run all. It clones the repo, installs the package and changes into the repo directory. Changes you make in Colab are **not** saved to GitHub. Copy them back into your local notebook (or use *File → Save a copy in GitHub* to your branch) and push from your machine.

## 6. Rules
- Don't invent data, results or citations. If something can't be verified, say so in the notebook and the PR.
- Don't change the research question or objectives. Raise concerns in the issue.
- Use `config.SEED` / `config.set_seeds()` for anything random.
- Ask before anything destructive (force-push, deleting branches or files, rewriting history).
