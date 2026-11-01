# Run evidence

Final clean execution of the whole pipeline on 10 October 2026, after `ruff` formatting of `src/`
and the notebooks. Every notebook was executed top to bottom from a fresh kernel with
`uv run jupyter nbconvert --to notebook --execute --inplace`; the executed notebooks keep their
cell outputs and are the primary run record. Every random element uses `config.SEED = 42`.

## Environment

| Item | Value |
|---|---|
| Machine | Apple M4, 24 GB, macOS 26.3 |
| Python | 3.12.12 (uv 0.11.14, `uv sync` from `uv.lock`) |
| pandas / numpy / pyarrow | 3.0.6 / 2.5.3 / 26.0.0 |
| scikit-learn / shap | 1.9.1 / 0.53.0 |
| matplotlib / seaborn / geopandas | 3.11.2 / 0.13.2 / 1.2.0 |
| nbconvert / jupyter_client | 7.17.1 / 8.10.0 |
| ruff (via uvx) | 0.17.0 |
| pdfTeX (paper) | TeX Live 2026 via TinyTeX, pdfTeX 1.40.29 |

## Steps and run times

| Step | Command | Wall time |
|---|---|---|
| Panel | `uv run python -m src.panel` | about 10 s |
| Feature matrix | `uv run python -m src.freeze` | about 5 s |
| Leakage checks | `uv run python -m src.checks` | about 10 s; output below |
| Notebook 01-data-preparation | `nbconvert --execute` | 4 s (exit 0, finished 2026-10-10 20:13:57) |
| Notebook 02-exploratory-analysis | `nbconvert --execute` | 4 s (exit 0, finished 2026-10-10 20:14:01) |
| Notebook 03-clustering-risk-regions | `nbconvert --execute` | 10 s (exit 0, finished 2026-10-10 20:14:11) |
| Notebook 04-classification | `nbconvert --execute` | 871 s (exit 0, finished 2026-10-10 20:28:42) |
| Notebook 05-evaluation-and-interpretation | `nbconvert --execute` | 404 s (exit 0, finished 2026-10-10 20:35:26) |
| Notebook 02-exploratory-analysis | `nbconvert --execute` | 4 s (exit 0, finished 2026-10-10 20:35:51) |
| Notebook 03-clustering-risk-regions | `nbconvert --execute` | 11 s (exit 0, finished 2026-10-10 20:36:02) |
| Notebook 02-exploratory-analysis | `nbconvert --execute` | 4 s (exit 0, finished 2026-10-10 20:36:57) |
| Notebook 03-clustering-risk-regions | `nbconvert --execute` | 11 s (exit 0, finished 2026-10-10 20:44:28) |
| Notebook 03-clustering-risk-regions | `nbconvert --execute` | 11 s (exit 0, finished 2026-10-10 20:46:14) |
| Notebook 03-clustering-risk-regions | `nbconvert --execute` | 11 s (exit 0, finished 2026-10-10 20:46:58) |
| Notebook 03-clustering-risk-regions | `nbconvert --execute` | 12 s (exit 0, finished 2026-10-10 20:47:21) |
| Notebook 05-evaluation-and-interpretation | `nbconvert --execute` | 405 s (exit 0, finished 2026-10-10 20:51:13) |
| Notebook 05-evaluation-and-interpretation | `nbconvert --execute` | 1360 s (exit 0, finished 2026-10-10 21:16:05) |
| All five notebooks, clean run (first five lines) | | 1293 s (21.6 min) |
| Paper tables | `uv run python -m src.tables` | 1 s |
| Paper | `pdflatex main && bibtex main && pdflatex main && pdflatex main` | about 1 min |

Notebook 04 holds the leave-one-season-out tuning of five models (6 folds each), 30 fitted and
calibrated models for the ablations and 1,000-resample weekly block bootstraps for every table.
Notebook 05 holds SHAP on the random forest (4,000 rows) and permutation importance (5 repeats).

## Later partial runs (27 October 2026)

Section 5 of notebook 05 (sensitivity checks computed from the saved test predictions; no model is
refitted) was added after the clean run and executed on its own: the setup cell and the two new
cells, in one kernel with `nbclient`, 219 s. Notebook 02 was re-run with `nbconvert` (5 s) after
its belt-map label changed. Both are listed at the end of `results/notebook-run-times.txt`. The
section 5 outputs are `results/ablation-paired.csv`, `model-gaps-paired.csv`, `alert-budget.csv`,
`alert-rules-own-rate.csv`, `alert-threshold-2023.csv` and `year-control-history.csv`.

## Formatting

`uvx ruff check --select I --fix src notebooks` (import order) and `uvx ruff format src notebooks`
were run before this final execution; `[tool.ruff]` in `pyproject.toml` sets line length 100 and
target Python 3.12 and includes the notebooks. `src.checks` was re-run after formatting and the
notebooks were re-executed; nothing changed in the results.

## Leakage checks (`uv run python -m src.checks`)

```
panel (156849, 20), incidents (16790, 8), terrain (77, 5)
ok  panel reproduces the reference numbers
ok  feature lists: no loss_*/target columns, set A has no future_*, NOWX has no weather
ok  no loss_* columns in panel or incidents table
ok  set A unchanged when the future is scrambled
  259 recent-incident window counts recomputed by hand
ok  entry-lag rule: known-at-t counts, synthetic pre-2020 lags
ok  percentiles and target encoding use training seasons only
ok  leave-one-season-out target encoding for training rows
ok  persistence baseline uses known-at-t incidents only
  real ERA5 coverage of panel rows: 100.00%
  real ERA5 rain features on cached seasons: (171710, 14), mean 7-day rain 71.2 mm
ok  real-weather smoke test
all checks passed
```

## Outputs produced

- `results/*.csv`: all metric tables; `results/leakage-audit.md`; `results/progress-log.md`.
- `paper/figures/*.pdf` and `*.png`; `paper/tables/*.tex` (from `src.tables`).
- `data/processed/feature-matrix.parquet`, `district-regions.csv`, `predictions.parquet`,
  `models/stage2-models.joblib`.

## Clean rerun in the standalone repository (1 November 2026)

After the code tidy-up (docstrings, comments, lambda to function, unused imports; no change to any
algorithm, parameter or seed), `uv run python -m src.checks` printed the same output as above, and
the five notebooks were executed in order with `nbconvert --execute` from a fresh `uv sync --frozen`
(9 s, 17 s, 11 s, 980 s and 628 s; also appended to `notebook-run-times.txt`). Notebook 01 now also
writes `feature-matrix.parquet`, the same file `src.freeze` writes.

Compared with the earlier run, every parquet file, the paper tables and all PR-AUC, ROC-AUC, Brier
and BSS values with their intervals are identical. The 5% alert-budget POD, FAR and CSI in the
notebook 04 tables changed for climatology (and the CSI differences measured against it), because
those tables were last written on 10 October, before tied scores at the cut were broken at random
(`evaluate.alert_metrics`). The rerun brings them in line with `alert-budget.csv` and the paper.
Fig. 4(c) changes for the same reason, and the SHAP beeswarm in Fig. 5(a) differs only in point
draw order, which `shap.summary_plot` shuffles without a seed.
