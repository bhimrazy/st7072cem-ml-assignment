# Run evidence

A clean execution of the whole pipeline. Every notebook was executed top to bottom, after a fresh
`uv sync --frozen`, from a fresh kernel with `uv run jupyter nbconvert --to notebook --execute --inplace`; the executed notebooks
keep their cell outputs and are the primary run record. Every random element uses
`config.SEED = 42`, and repeated runs reproduce the result tables exactly.

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
| Environment | `uv sync` | a few seconds |
| Panel | `uv run python -m src.panel` | about 10 s |
| Feature matrix | `uv run python -m src.freeze` | about 5 s |
| Leakage checks | `uv run python -m src.checks` | about 10 s; output below |
| Notebook 01-data-preparation | `nbconvert --execute` | 9 s |
| Notebook 02-exploratory-analysis | `nbconvert --execute` | 17 s |
| Notebook 03-clustering-risk-regions | `nbconvert --execute` | 11 s |
| Notebook 04-classification | `nbconvert --execute` | 980 s |
| Notebook 05-evaluation-and-interpretation | `nbconvert --execute` | 628 s |
| All five notebooks | | 1,645 s (27.4 min) |
| Expanding-window evaluation | `uv run python -m src.rolling` | about 18 min |
| Rule baselines (expanding window) | `uv run python -m src.rolling baselines` | under 1 min |
| Rescoring only, from saved predictions | `uv run python -m src.rolling score` | about 15 min |
| Main-result figure | `uv run python -m src.plotting` | a few seconds |
| Paper tables | `uv run python -m src.tables` | 1 s |
| Paper | `pdflatex main && bibtex main && pdflatex main && pdflatex main` | about 1 min |

Notebook 04 holds the leave-one-season-out tuning of five models (6 folds each), 30 fitted and
calibrated models for the ablations and 1,000-resample weekly block bootstraps for every table.
Notebook 05 holds SHAP on the random forest (4,000 rows), permutation importance (5 repeats) and
the sensitivity checks computed from the saved test predictions. `src.rolling` refits every model
at each origin of the expanding window and writes `data/processed/rolling-origin-predictions.parquet`
and the `results/rolling-origin*.csv` tables; `src.rolling score` recomputes those tables from the
saved predictions without refitting.

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

## Expanding-window results (`uv run python -m src.rolling`)

Mean per-season PR-AUC over 2024 and 2025, paired difference from climatology with its 95%
season-stratified weekly block-bootstrap interval, BSS and its lower bound, and POD, FAR and CSI at
the 5% alert budget (set within each season). Training starts in 2017.

```
model         pr_auc  diff     diff_lo  diff_hi  bss     bss_lo   pod     far     csi
climatology   0.1514  0.0000   0.0000   0.0000   0.0000  0.0000   0.1182  0.7999  0.0803
lr            0.1773  0.0259   0.0051   0.0489   0.0166  0.0030   0.1433  0.7574  0.0990
rf            0.1687  0.0174  -0.0052   0.0407   0.0132  0.0009   0.1157  0.8042  0.0784
hgb           0.1788  0.0274   0.0041   0.0509   0.0185  0.0044   0.1382  0.7660  0.0952
svm_linear    0.1774  0.0261   0.0052   0.0488   0.0167  0.0039   0.1386  0.7654  0.0954
svm_rbf       0.1524  0.0011  -0.0166   0.0201   0.0062 -0.0055   0.1102  0.8134  0.0744
lr_B          0.2074  0.0560   0.0155   0.1050   0.0348  0.0071   0.1753  0.7032  0.1239
```

The per-season, 2026 check and training-start-2020 results are in `results/rolling-origin.csv` and
`results/rolling-origin-per-season.csv`; the rule baselines are in
`results/rolling-origin-baselines.csv`.

## Outputs produced

- `results/*.csv`: all metric tables; `results/leakage-audit.md`.
- `paper/figures/*.pdf` and `*.png`; `paper/tables/*.tex` (from `src.tables`).
- `data/processed/feature-matrix.parquet`, `district-regions.csv`, `predictions.parquet`,
  `rolling-origin-predictions.parquet`, `models/stage2-models.joblib`.
