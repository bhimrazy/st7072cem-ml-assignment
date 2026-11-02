# Monsoon hazard prediction

Code, frozen data, results and paper for "Analysis and Prediction of Monsoon Hazards in Nepal Using
Machine Learning Approaches". For each of Nepal's 77 districts and each day from April to October,
the project forecasts whether a damaging flood, landslide, heavy-rainfall or lightning incident (a
death, missing person, injury or destroyed house) will be recorded in the next three days. Stage 1
clusters the districts into risk regions (K-means, Ward, Gaussian mixture); Stage 2 compares
logistic regression, random forest, gradient boosting and support vector machines. The main test
uses an expanding window: each of the 2024 and 2025 seasons, and a 2026 check, is scored by models
refitted on all earlier seasons from 2017. Headline result: district-by-month climatology scores
0.151 in mean per-season PR-AUC; logistic regression, gradient boosting and the linear SVM gain
about 0.026 (lower bounds near 0.005), all of it in 2024, and no model clearly beats climatology in
2025 or the 2026 check. A perfect three-day rain forecast raises PR-AUC to about 0.21, and the risk
regions add nothing to a pooled model.

Author: Bhimraj Yadav, ST7072CEM Machine Learning, Softwarica College / Coventry University.

## Quick start

1. Install [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and every package for you:

   ```bash
   # macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Get the code and install the environment (versions are pinned in `uv.lock`):

   ```bash
   git clone https://github.com/bhimrazy/st7072cem-ml-assignment.git
   cd st7072cem-ml-assignment
   uv sync
   ```

3. Run the leakage and sanity checks (under a minute; the last line should read `all checks passed`):

   ```bash
   uv run python -m src.checks
   ```

4. Run the notebooks in order. Open them with `uv run jupyter lab`, or run each one from the
   command line:

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01-data-preparation.ipynb
   ```

   | Notebook | What it does | Time |
   |---|---|---|
   | `01-data-preparation` | Builds the district-day panel and the feature matrix | under 1 min |
   | `02-exploratory-analysis` | Seasonality, trends, district maps, class balance | about 20 s |
   | `03-clustering-risk-regions` | Stage 1: risk regions by clustering | about 10 s |
   | `04-classification` | Stage 2 on the fixed split: tuning, calibration, ablations, 2026 check | about 16 min |
   | `05-evaluation-and-interpretation` | Curves, errors, SHAP, case study, sensitivity checks | 10 to 25 min |

5. Run the expanding-window evaluation that produces the main results (refits every model at each
   origin; about 18 minutes), then the rule baselines and the main-result figure:

   ```bash
   uv run python -m src.rolling
   uv run python -m src.rolling baselines
   uv run python -m src.plotting
   ```

   `uv run python -m src.rolling score` recomputes the result tables from the saved predictions
   without refitting.

   Times are from an Apple M4 laptop. Metric tables go to `results/`, figures to `paper/figures/`
   and intermediate data to `data/processed/`. Every random step uses `SEED = 42` in
   `src/config.py`, so a rerun reproduces the same numbers.

Using pip instead of uv: create a Python 3.12 virtual environment, run
`pip install -r requirements.txt`, and use `python` wherever this README says `uv run python`.

## Project layout

```
st7072cem-ml-assignment/
  src/                 Python modules; the notebooks call these
    config.py          paths, seasons, splits, hazards, seed, alert budget
    geography.py       BIPAD geography: incident to district link, district neighbours
    panel.py           district-day panel and targets
    weather.py         Open-Meteo ERA5 weather pull and ward elevations (optional regeneration)
    features.py        feature sets A, B and NOWX; entry-lag rule; target encoding
    freeze.py          writes data/processed/feature-matrix.parquet from the command line
    checks.py          leakage and sanity checks
    regions.py         Stage 1 clustering helpers
    models.py          classifiers, leave-one-season-out tuning, isotonic calibration
    experiments.py     fit, calibrate and summarise one experiment (notebook 04)
    evaluate.py        baselines, metrics, weekly block bootstrap
    sensitivity.py     sensitivity checks from the saved predictions (notebook 05)
    rolling.py         expanding-window evaluation (main results) and rule baselines
    plotting.py        figure style and map helpers
    tables.py          writes paper/tables/*.tex from results/*.csv
  notebooks/           01 to 05, run in order; saved with their outputs
  data/
    raw/               frozen inputs: BIPAD snapshot and geography, ward elevations, boundaries
    weather/           ERA5 daily weather at the 77 district centroids, 2016 to 2026
    processed/         panel, feature matrix, regions, predictions (built by the notebooks and src.rolling)
  results/             metric tables (CSV), leakage audit, run evidence, lab log
  paper/               IEEE paper: main.tex, references.bib, figures/, tables/, main.pdf
  pyproject.toml       project settings and pinned dependencies
  uv.lock              exact versions for uv
  requirements.txt     the same versions for pip
```

## Data

| Input | Source and licence | File |
|---|---|---|
| Disaster incidents (63,549 records) | BIPAD portal, NDRRMA, Government of Nepal; public, licence not stated | `data/raw/bipad-incidents-2026-09-19.csv` |
| Ward, municipality and district tables, incident to ward link | BIPAD portal API, same terms | `data/raw/bipad-*.json` |
| Daily weather at district centroids | ERA5 reanalysis via the Open-Meteo historical API, CC BY 4.0 | `data/weather/era5-daily-districts.csv` |
| Ward-centroid elevation | Copernicus DEM GLO-90 via the Open-Meteo elevation API | `data/raw/open-meteo-ward-elevation.csv` |
| District and province polygons (maps only) | OCHA COD-AB Nepal (HDX), CC BY-IGO | `data/raw/npl-*.geojson` |

Each file has a `.meta.json` next to it with its URL and download date.

- **Frozen inputs.** The BIPAD incident record is a live database that changes over time, so the
  analysis uses a snapshot taken on 19 September 2026. The geography tables, weather and elevation
  were downloaded once on 10 October 2026 and are never refreshed, so every run sees the same
  data. Nothing in the quick start needs the network.
- **Regenerating inputs (optional).** `uv run python -m src.weather --help` describes the weather
  and elevation pulls. They are slow because of Open-Meteo's free-tier quota (the full weather pull
  takes two to three days) and they cache raw responses in `data/cache/`, which is not tracked.
- **Trained models are not included.** `data/processed/models/` (about 140 MB) is ignored by git.
  Notebook 04 rebuilds it, and notebook 05 needs it.

## Reproducing the paper

The paper's tables are generated from `results/`, so they always match the notebooks:

```bash
uv run python -m src.tables
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

This needs a LaTeX distribution with the `IEEEtran` class (TeX Live, MacTeX or TinyTeX). Without
one, upload the `paper/` folder to [Overleaf](https://www.overleaf.com/) and compile with pdfLaTeX.

## Making the submission zip

```bash
git archive --format=zip -o Bhimraj_Yadav_17624402_Code.zip HEAD
```

The zip holds exactly the files tracked in git at the current commit, so it never picks up the
virtual environment, caches, LaTeX build files or the trained models. Commit any changes first.
Add `--prefix=st7072cem-ml-assignment/` to put everything inside one top-level folder.

## Troubleshooting

- **`uv: command not found`.** Open a new terminal after installing uv so the updated `PATH` is
  loaded, or follow the shell setup printed by the installer.
- **`ModuleNotFoundError` in a notebook.** The notebook is running on a different Python. Start
  Jupyter with `uv run jupyter lab`, or in VS Code pick the interpreter at `.venv/bin/python`.
- **Notebook 05 fails with `FileNotFoundError: ... stage2-models.joblib`.** The trained models are
  not in git. Run notebook 04 first (about 16 minutes), then notebook 05.
- **Notebooks 04 and 05 seem stuck.** They are slow, not stuck: notebook 04 tunes five models over
  six folds and runs 1,000-resample bootstraps, and notebook 05 computes SHAP values. Close other
  heavy programs; the timings above are from a laptop with 24 GB of memory.
- **`panel.check` fails in notebook 01.** It asserts the expected row counts and rates on purpose.
  Make sure `data/raw/bipad-incidents-2026-09-19.csv` is present and unchanged.
