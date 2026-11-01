# Lab log

Dated notes on the experiments, figures, results and paper, oldest first. Numbers are copied from
the result files in this folder.

## 10 October 2026

### Starting point
- Panel: 156,849 rows (146,223 for 2017 to 2025 plus the 2026 check). ERA5 complete (201,971
  district-days, 2016-03-02 to 2026-08-31, no nulls). Terrain for all 77 districts.
- Entry lags of damaging incidents, 2020 to 2025 (n = 3,476): 20.8% entered on the day, 74.3%
  within 1 day, 94.2% within 3 days; one negative lag, clipped to 0.

### Step 1: pipeline corrections
- `config.py`: BIPAD snapshot read from `data/raw/` first; `SEED = 42`; `RAIN_HAZARDS`;
  `ALERT_BUDGET = 0.05`; figures written to `paper/figures/`.
- `panel.py`: rain-driven target `y_rain` (4.35%), added to the reproduction check.
- `features.py`: synthetic pre-2020 entry lags per hazard; leave-one-season-out target encoding for
  training rows; `FEATURES_NOWX`; set A documented as an analysis-based hindcast.
- `evaluate.py`: climatology, known-at-t persistence, Dahal and Hasegawa 144 mm / 182 mm thresholds,
  refitted percentile threshold, metrics, weekly block bootstrap, reliability table.
- `checks.py`: four new tests. `requirements.txt` exported with uv (118 pins).
- `uv run python -m src.checks`: all checks passed (`results/checks-output.txt`).

### Step 2: feature matrix
- `uv run python -m src.freeze` gives `data/processed/feature-matrix.parquet`, 156,849 x 51, no
  missing values in set B.
- Sanity checks: climatology on test PR-AUC 0.1425, Brier 0.0760. Persistence (known at t)
  precision 0.18, recall 0.17, alert rate 8.1%. Dahal threshold fires on 1.4% of test rows (POD
  0.04). Refitted percentile cut 0.79 (best CSI on training seasons), POD 0.39 at a 21% alert rate.
- Wrote `results/leakage-audit.md`.

### Step 3: exploratory analysis (notebook 02)
- Figures: seasonality by hazard; damaging days per year (any incident x2.8 against damaging x1.4,
  2017 to 2024); district map and belt map; rain percentile against positive rate. Class balance and
  completeness tables in `results/eda-*.csv`.
- Lightning is in 38.2% of positives and the only hazard in 36.1%.

### Step 4: clustering (notebook 03)
- Profile 77 x 37; the first 4 principal components carry 49% of the variance (19 reach 90%, where
  the diagonal mixture degenerates), so the main space has 4 components.
- Rule fixed before looking at the solutions: K-means k with the best silhouette and smallest region
  of at least 5 districts, which gives k = 6 (silhouette 0.33, Davies-Bouldin 0.93).
- K-means against Ward ARI 0.75, against the mixture 0.73. Bootstrap ARI over resampled seasons:
  0.55 (K-means), 0.47 (Ward), 0.22 (mixture). ARI against belts 0.24, provinces 0.22. BIC favours
  k = 3 but is not used alone.
- Encodings saved to `data/processed/district-regions.csv`.

### Step 5: classification (notebook 04, 14.1 min)
- Expectation written down before scoring the test seasons: if the regions help at all, they help
  logistic regression more than the tree models.
- Tuning (leave-one-season-out mean PR-AUC): lr 0.136, rf 0.150 (best by the rule), hgb 0.148,
  svm_linear 0.135, svm_rbf 0.118.
- Test 2024 to 2025, set A, PR-AUC with 95% weekly block-bootstrap interval: climatology 0.143
  [0.120, 0.166]; lr 0.177 [0.142, 0.216] (difference +0.035 [+0.010, +0.066]); hgb 0.175;
  svm_linear 0.177; rf 0.165 (difference +0.023 [-0.002, +0.052]); svm_rbf 0.153.
- BSS with 2023-only calibration is about 0 or slightly negative (2023 base rate 4.4% against 8.5%
  on test; mean predicted 0.047 against observed 0.085). Pooled out-of-fold calibration on 2017 to
  2023 gives BSS 0.016 to 0.020. Both reported, the pooled one as a sensitivity.
- Set B (perfect forecast) PR-AUC 0.21 to 0.22; NOWX 0.14 to 0.16.
- Regions: no encoding helps logistic regression (-0.004 to +0.000); for the random forest the
  posterior and distance encodings add +0.004 to +0.005 (interval includes 0); a district one-hot
  adds +0.005 [+0.002, +0.008]. Objective 3 is a null result.
- Rain-driven target: climatology 0.123, lr set A 0.177, lr set B 0.229.
- 2026 check: climatology 0.087, models 0.100 to 0.108; BSS 0.011 to 0.027.
- Baselines: persistence PR-AUC 0.101 (own alerts: POD 0.17, FAR 0.82); Dahal threshold fires on
  1.4% of rows (POD 0.04); refitted threshold PR-AUC 0.150; year-only control 0.128.
- The random forest, chosen by the tuning rule, is not the best on test; the differences are inside
  the intervals. I kept the rule and report it as it is.

### District boundaries and figure style
- OCHA COD-AB polygons saved as `data/raw/npl-districts-adm2.geojson` and
  `npl-provinces-adm1.geojson` with `.meta.json` (v02, CC BY-IGO, simplified 0.0015 deg). HDX
  "Chitawan" mapped to BIPAD "Chitwan"; all 77 names match. Added geopandas and seaborn.
- `src/plotting.py`: seaborn whitegrid, STIX serif 8 pt, Okabe-Ito hazard and model colours, region
  palette, choropleth and categorical map helpers, panel letters, PDF and PNG output.
- Eleven further references checked through Crossref and added to `paper/references.bib`
  (Iglesias2024, Kratzert2019, Fang2022, Kansakar2004, Davies1979, Calinski1974, Hubert1985,
  Brier1950, Hamill2006, Kapoor2023, Kim2019a).

### Step 6: evaluation (notebook 05, 6.6 min)
- Figures: precision-recall curves, reliability, alert budget; errors by region, hazard and month;
  SHAP and permutation importance; case study of 26 to 28 September 2024 (232 damaging incidents in
  42 districts).
- Case study: set A no better than climatology on that day (PR-AUC 0.579 against 0.595); set B 0.771.
- Lightning is not predictable from daily rain (PR-AUC 0.038 against climatology 0.043; set B 0.045).
- Fixed: the case-study ranks first used isotonic probabilities, which tie; they now use raw scores
  with minimum ranks.

### Steps 7 and 8: formatting, clean run, paper
- `uvx ruff check --select I --fix src notebooks` (9 fixes) and `uvx ruff format src notebooks`
  (18 files). `[tool.ruff]` added to `pyproject.toml` (line length 100, Python 3.12, notebooks
  included). The remaining lint messages are style only.
- `src.checks` passed again after formatting.
- Clean run: 01 4 s, 02 4 s, 03 10 s, 04 871 s, 05 404 s (21.5 min); numbers identical to the
  earlier run. Notebooks 02, 03 and 05 were re-run afterwards for figure layout only
  (`results/notebook-run-times.txt`). Wrote `results/run-evidence.md`.
- Figures use a muted palette (desaturated Okabe-Ito for hazards and models, Paul Tol "light" for
  regions, trimmed YlGnBu and OrRd ramps for maps). Each figure checked as PNG and the case-study map
  also in greyscale.
- Paper compiled to `paper/main.pdf`. Secondary figures and the feature-set table moved to the
  appendix.
- Open point: the rule-selected random forest is not the best model on test (logistic regression,
  gradient boosting and the linear SVM are about 0.01 higher, inside the intervals); reported as such.

### Paper corrections (evening)
- Related Work and Table I corrected against the sources: Kooshki Forooshani 2024 is a
  tropical-cyclone housing-damage model; Sedhain 2025 evaluates tropical-cyclone anticipatory-action
  triggers; Doerksen 2023 label source taken from the abstract only; Morales 2026 is municipality,
  next hour, 112 calls, against climatology-based warnings; Iglesias 2024 is Catalonia,
  municipality-day, emergency-incident counts.
- Run times filled in from the clean run (21.6 min, notebook 04 14.5 min); email added to the
  author block.
- Bibliography: proper nouns and acronyms braced.
- Strict two-column layout: every figure and table set to one column; the main table split into a
  ranking table and a probabilistic and operational table; multi-panel figures stacked at 3.5 in;
  notebooks 03 and 05 re-executed for the new figure sizes.

## 24 October 2026

- Introduction and Limitations mention the 11 October 2026 Gorkha flood (two news sources).
- LaTeX build files ignored in git.

## 27 October 2026

### Statistics
- Paired feature-set ablation from the saved test predictions (`src/sensitivity.py`, notebook 05
  section 5; `results/ablation-paired.csv`). Set A minus NOWX: +0.011 to +0.022, interval above zero
  only for the random forest (+0.022 [+0.001, +0.045]). Set B minus set A: +0.026 to +0.054, every
  interval above zero (logistic regression +0.035 [+0.009, +0.072]).
- Paired gaps between models in set A against the random forest (`results/model-gaps-paired.csv`):
  logistic regression +0.012 [-0.001, +0.026], gradient boosting +0.010 [+0.001, +0.018], linear SVM
  +0.012 [-0.001, +0.025], RBF SVM -0.013 [-0.026, +0.002].
- Alert budget: `alert_metrics` now breaks tied scores at the cut at random with the fixed seed.
  Only climatology and the 0/1 rules have ties at the 5% cut; the models are unchanged. Climatology
  at 5%: POD 0.115, FAR 0.805, CSI 0.078 (`results/alert-budget.csv`). The POD, FAR and CSI columns
  of the notebook 04 result files keep the earlier row-order tie-break; the paper takes these
  metrics from `alert-budget.csv`.
- Rules at their own alert rate against climatology and models re-thresholded to that rate
  (`results/alert-rules-own-rate.csv`): at persistence's 8.1%, POD 0.172 (rule), 0.181
  (climatology), 0.206 (lr), 0.190 (rf).
- Threshold fixed on 2023 scores (`results/alert-threshold-2023.csv`): the random forest would alert
  6.3% of test district-days (7.7% in 2024, 4.9% in 2025), logistic regression 8.6%.
- Year control: a district-history control (shrunk district x month rate and month) moves from
  0.136 to 0.138 when year is added (`results/year-control-history.csv`).
- Leakage audit rows 10 and 12 corrected (the 5% budget is set on the scored rows; the year-only
  control has no district term), and two residual points added (tuning-fold encodings; lag metadata
  from 2020 to 2025).

### Paper
- Interpretation of the ablation corrected: reanalysis rain adds 0.01 to 0.02 over the floor, a
  perfect three-day forecast 0.03 to 0.05 more.
- Region result reworded: the regions add nothing to one pooled model that already holds
  district-by-month history; no per-region models were fitted.
- Domain corrections: Gorkha flood cause described as a suspected landslide blockage; rainfall trend
  wording; lightning seasonality cited to Sharma et al. (2022); Stanley et al. resolution; loss-record
  bias sentence; practical use recast as a readiness ranking; licences.
- Added Dahal et al. (2024), Sharma et al. (2022) and Lamichhane et al. (2025) to the references;
  news items now use the url field.
- New table of rule baselines at their own alert rate; appendix now lists the clean-run times and the
  `src.checks` output, and comes before the Acknowledgment and References.
- Belt label in the notebook 02 map changed to "ecological belt"; notebook 02 re-run (5 s).
