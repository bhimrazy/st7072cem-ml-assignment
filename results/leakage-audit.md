# Leakage audit

Written after the feature matrix was frozen (`uv run python -m src.freeze`, 10 October 2026).
The automated checks in `src/checks.py` are run with `uv run python -m src.checks`; their output
is in `results/checks-output.txt`.

## Controls

| # | Control | Where | Checked by |
|---|---|---|---|
| 1 | Seasons are never split. Main evaluation (expanding window, `src/rolling.py`): 2024 is scored by models fitted on 2017 to 2023, 2025 on 2017 to 2024, the 2026 check on 2017 to 2025. Diagnostic analyses (notebooks 04 and 05) use the fixed split: train 2017 to 2022, validate 2023, test 2024 to 2025, check 2026 | `src/rolling.py` ORIGINS (asserts every training season precedes the scored one); `src/config.py` SPLITS | inspection |
| 2 | No `loss_*` field, target, same-day outcome (`day_*`) or identifier is a feature | `src/features.py` feature lists | `test_feature_lists`, `test_no_loss_columns_in_inputs` |
| 3 | Set A has no `future_*` column; set B's extra columns are all `future_*` and are labelled "perfect forecast" | `src/features.py` | `test_feature_lists` |
| 4 | Set A is unchanged when every weather value after a cutoff is scrambled and every incident not yet entered at the cutoff is removed | `src/features.py` | `test_set_a_ignores_the_future` |
| 5 | Recent-incident counts use only incidents entered by day t (entry-lag rule); pre-2020 incidents get a seeded synthetic lag drawn from the 2020 to 2025 distribution of the same hazard (fixed split; in the expanding window from 2020 up to the season before the origin), so the feature means the same thing in every season | `features.harmonise_entry_lags`, `available_day` | `test_entry_lag_rule` (259 window counts recomputed by hand) |
| 6 | Rain percentiles are fitted on the training seasons per district | `fit_rain_percentiles` | `test_fits_use_training_seasons_only` |
| 7 | District x month history rates (target encoding) are fitted on training seasons; training rows get a leave-one-season-out encoding so a row never sees its own season's outcomes | `apply_target_encoding_loso` | `test_target_encoding_leave_one_season_out` |
| 8 | The persistence baseline uses incidents known at t (`recent_dam_3d > 0`), not the same-day flag | `evaluate.persistence_score` | `test_persistence_known_at_t` |
| 9 | Clustering profiles (notebook 03) use 2017 to 2022 same-day rates only; the region encodings are fitted once and then attached to every row | `src/regions.py` | inspection |
| 10 | Model tuning uses leave-one-season-out folds inside 2017 to 2022, and the chosen settings are not retuned per origin. In the expanding window the isotonic calibrator is fitted on leave-one-season-out scores within the origin's training seasons; in the fixed split, calibration uses 2023. 2024 to 2025 is scored once per design. The 5% alert budget is a share of the scored test rows themselves (`evaluate.alert_metrics`), not a threshold carried over from 2023; a threshold fixed on 2023 scores is reported separately as a sensitivity (`results/alert-threshold-2023.csv`). Under the expanding window (from 1 November 2026) each season is scored by its own fit, so the budget is set within each season, PR-AUC and ROC-AUC over 2024 to 2025 are means of the per-season values, and the week-block bootstrap resamples within each season (`evaluate.seasonal_metrics`, `block_bootstrap(strata=...)`) | `src/models.py`, `src/evaluate.py`, notebooks 04 and 05 | inspection |
| 11 | The 2026 check ends its target window on 19 August 2026, 31 days before the snapshot | `src/config.py` | `check_panel` |
| 12 | The year itself is never a feature. The year-only control (season and month, no district term) sits below climatology by construction, so it only shows that year alone does not rank district-days; the stronger check adds year to a district-history control (shrunk district x month rate and month), which moves test PR-AUC from 0.136 to 0.138 | notebooks 04 and 05 | `results/test-baselines.csv`, `results/year-control-history.csv` |
| 13 | Every data-fitted quantity is refitted per origin of the expanding window on that origin's training seasons only: rain percentiles, district x month history rates (features rebuilt), the synthetic pre-2020 entry lags, climatology, the refitted ERA5 threshold and the calibrator. Before refitting, the per-origin build path is asserted to reproduce the frozen feature matrix for the fixed split | `src/rolling.py` (`run_origin`, `rule_baselines`, `check_build_path`) | `check_build_path` assertion |

## Residual risks stated in the paper

- Set A rain is ERA5 analysis of day t, available days later in practice: an analysis-based hindcast, not an operational forecast.
- The target window of consecutive issue days overlaps (lag-1 autocorrelation of y about 0.69), so tuning folds are whole seasons and bootstrap blocks are ISO weeks.
- The clustering profile uses training-season same-day rates at 77-row granularity; harmless but in-sample for training rows.
- Within the tuning folds, rain percentiles and the training rows' history encoding were fitted once on all six training seasons, so a held-out tuning season is not fully unseen. This can flatter fold scores; it cannot reach the test seasons. The expanding window's calibration folds do not share this (fix 2 below).
- In the fixed-split model (diagnostic analyses only), the synthetic pre-2020 entry lags are drawn from the entry-date metadata of 2020 to 2025 incidents, test seasons included. No label enters them. The expanding window removes this (fix 1 below).
- Known-at-t counts use entry dates as they stand in the 19 September 2026 snapshot; incidents deleted or re-dated before the snapshot are invisible to the entry-lag check.
- Reporting bias in BIPAD is not leakage but shapes the label (see the Discussion).

## Expanding-window evaluation (1 November 2026)

The paper's main results now come from `src/rolling.py`: each scored season (2024, 2025, the 2026
check) is predicted by models fitted on every season before it, starting in 2017 (2020 as a
sensitivity), with the hyperparameters of `best-params.csv`. Everything fitted on data is refitted
per origin (control 13). The run removed two leaks that the fixed split had:

1. Synthetic entry lags. `features.harmonise_entry_lags` now takes `ref_seasons`; each origin draws
   the pre-2020 lags only from incidents of 2020 up to the season before the origin, so no scored
   season's entry-date metadata reaches its own features.
2. Calibration and fold encodings. There is no calibration season. The isotonic calibrator is fitted
   on leave-one-season-out scores within the origin's training seasons, and each fold rebuilds its
   rain percentiles and history encoding on its own training seasons, so a held-out season never
   enters the encoding it is scored with.

The rule baselines of the main table (`results/rolling-origin-baselines.csv`) are scored against the
rolling climatology; persistence and the Dahal and Hasegawa thresholds have nothing fitted, and the
ERA5 percentile cut is refitted per origin.
