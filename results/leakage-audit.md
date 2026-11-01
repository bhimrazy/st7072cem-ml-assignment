# Leakage audit

Written after the feature matrix was frozen (`uv run python -m src.freeze`, 10 October 2026).
The automated checks in `src/checks.py` are run with `uv run python -m src.checks`; their output
is in `results/checks-output.txt`.

## Controls

| # | Control | Where | Checked by |
|---|---|---|---|
| 1 | Splits are by season, never random: train 2017 to 2022, validate 2023, test 2024 to 2025, check 2026 | `src/config.py` SPLITS | inspection |
| 2 | No `loss_*` field, target, same-day outcome (`day_*`) or identifier is a feature | `src/features.py` feature lists | `test_feature_lists`, `test_no_loss_columns_in_inputs` |
| 3 | Set A has no `future_*` column; set B's extra columns are all `future_*` and are labelled "perfect forecast" | `src/features.py` | `test_feature_lists` |
| 4 | Set A is unchanged when every weather value after a cutoff is scrambled and every incident not yet entered at the cutoff is removed | `src/features.py` | `test_set_a_ignores_the_future` |
| 5 | Recent-incident counts use only incidents entered by day t (entry-lag rule); pre-2020 incidents get a seeded synthetic lag drawn from the 2020 to 2025 distribution of the same hazard, so the feature means the same thing in every season | `features.harmonise_entry_lags`, `available_day` | `test_entry_lag_rule` (259 window counts recomputed by hand) |
| 6 | Rain percentiles are fitted on the training seasons per district | `fit_rain_percentiles` | `test_fits_use_training_seasons_only` |
| 7 | District x month history rates (target encoding) are fitted on training seasons; training rows get a leave-one-season-out encoding so a row never sees its own season's outcomes | `apply_target_encoding_loso` | `test_target_encoding_leave_one_season_out` |
| 8 | The persistence baseline uses incidents known at t (`recent_dam_3d > 0`), not the same-day flag | `evaluate.persistence_score` | `test_persistence_known_at_t` |
| 9 | Clustering profiles (notebook 03) use 2017 to 2022 same-day rates only; the region encodings are fitted once and then attached to every row | `src/regions.py` | inspection |
| 10 | Model tuning uses leave-one-season-out folds inside 2017 to 2022; calibration uses 2023; 2024 to 2025 is scored once. The 5% alert budget is a share of the scored test rows themselves (`evaluate.alert_metrics`), not a threshold carried over from 2023; a threshold fixed on 2023 scores is reported separately as a sensitivity (`results/alert-threshold-2023.csv`) | `src/models.py`, `src/evaluate.py`, notebooks 04 and 05 | inspection |
| 11 | The 2026 check ends its target window on 19 August 2026, 31 days before the snapshot | `src/config.py` | `check_panel` |
| 12 | The year itself is never a feature. The year-only control (season and month, no district term) sits below climatology by construction, so it only shows that year alone does not rank district-days; the stronger check adds year to a district-history control (shrunk district x month rate and month), which moves test PR-AUC from 0.136 to 0.138 | notebooks 04 and 05 | `results/test-baselines.csv`, `results/year-control-history.csv` |

## Residual risks stated in the paper

- Set A rain is ERA5 analysis of day t, available days later in practice: an analysis-based hindcast, not an operational forecast.
- The target window of consecutive issue days overlaps (lag-1 autocorrelation of y about 0.69), so tuning folds are whole seasons and bootstrap blocks are ISO weeks.
- The clustering profile uses training-season same-day rates at 77-row granularity; harmless but in-sample for training rows.
- Within the tuning folds, rain percentiles and the training rows' history encoding were fitted once on all six training seasons, so a held-out tuning season is not fully unseen. This can flatter fold scores; it cannot reach the test seasons.
- The synthetic pre-2020 entry lags are drawn from the entry-date metadata of 2020 to 2025 incidents, test seasons included. No label enters them.
- Reporting bias in BIPAD is not leakage but shapes the label (see the Discussion).
