"""Expanding-window (rolling-origin) evaluation and a training-start sensitivity.

    uv run python -m src.rolling

Each scored season is predicted by a model fitted on every season before it: 2024 from seasons up
to 2023, 2025 up to 2024, the 2026 check up to 2025. Training starts in 2017 or in 2020 (the first
season with prompt incident entry). Hyperparameters stay at results/best-params.csv (tuned on 2017
to 2022). Everything fitted on data is refitted per origin on that origin's training seasons only:
the rain percentiles and target encoding (features rebuilt), the synthetic pre-2020 entry lags
(drawn from the 2020 to origin-1 lags), climatology, and the isotonic calibrator. There is no
calibration season: the calibrator is fitted on leave-one-season-out scores of the training seasons,
and each fold rebuilds its features on its own training seasons, so a held-out season never enters
the encoding it is scored with.

Writes results/rolling-origin.csv, results/rolling-origin-per-season.csv and
data/processed/rolling-origin-predictions.parquet.

    uv run python -m src.rolling baselines

reads the saved predictions and writes results/rolling-origin-baselines.csv: the rule baselines on
2024 to 2025 with paired differences against the rolling climatology (under a minute).

    uv run python -m src.rolling score

rescores the saved predictions (no refitting) and rewrites rolling-origin.csv and
rolling-origin-per-season.csv.

Scoring over several seasons (`E.seasonal_metrics`): each season is scored by its own fit, whose raw
scores are not on one scale, so PR-AUC and ROC-AUC are the mean of the per-season values, the 5%
budget is set within each season, and Brier and BSS pool the calibrated probabilities. Weekly blocks
are resampled within each season, with the same resamples for every model (paired).
"""

import json
import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from . import experiments as X
from . import features as Fe
from . import models as M

ORIGINS = [2024, 2025, 2026]  # scored season; training uses start .. origin - 1
STARTS = [2017, 2020]
TEST, CHECK = [2024, 2025], [C.CHECK_SEASON]
RUNS = {m: Fe.FEATURES_A for m in M.GRIDS} | {"lr_B": Fe.FEATURES_B}  # run name -> features
PRED_PATH = C.PROCESSED / "rolling-origin-predictions.parquet"
N_BOOT = 1000


def model_name(run):
    return run.split("_B")[0]


def load_params():
    t = pd.read_csv(C.RESULTS / "best-params.csv")
    return {r.model: json.loads(r.params) for r in t.itertuples()}


def load_inputs():
    """Frozen panel, raw incidents (lags not yet harmonised), weather, terrain and neighbours."""
    from .geography import neighbours
    from .panel import INCIDENTS_PATH, load_panel
    from .weather import load_terrain, load_weather

    return {
        "P": load_panel(),
        "inc": pd.read_parquet(INCIDENTS_PATH),
        "W": load_weather(),
        "T": load_terrain(),
        "nbrs": neighbours(),
    }


def build(inputs, inc, train):
    d = inputs
    return Fe.build_features(d["P"], inc, d["W"], d["T"], d["nbrs"], train)


def check_build_path(inputs):
    """The rebuild path must reproduce the frozen feature matrix for the fixed split."""
    F = build(inputs, Fe.harmonise_entry_lags(inputs["inc"]), C.TRAIN_SEASONS)
    frozen = pd.read_parquet(C.PROCESSED / "feature-matrix.parquet")
    pd.testing.assert_frame_equal(F[Fe.FEATURES_B], frozen[Fe.FEATURES_B], check_dtype=False)


def fit_and_score(F, train, scored, params):
    """Fit every run on the `train` seasons of F and return raw scores on the `scored` seasons."""
    tr, sc = F[F["season"].isin(train)], F[F["season"].isin(scored)]
    out = {}
    for run, cols in RUNS.items():
        name = model_name(run)
        m = M.fit(name, params[name], tr[cols].to_numpy(float), tr["y"].to_numpy())
        out[run] = m, M.raw_score(m, sc[cols].to_numpy(float))
    return out, sc


def run_origin(inputs, params, start, origin, log=print):
    """Scores, calibrated probabilities and climatology for one origin. Returns the scored rows."""
    t0 = time.time()
    train = list(range(start, origin))
    assert max(train) < origin
    inc = Fe.harmonise_entry_lags(
        inputs["inc"], ref_seasons=[s for s in C.LAG_REFERENCE_SEASONS if s < origin]
    )
    # Leave-one-season-out scores inside the training seasons, for the calibrator.
    oof = {run: [] for run in RUNS}
    oof_y = []
    for s in train:
        others = [t for t in train if t != s]
        fitted, sc = fit_and_score(build(inputs, inc, others), others, [s], params)
        for run in RUNS:
            oof[run].append(fitted[run][1])
        oof_y.append(sc["y"].to_numpy())
    oof_y = np.concatenate(oof_y)
    # Final fit on all training seasons, scored on the origin season.
    F = build(inputs, inc, train)
    fitted, sc = fit_and_score(F, train, [origin], params)
    out = sc[["district", "date", "season", "month", "y"]].reset_index(drop=True)
    out["start"], out["origin"] = start, origin
    out["clim"] = E.climatology_score(sc, E.fit_climatology(F, train))
    for run in RUNS:
        cal = M.Calibrated(model_name(run), params[model_name(run)])
        cal.model = fitted[run][0]
        cal.refit_calibration(np.concatenate(oof[run]), oof_y)
        out[f"score_{run}"] = fitted[run][1]
        out[f"prob_{run}"] = cal.predict_proba(sc[RUNS[run]].to_numpy(float))
    log(f"start {start}, origin {origin}: {len(train)} training seasons, {time.time() - t0:.0f} s")
    return out


def score_table(pred, seasons, n_boot=N_BOOT):
    """X.summarise over the given seasons: climatology first, then every run."""
    d = pred[pred["season"].isin(seasons)]
    named = {"climatology": (d["clim"], d["clim"])}
    named |= {run: (d[f"score_{run}"], d[f"prob_{run}"]) for run in RUNS}
    return X.summarise(
        named, d["y"].to_numpy(), d["clim"].to_numpy(), d["date"], n_boot, strata=d["season"]
    )


def summarise_all(pred, n_boot=N_BOOT):
    pooled, per_season = [], []
    for start, d in pred.groupby("start"):
        for scope, seasons in (("test_2024_2025", TEST), ("check_2026", CHECK)):
            t = score_table(d, seasons, n_boot)
            pooled.append(t.assign(start=start, scope=scope).reset_index())
        for s in TEST + CHECK:
            t = score_table(d, [s], n_boot)
            per_season.append(t.assign(start=start, season=s).reset_index())

    def front(t, keys):
        """Move the key columns to the front."""
        return t[keys + [c for c in t if c not in keys]]

    return (
        front(pd.concat(pooled, ignore_index=True), ["start", "scope", "model"]),
        front(pd.concat(per_season, ignore_index=True), ["start", "season", "model"]),
    )


def rule_baselines(inputs, pred, start=2017, n_boot=N_BOOT):
    """Rule baselines on the test seasons against the rolling climatology of `pred`, scored as in
    `score_table`.

    Persistence and the Dahal and Hasegawa thresholds have nothing fitted; the ERA5 percentile cut
    and the percentiles it reads are refitted per origin on that origin's training seasons.
    """
    parts = []
    for origin in TEST:
        train = list(range(start, origin))
        inc = Fe.harmonise_entry_lags(
            inputs["inc"], ref_seasons=[s for s in C.LAG_REFERENCE_SEASONS if s < origin]
        )
        F = build(inputs, inc, train)
        sc = F[F["season"] == origin].reset_index(drop=True)
        cut = E.fit_rain_threshold(F, train)
        parts.append(
            sc[["district", "date"]].assign(
                persistence=E.persistence_score(sc),
                dahal_threshold=E.dahal_threshold_score(sc),
                refit_threshold=E.rain_threshold_score(sc, cut)[0],
                cut=cut,
            )
        )
    d = pred[(pred["start"] == start) & pred["season"].isin(TEST)]
    d = d.merge(pd.concat(parts), on=["district", "date"], how="left", validate="one_to_one")
    assert d["persistence"].notna().all()
    names = ["climatology", "persistence", "dahal_threshold", "refit_threshold"]
    cols = {"climatology": "clim"} | {k: k for k in names[1:]}
    named = {k: (d[cols[k]], d[cols[k]]) for k in names}
    t = X.summarise(
        named, d["y"].to_numpy(), d["clim"].to_numpy(), d["date"], n_boot, strata=d["season"]
    )
    for origin, c in d.groupby("season")["cut"].first().items():
        t.loc["refit_threshold", f"cut_{origin}"] = c
    return t


def write_scores(pred):
    pooled, per_season = summarise_all(pred)
    pooled.to_csv(C.RESULTS / "rolling-origin.csv", index=False)
    per_season.to_csv(C.RESULTS / "rolling-origin-per-season.csv", index=False)
    cols = ["pr_auc", "pr_auc_diff_vs_ref", "pr_auc_diff_lo", "pr_auc_diff_hi", "bss", "bss_lo"]
    with pd.option_context(
        "display.width", 200, "display.max_columns", 20, "display.max_rows", 100
    ):
        print(pooled.set_index(["start", "scope", "model"])[cols + ["pod", "far", "csi"]].round(4))
        print(per_season.set_index(["start", "season", "model"])[cols].round(4))


def main():
    t0 = time.time()
    if sys.argv[1:] == ["score"]:
        write_scores(pd.read_parquet(PRED_PATH))
        print(f"rolling-origin rescoring: {(time.time() - t0) / 60:.1f} min")
        return
    inputs = load_inputs()
    if sys.argv[1:] == ["baselines"]:
        pred = pd.read_parquet(PRED_PATH)
        t = rule_baselines(inputs, pred)
        t.to_csv(C.RESULTS / "rolling-origin-baselines.csv")
        cols = ["pr_auc", "pr_auc_diff_vs_ref", "pr_auc_diff_lo", "pr_auc_diff_hi", "pod", "far"]
        print(t[cols + ["alert_rate"]].round(4))
        return
    check_build_path(inputs)
    print("rebuild path reproduces the frozen feature matrix")
    params = load_params()
    pred = pd.concat(
        [run_origin(inputs, params, s, o) for s in STARTS for o in ORIGINS], ignore_index=True
    )
    pred.to_parquet(PRED_PATH, index=False)
    write_scores(pred)
    print(f"rolling-origin run: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
