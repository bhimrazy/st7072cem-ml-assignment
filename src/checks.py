"""Leakage and sanity checks for the panel and the feature sets.

    uv run python -m src.checks

Weather-dependent checks run on synthetic rain (so they cover every district and season) and on
the frozen ERA5 archive in data/weather/. Each check raises AssertionError on failure.
"""

import numpy as np
import pandas as pd

from . import config as C
from . import features as Fe
from .geography import neighbours
from .panel import TARGETS, load_panel
from .panel import check as check_panel
from .weather import load_terrain, load_weather

CUTOFF = pd.Timestamp("2021-07-15")  # an arbitrary issue day inside a training season


def synthetic_weather(P, seed=0):
    """Random daily weather for every district, 2 March to 31 October of each panel season."""
    rng = np.random.default_rng(seed)
    days = pd.DatetimeIndex(
        [d for y in sorted(P["season"].unique()) for d in pd.date_range(f"{y}-03-02", f"{y}-10-31")]
    )
    W = pd.MultiIndex.from_product(
        [sorted(P["district"].unique()), days], names=["district", "date"]
    ).to_frame(index=False)
    W["precipitation_sum"] = rng.gamma(0.6, 12, len(W)).round(1)
    W["precipitation_hours"] = rng.integers(0, 25, len(W))
    W["temperature_2m_max"] = rng.normal(25, 4, len(W)).round(1)
    return W


def test_feature_lists():
    """No loss_*, target, same-day outcome or identifier column is a feature; set A has no future_* column."""
    banned = set(TARGETS) | {"district", "date", "season", "split", "year"}
    for name in Fe.FEATURES_B:
        assert not name.startswith("loss_"), name
        assert not name.startswith("day_") and not name.startswith("y_") and name not in banned, (
            name
        )
    assert not [c for c in Fe.FEATURES_A if c.startswith("future_")], "future column in set A"
    extra = [c for c in Fe.FEATURES_B if c not in Fe.FEATURES_A]
    assert extra and all(c.startswith("future_") for c in extra), extra
    assert len(set(Fe.FEATURES_B)) == len(Fe.FEATURES_B), "duplicate feature names"
    weather = set(Fe.RAIN_A) | set(Fe.FUTURE)
    assert not weather & set(Fe.FEATURES_NOWX), "weather column in the no-weather set"
    assert set(Fe.FEATURES_NOWX) < set(Fe.FEATURES_A)


def test_no_loss_columns_in_inputs(P, inc):
    """Loss counts define the labels only; they must not survive into the panel or incidents."""
    assert not [c for c in P.columns if c.startswith("loss_")]
    assert not [c for c in inc.columns if c.startswith("loss_")]


def test_set_a_ignores_the_future(P, inc, W, T, nbrs):
    """Scramble all weather after CUTOFF and drop incidents not yet known at CUTOFF: set A up to CUTOFF must not change."""
    full = Fe.build_features(P, inc, W, T, nbrs, C.TRAIN_SEASONS, loso=False)
    rain_pct = Fe.fit_rain_percentiles(full, C.TRAIN_SEASONS)
    te = Fe.fit_target_encoding(P, C.TRAIN_SEASONS)

    W2 = W.copy()
    late = pd.to_datetime(W2["date"]) > CUTOFF
    W2.loc[late, "precipitation_sum"] = 999.0
    W2.loc[late, "precipitation_hours"] = 24
    inc2 = inc[Fe.available_day(inc) <= CUTOFF]

    def set_a(Pw, Iw, Ww):  # same fitted objects, so only the raw inputs differ
        F = Fe.rain_features(Pw, Ww)
        F = Fe.apply_rain_percentiles(F, rain_pct)
        F = Fe.terrain_features(Fe.season_features(F), T)
        F = Fe.recent_incident_features(F, Iw, nbrs)
        F = Fe.apply_target_encoding(F, te)
        return F.sort_values(["district", "date"]).reset_index(drop=True)

    a, b = set_a(P, inc, W), set_a(P, inc2, W2)
    mask = a["date"] <= CUTOFF
    pd.testing.assert_frame_equal(a.loc[mask, Fe.FEATURES_A], b.loc[mask, Fe.FEATURES_A])
    # Sanity: the future columns do see the change (they are meant to).
    day_before = a["date"] == CUTOFF
    assert (a.loc[day_before, "future_rain_3d"] != b.loc[day_before, "future_rain_3d"]).all()
    return full


def test_entry_lag_rule(F, inc, raw, n_sample=200):
    """Nothing entered after the issue day may count, in any season; synthetic lags match the reference lags."""
    dam = inc[inc["damaging"]]
    avail = Fe.available_day(dam)
    assert (avail >= dam["date"]).all(), "an incident is available before it happened"
    assert inc["lag_synthetic"].eq(inc["date"].dt.year < C.PROMPT_ENTRY_FROM).all()
    # From 2020 the real entry dates are untouched.
    post = inc["date"].dt.year >= C.PROMPT_ENTRY_FROM
    assert inc.loc[post, "created"].equals(raw.loc[post, "created"]), (
        "post-2020 entry dates changed"
    )
    # Pre-2020 synthetic lags come from the 2020 to 2025 distribution: same quantiles within tolerance.
    lag = Fe.entry_lag_days(inc)
    ref = lag[inc["date"].dt.year.isin(C.LAG_REFERENCE_SEASONS)]
    syn = lag[inc["lag_synthetic"]]
    assert (
        abs(syn.mean() - ref.mean()) < 0.5 and abs((syn <= 1).mean() - (ref <= 1).mean()) < 0.05
    ), (syn.mean(), ref.mean())
    assert (lag[inc["lag_synthetic"]] >= 0).all()
    # Recount: for a sample of late-entered incidents, the 3-day window on the incident day excludes them
    # and the window on the entry day includes them (if both rows exist in the panel).
    late = dam[avail > dam["date"]].sample(
        min(n_sample, int((avail > dam["date"]).sum())), random_state=C.SEED
    )
    key = F.set_index(["district", "date"])["recent_dam_3d"]
    checked = 0
    for _, row in late.iterrows():
        d, t0, t1 = row["district"], row["date"], Fe.available_day(dam.loc[[row.name]]).iloc[0]
        for t in (t0, t1):
            if (d, t) not in key.index:
                continue
            window = dam[
                (dam["district"] == d)
                & dam["date"].between(t - pd.Timedelta(days=3), t)
                & (avail <= t)
            ]
            assert key.loc[(d, t)] == len(window), (d, t, key.loc[(d, t)], len(window))
            checked += 1
    assert checked >= n_sample // 2, f"only {checked} window counts checked"
    print(f"  {checked} recent-incident window counts recomputed by hand")


def test_target_encoding_leave_one_season_out(P):
    """A training row's encoding must not change when its own season's outcomes change."""
    F = P.copy()
    F["rain_3d"] = np.nan
    a = Fe.apply_target_encoding_loso(F, P, C.TRAIN_SEASONS)
    P2 = P.copy()
    P2.loc[
        P2["season"] == 2019,
        ["day_damaging", "day_flood", "day_landslide", "day_heavy_rain", "day_lightning"],
    ] = True
    b = Fe.apply_target_encoding_loso(F, P2, C.TRAIN_SEASONS)
    own = a["season"] == 2019
    pd.testing.assert_frame_equal(a.loc[own, Fe.HISTORY], b.loc[own, Fe.HISTORY])
    assert (a.loc[~own, "te_dam"] != b.loc[~own, "te_dam"]).mean() > 0.5, (
        "other seasons should see the change"
    )
    # Test rows use all six training seasons: fitted on the full training set.
    full = Fe.apply_target_encoding(
        F[F["split"] == "test"], Fe.fit_target_encoding(P, C.TRAIN_SEASONS)
    )
    pd.testing.assert_frame_equal(a.loc[a["split"] == "test", Fe.HISTORY], full[Fe.HISTORY])


def test_persistence_known_at_t(F, inc):
    """The persistence baseline must only use incidents already entered by the issue day."""
    from .evaluate import persistence_score

    s = persistence_score(F)
    assert set(np.unique(s)) <= {0, 1}
    dam = inc[inc["damaging"]]
    late = dam[Fe.available_day(dam) > dam["date"] + pd.Timedelta(days=3)]
    key = F.set_index(["district", "date"])
    hits = 0
    for _, row in late.head(50).iterrows():
        k = (row["district"], row["date"])
        if k in key.index and key.loc[k, "recent_dam_3d"] == 0:
            assert s[key.index.get_loc(k)] == 0
            hits += 1
    assert hits > 0


def test_fits_use_training_seasons_only(F, P):
    """Changing test-season data must not change anything fitted on the training seasons."""
    F2 = F.copy()
    F2.loc[F2["season"].isin(C.SPLITS["test"]), "rain_3d"] = 12345.0
    a, b = Fe.fit_rain_percentiles(F, C.TRAIN_SEASONS), Fe.fit_rain_percentiles(F2, C.TRAIN_SEASONS)
    assert all(np.array_equal(a[d], b[d]) for d in a)
    P2 = P.copy()
    P2.loc[P2["season"].isin(C.SPLITS["test"]), "day_damaging"] = True
    pd.testing.assert_frame_equal(
        Fe.fit_target_encoding(P, C.TRAIN_SEASONS)["cells"],
        Fe.fit_target_encoding(P2, C.TRAIN_SEASONS)["cells"],
    )


def test_real_weather_smoke(P, inc, T, nbrs):
    """Build features from the frozen ERA5 archive: no set B feature may be missing."""
    W = load_weather()
    F = Fe.build_features(P, inc, W, T, nbrs, C.TRAIN_SEASONS)
    covered = F["rain_1d"].notna().mean()
    print(f"  real ERA5 coverage of panel rows: {100 * covered:.2f}%")
    assert F[Fe.FEATURES_B].notna().all().all(), "NaN feature in the panel with full ERA5"
    if W is not None:  # rain features on the cached seasons themselves, even if outside the panel
        mini = W[["district", "date"]].assign(season=W["date"].dt.year)
        mini = mini[mini["date"].dt.strftime("%m-%d").between(C.ISSUE_START, C.ISSUE_END)]
        # 2026 ends 31 August, so its last 3 issue days have no future rain.
        last = mini.groupby("season")["date"].transform("max")
        mini = mini[mini["date"] <= last - pd.Timedelta(days=3)]
        R = Fe.rain_features(mini, W)
        assert R["rain_30d"].notna().all() and R["future_rain_3d"].notna().all(), (
            "gaps in real rain windows"
        )
        assert (R["rain_7d"] >= R["rain_3d"] - 1e-9).all() and (
            R["rain_api"] >= R["rain_1d"] - 1e-9
        ).all()
        print(
            f"  real ERA5 rain features on cached seasons: {R.shape}, mean 7-day rain {R['rain_7d'].mean():.1f} mm"
        )
    return F


def main():
    from .panel import INCIDENTS_PATH

    P, raw, T, nbrs = load_panel(), pd.read_parquet(INCIDENTS_PATH), load_terrain(), neighbours()
    inc = Fe.harmonise_entry_lags(raw)
    print(f"panel {P.shape}, incidents {inc.shape}, terrain {'missing' if T is None else T.shape}")
    assert check_panel(P)
    print("ok  panel reproduces the reference numbers")
    test_feature_lists()
    print("ok  feature lists: no loss_*/target columns, set A has no future_*, NOWX has no weather")
    test_no_loss_columns_in_inputs(P, inc)
    print("ok  no loss_* columns in panel or incidents table")
    W = synthetic_weather(P)
    F = test_set_a_ignores_the_future(P, inc, W, T, nbrs)
    print("ok  set A unchanged when the future is scrambled")
    test_entry_lag_rule(F, inc, raw)
    print("ok  entry-lag rule: known-at-t counts, synthetic pre-2020 lags")
    test_fits_use_training_seasons_only(F, P)
    print("ok  percentiles and target encoding use training seasons only")
    test_target_encoding_leave_one_season_out(P)
    print("ok  leave-one-season-out target encoding for training rows")
    test_persistence_known_at_t(F, inc)
    print("ok  persistence baseline uses known-at-t incidents only")
    test_real_weather_smoke(P, inc, T, nbrs)
    print("ok  real-weather smoke test")
    print("all checks passed")


if __name__ == "__main__":
    main()
