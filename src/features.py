"""Feature sets for the district-day panel.

Set A holds only what is known at the end of issue day t. Its rain comes from ERA5 analysis of day t,
so it is an analysis-based hindcast (what a perfect nowcast would know), not an operational forecast.
Set B is A plus observed ERA5 rain for t+1 to t+3, named `future_*`: a "perfect forecast" upper bound.
Set NOWX (no weather) drops every rain and temperature column: the floor that shows what rainfall adds.

Every fitted quantity (district rain percentiles, target encoding) is a fit_* function that takes
the training seasons explicitly, so it can be refitted inside each cross-validation fold. Training
rows get a leave-one-season-out target encoding so a row never sees its own season's outcomes.
"""

import numpy as np
import pandas as pd

from . import config as C

HAZ = list(C.HAZARDS.values())

RAIN_A = [
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "rain_14d",
    "rain_30d",
    "rain_api",
    "rain_hours_1d",
    "tmax_1d",
    "rain_3d_pct",
]
FUTURE = ["future_rain_3d", "future_rain_max_1d", "future_rain_hours_3d"]
SEASON = ["doy_sin", "doy_cos", "month"]
TERRAIN = ["elev_mean", "elev_relief", "share_above_1000m", "share_above_3000m"]
RECENT = ["recent_dam_3d", "recent_dam_7d", "nbr_recent_dam_3d", "nbr_recent_dam_7d"] + [
    f"recent_{h}_7d" for h in HAZ
]
HISTORY = ["te_dam"] + [f"te_{h}" for h in HAZ]

FEATURES_A = RAIN_A + SEASON + TERRAIN + RECENT + HISTORY
FEATURES_B = FEATURES_A + FUTURE
FEATURES_NOWX = SEASON + TERRAIN + RECENT + HISTORY  # no-weather ablation
FEATURE_SETS = {"A": FEATURES_A, "B": FEATURES_B, "NOWX": FEATURES_NOWX}


# ---------- weather ----------


def antecedent_index(rain, k=0.9):
    """Antecedent precipitation index: API_t = k * API_{t-1} + r_t, starting from 0 on 2 March."""
    api, out = 0.0, []
    for r in rain.fillna(0).to_numpy():
        api = k * api + r
        out.append(api)
    return pd.Series(out, index=rain.index).where(rain.notna())


def rain_features(P, W):
    """Antecedent rain (known at t) and future rain (t+1..t+3) per district-day. NaN where weather is missing."""
    cols = [c for c in RAIN_A if c != "rain_3d_pct"] + FUTURE
    if W is None or W.empty:
        return P.assign(**{c: np.nan for c in cols})
    W = W.copy()
    W["date"] = pd.to_datetime(W["date"])
    W["season"] = W["date"].dt.year
    W = W.sort_values(["district", "date"])
    g = W.groupby(["district", "season"])
    rain = W["precipitation_sum"]
    out = W[["district", "date"]].copy()
    out["rain_1d"] = rain
    for k in (3, 7, 14, 30):  # sums over t-k+1 .. t, NaN until the window is full
        out[f"rain_{k}d"] = g["precipitation_sum"].transform(
            lambda s: s.rolling(k, min_periods=k).sum()
        )
    out["rain_api"] = g["precipitation_sum"].transform(antecedent_index)
    out["rain_hours_1d"] = W["precipitation_hours"]
    out["tmax_1d"] = W["temperature_2m_max"]

    def ahead(col):
        """Values of `col` on days t+1, t+2 and t+3, one column each."""
        return pd.concat([g[col].shift(-k) for k in (1, 2, 3)], axis=1)

    out["future_rain_3d"] = ahead("precipitation_sum").sum(axis=1, min_count=3)
    out["future_rain_max_1d"] = ahead("precipitation_sum").max(axis=1, skipna=False)
    out["future_rain_hours_3d"] = ahead("precipitation_hours").sum(axis=1, min_count=3)
    return P.merge(out, on=["district", "date"], how="left")


def fit_rain_percentiles(F, train_seasons):
    """Per district, the sorted 3-day rain sums of the training seasons' issue days."""
    tr = F[F["season"].isin(train_seasons)].dropna(subset=["rain_3d"])
    return {d: np.sort(s.to_numpy()) for d, s in tr.groupby("district")["rain_3d"]}


def apply_rain_percentiles(F, fitted):
    """Percentile (0 to 1) of each day's 3-day rain within its district's training distribution."""
    pct = pd.Series(np.nan, index=F.index)
    for d, idx in F.groupby("district").groups.items():
        ref = fitted.get(d)
        if ref is None or len(ref) == 0:
            continue
        x = F.loc[idx, "rain_3d"].to_numpy()
        pct.loc[idx] = np.where(
            np.isnan(x), np.nan, np.searchsorted(ref, x, side="right") / len(ref)
        )
    return F.assign(rain_3d_pct=pct)


# ---------- calendar and terrain ----------


def season_features(F):
    """Day of year as a sine and cosine pair, so 31 December sits next to 1 January."""
    doy = F["date"].dt.dayofyear
    return F.assign(
        doy_sin=np.sin(2 * np.pi * doy / 365.25), doy_cos=np.cos(2 * np.pi * doy / 365.25)
    )


def terrain_features(F, T):
    """District terrain statistics (NaN if the terrain table is missing)."""
    if T is None:
        return F.assign(**{c: np.nan for c in TERRAIN})
    return F.merge(T[["district"] + TERRAIN], on="district", how="left")


# ---------- recent incidents (entry-lag rule) ----------


def entry_lag_days(inc):
    """Entry lag in whole days: calendar day of `createdOn` minus the incident day, floored at 0."""
    return (inc["created"].dt.normalize() - inc["date"]).dt.days.clip(lower=0)


def harmonise_entry_lags(inc, seed=C.SEED):
    """Give pre-2020 incidents a synthetic entry date so `created` means the same thing in every season.

    From 2020 incidents are entered within about 1.5 days, so their real `createdOn` is kept. Before
    2020 they were entered in bulk years later, so `createdOn` says nothing about when the incident
    would have been known. Each pre-2020 incident is given a lag drawn (with a fixed seed) from the
    empirical lag distribution of the 2020 to 2025 incidents of the same hazard, and `created` is set
    to incident day plus that lag. Adds a boolean `lag_synthetic` column.
    """
    inc = inc.copy()
    rng = np.random.default_rng(seed)
    lag = entry_lag_days(inc)
    year = inc["date"].dt.year
    ref = inc[year.isin(C.LAG_REFERENCE_SEASONS)]
    pre = year < C.PROMPT_ENTRY_FROM
    inc["lag_synthetic"] = pre
    for hazard, idx in inc[pre].groupby("hazard").groups.items():
        pool = lag[ref.index[ref["hazard"] == hazard]].to_numpy()
        if len(pool) == 0:
            pool = lag[ref.index].to_numpy()
        drawn = rng.choice(pool, size=len(idx), replace=True)
        inc.loc[idx, "created"] = inc.loc[idx, "date"] + pd.to_timedelta(drawn, unit="D")
    return inc


def available_day(inc):
    """First issue day on which an incident could be known: the calendar day it was entered.

    Expects `harmonise_entry_lags` to have run, so that pre-2020 entry dates are synthetic rather
    than the bulk-entry dates in the snapshot. Entry before the incident day (one record) counts
    from the incident day.
    """
    return inc["created"].dt.normalize().clip(lower=inc["date"])


def _window_counts(inc, k, name):
    """Count incidents dated t-k .. t that are already available at t, per district and issue day t."""
    if inc.empty:
        return pd.DataFrame(columns=["district", "date", name])
    a = available_day(inc)
    rows = []
    for lag in range(k + 1):
        t = inc["date"] + pd.Timedelta(days=lag)
        ok = t >= a
        rows.append(pd.DataFrame({"district": inc.loc[ok, "district"], "date": t[ok]}))
    r = pd.concat(rows)
    return r.groupby(["district", "date"]).size().rename(name).reset_index()


def recent_incident_features(F, inc, nbrs):
    """Damaging incidents in the last 3 and 7 days in the district, per hazard, and in neighbouring districts."""
    dam = inc[inc["damaging"]]
    parts = {
        "recent_dam_3d": (dam, 3),
        "recent_dam_7d": (dam, 7),
        **{f"recent_{h}_7d": (dam[dam["hazard"] == h], 7) for h in HAZ},
    }
    for name, (sub, k) in parts.items():
        F = F.merge(_window_counts(sub, k, name), on=["district", "date"], how="left")
        F[name] = F[name].fillna(0).astype("int16")
    pairs = pd.DataFrame(
        [(d, n) for d, ns in nbrs.items() for n in ns], columns=["district", "nbr"]
    )
    for k in (3, 7):
        own = F[["district", "date", f"recent_dam_{k}d"]].rename(columns={"district": "nbr"})
        nb = (
            pairs.merge(own, on="nbr")
            .groupby(["district", "date"])[f"recent_dam_{k}d"]
            .sum()
            .rename(f"nbr_recent_dam_{k}d")
        )
        F = F.merge(nb.reset_index(), on=["district", "date"], how="left")
        F[f"nbr_recent_dam_{k}d"] = F[f"nbr_recent_dam_{k}d"].fillna(0).astype("int16")
    return F


# ---------- district history (target encoding) ----------


def fit_target_encoding(P, train_seasons, prior_weight=20):
    """District x month rate of damaging days (all hazards and per hazard), from training seasons only.

    Each rate is shrunk towards the national rate for that month with `prior_weight` pseudo-days.
    Same-day flags are used, so the encoding describes history, not the t+1..t+3 target itself.
    """
    tr = P[P["season"].isin(train_seasons)]
    cols = {"te_dam": "day_damaging", **{f"te_{h}": f"day_{h}" for h in HAZ}}
    cell = tr.groupby(["district", "month"])[list(cols.values())].agg(["sum", "count"])
    national = tr.groupby("month")[list(cols.values())].mean()
    out = pd.DataFrame(index=cell.index)
    for te, src in cols.items():
        prior = national[src].reindex(cell.index.get_level_values("month")).to_numpy()
        out[te] = (cell[(src, "sum")] + prior_weight * prior) / (
            cell[(src, "count")] + prior_weight
        )
    return {
        "cells": out.reset_index(),
        "national": national.rename(columns={v: k for k, v in cols.items()}).reset_index(),
    }


def apply_target_encoding(F, fitted):
    """Attach the fitted district x month rates to F (index preserved)."""
    out = F.drop(columns=[c for c in HISTORY if c in F]).merge(
        fitted["cells"], on=["district", "month"], how="left"
    )
    out.index = F.index
    fill = out[["month"]].merge(fitted["national"], on="month", how="left")
    fill.index = F.index
    for c in HISTORY:  # districts or months unseen in training fall back to the national rate
        out[c] = out[c].fillna(fill[c])
    return out


def apply_target_encoding_loso(F, P, train_seasons, prior_weight=20):
    """Leave-one-season-out target encoding.

    A training row of season s gets rates fitted on the other training seasons, so its own day t
    and t+1..t+3 outcomes never enter its features. Validation, test and check rows get rates fitted
    on all training seasons.
    """
    parts = [
        apply_target_encoding(
            F[~F["season"].isin(train_seasons)], fit_target_encoding(P, train_seasons, prior_weight)
        )
    ]
    for s in train_seasons:
        others = [t for t in train_seasons if t != s]
        parts.append(
            apply_target_encoding(F[F["season"] == s], fit_target_encoding(P, others, prior_weight))
        )
    return pd.concat(parts).loc[F.index]


# ---------- everything ----------


def build_features(P, inc, W, T, nbrs, train_seasons, loso=True):
    """Panel plus feature sets A, B and NOWX. Fitted parts use `train_seasons` only.

    `inc` must already have harmonised entry lags. With `loso` the training rows get the
    leave-one-season-out target encoding; without it (used only by the leakage checks) every row
    gets the encoding fitted on all training seasons.
    """
    F = rain_features(P, W)
    F = apply_rain_percentiles(F, fit_rain_percentiles(F, train_seasons))
    F = season_features(F)
    F = terrain_features(F, T)
    F = recent_incident_features(F, inc, nbrs)
    if loso:
        F = apply_target_encoding_loso(F, P, train_seasons)
    else:
        F = apply_target_encoding(F, fit_target_encoding(P, train_seasons))
    assert len(F) == len(P), "feature merges must not add or drop rows"
    return F.sort_values(["season", "district", "date"]).reset_index(drop=True)


def load_incidents_harmonised():
    """The frozen incidents table with synthetic pre-2020 entry lags (seeded)."""
    from .panel import INCIDENTS_PATH

    return harmonise_entry_lags(pd.read_parquet(INCIDENTS_PATH))


def load_all(train_seasons=C.TRAIN_SEASONS):
    """Convenience loader for notebooks: frozen inputs to the full feature table."""
    from .geography import neighbours
    from .panel import load_panel
    from .weather import load_terrain, load_weather

    return build_features(
        load_panel(),
        load_incidents_harmonised(),
        load_weather(),
        load_terrain(),
        neighbours(),
        train_seasons,
    )
