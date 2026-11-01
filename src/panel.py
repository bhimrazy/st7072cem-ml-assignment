"""District-day panel and its targets.

One row per district d and issue day t. The primary target y = 1 if district d has at least one
damaging Flood, Landslide, Heavy Rainfall or Thunderbolt incident dated t+1, t+2 or t+3.

    uv run python -m src.panel        # builds data/processed/*.parquet and prints the checks
"""

import pandas as pd

from . import config as C
from .geography import districts, incident_districts

PANEL_PATH = C.PROCESSED / "district-day-panel.parquet"
INCIDENTS_PATH = C.PROCESSED / "hydromet-incidents.parquet"
TARGETS = ["y", "y_fatal", "y_rain"] + [f"y_{h}" for h in C.HAZARDS.values()]


def load_incidents() -> pd.DataFrame:
    """Snapshot incidents of the four hazards, with district, local date, entry time and damage flags.

    Loss fields are used here only to define `damaging` and `fatal`, then dropped.
    """
    df = pd.read_csv(C.SNAPSHOT, low_memory=False)
    df = df.merge(incident_districts(), on="id", how="left")
    assert df["district"].notna().all(), "every snapshot incident should map to a district"

    def local(col):
        """UTC timestamp column to naive Nepal local time."""
        return (
            pd.to_datetime(df[col], utc=True, errors="coerce", format="ISO8601")
            .dt.tz_convert(C.TIMEZONE)
            .dt.tz_localize(None)
        )

    df["date"] = local("incidentOn").dt.normalize()
    df["created"] = local("createdOn")
    df = df[
        df["date"].notna()
        & (df["date"] <= pd.Timestamp(C.SNAPSHOT_DATE))
        & df["hazardTitle"].isin(list(C.HAZARDS))
    ].copy()
    loss = {c: pd.to_numeric(df[c], errors="coerce").fillna(0) for c in C.LOSS_DAMAGING}
    df["fatal"] = sum(loss[c] for c in C.LOSS_FATAL) > 0
    df["damaging"] = sum(loss.values()) > 0
    df["hazard"] = df["hazardTitle"].map(C.HAZARDS)
    return df[["id", "district", "hazard", "date", "created", "damaging", "fatal"]].reset_index(
        drop=True
    )


def build_panel(inc, district_names, seasons, last_issue=None):
    """District x issue day with same-day flags and the t+1..t+3 targets for the given seasons."""
    days = pd.DatetimeIndex([d for y in seasons for d in pd.date_range(f"{y}-04-01", f"{y}-10-31")])
    P = pd.MultiIndex.from_product(
        [sorted(district_names), days], names=["district", "date"]
    ).to_frame(index=False)

    # Same-day flags (outcomes on day t, used only to build the targets and for EDA).
    g = inc.groupby(["district", "date"])
    day = pd.DataFrame(
        {
            "day_any": g.size() > 0,
            "day_damaging": g["damaging"].any(),
            "day_fatal": g["fatal"].any(),
        }
    )
    for h in C.HAZARDS.values():
        hit = inc[inc["damaging"] & (inc["hazard"] == h)].groupby(["district", "date"]).size()
        day[f"day_{h}"] = day.index.isin(hit.index)
    # Rain-driven hazards only (no lightning).
    day["day_rain"] = day[[f"day_{h}" for h in C.RAIN_HAZARDS]].any(axis=1)
    P = P.merge(day.reset_index(), on=["district", "date"], how="left")
    flags = list(day.columns)
    P[flags] = P[flags].fillna(False).astype(bool)

    # Targets: any flag in the next HORIZON days, within the same district and season.
    P["season"] = P["date"].dt.year
    P = P.sort_values(["district", "date"]).reset_index(drop=True)
    grp = P.groupby(["district", "season"])
    sources = {
        "y": "day_damaging",
        "y_fatal": "day_fatal",
        "y_rain": "day_rain",
        **{f"y_{h}": f"day_{h}" for h in C.HAZARDS.values()},
    }
    for target, src in sources.items():
        ahead = pd.concat([grp[src].shift(-k) for k in range(1, C.HORIZON + 1)], axis=1)
        P[target] = ahead.fillna(False).astype(bool).any(axis=1).astype("int8")

    keep = P["date"].dt.strftime("%m-%d").between(C.ISSUE_START, C.ISSUE_END)
    if last_issue:
        keep &= P["date"] <= pd.Timestamp(last_issue)
    P = P[keep].reset_index(drop=True)
    P["month"] = P["date"].dt.month.astype("int8")
    P["split"] = P["season"].map({y: s for s, ys in C.SPLITS.items() for y in ys})
    return P


def build():
    """Build and save the incidents table and the full panel (2017 to 2025 plus the 2026 check)."""
    inc = load_incidents()
    names = districts()["district"].tolist()
    P = pd.concat(
        [
            build_panel(inc, names, C.SEASONS),
            build_panel(inc, names, [C.CHECK_SEASON], last_issue=C.CHECK_LAST_ISSUE),
        ],
        ignore_index=True,
    )
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    inc.to_parquet(INCIDENTS_PATH, index=False)
    P.to_parquet(PANEL_PATH, index=False)
    return inc, P


def load_panel() -> pd.DataFrame:
    """The frozen district-day panel written by `build()`."""
    return pd.read_parquet(PANEL_PATH)


def split_table(P):
    """Rows, positives and positive rate per split for the primary target."""
    t = P.groupby("split", sort=False)["y"].agg(rows="size", positives="sum", rate="mean")
    t["rate"] = (100 * t["rate"]).round(2)
    return t.reindex(["train", "validate", "test", "check"])


def check(P):
    """Assert the panel reproduces the expected row counts and positive rates.

    These fixed numbers fail by design if the hazard set, the damage filter or the window changes.
    """
    main = P[P["season"].isin(C.SEASONS)]

    def rate(s):
        return round(100 * s.mean(), 2)

    assert len(main) == 146_223, len(main)
    assert rate(main["y"]) == 6.81, rate(main["y"])
    assert rate(main["y_fatal"]) == 2.60, rate(main["y_fatal"])
    assert rate(main["y_rain"]) == 4.35, rate(main["y_rain"])
    st = split_table(P)
    assert st.loc[["train", "validate", "test"], "rate"].tolist() == [6.65, 4.44, 8.46], st
    assert st.loc["check", "rows"] == 10_626 and st.loc["check", "rate"] == 5.78, st
    return True


if __name__ == "__main__":
    inc, P = build()
    print(
        f"incidents (four hazards, to {C.SNAPSHOT_DATE}): {len(inc):,}; damaging {int(inc['damaging'].sum()):,}"
    )
    print(f"panel: {P.shape}; seasons {P['season'].min()} to {P['season'].max()}")
    print(split_table(P).to_string())
    print(
        "target rates (%), 2017 to 2025:",
        (100 * P[P["season"].isin(C.SEASONS)][TARGETS].mean()).round(2).to_dict(),
    )
    print("reproduction check:", "PASS" if check(P) else "FAIL")
