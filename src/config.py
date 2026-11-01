"""Paths, constants and split definitions shared by every module and notebook."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
WEATHER = DATA / "weather"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
FIGURES = ROOT / "paper" / "figures"
RESULTS = ROOT / "results"

# Frozen BIPAD incident snapshot. Only src/panel.py reads it; data/processed/ holds what it builds.
SNAPSHOT_DATE = "2026-09-19"
SNAPSHOT = RAW / f"bipad-incidents-{SNAPSHOT_DATE}.csv"

SEED = 42  # every random element (bootstrap, clustering restarts, synthetic lags, models) uses this

BIPAD_API = "https://bipadportal.gov.np/api/v1/"
TIMEZONE = "Asia/Kathmandu"

# Hazards in the target (BIPAD hazardTitle) and short column names for them.
HAZARDS = {
    "Flood": "flood",
    "Landslide": "landslide",
    "Heavy Rainfall": "heavy_rain",
    "Thunderbolt": "lightning",
}
RAIN_HAZARDS = ["flood", "landslide", "heavy_rain"]  # the rain-driven target excludes lightning
LOSS_FATAL = ["loss_peopleDeathCount", "loss_peopleMissingCount"]
LOSS_DAMAGING = LOSS_FATAL + ["loss_peopleInjuredCount", "loss_infrastructureDestroyedHouseCount"]

# Panel span: issue days 1 April to 28 October so the t+1..t+3 window stays in the season.
SEASONS = list(range(2017, 2026))
ISSUE_START, ISSUE_END = "04-01", "10-28"
CHECK_SEASON, CHECK_LAST_ISSUE = 2026, "2026-08-16"
HORIZON = 3  # days ahead in the target window

# Time-based splits (never random).
SPLITS = {
    "train": list(range(2017, 2023)),
    "validate": [2023],
    "test": [2024, 2025],
    "check": [2026],
}
TRAIN_SEASONS = SPLITS["train"]

# Recording regime: hydromet incidents are entered promptly from 2020.
# Before 2020 they were entered in bulk years later, so a synthetic entry lag is drawn for each of them
# from the 2020 to 2025 empirical lag distribution (features.harmonise_entry_lags), which makes the
# recent-incident features mean the same thing in every season.
PROMPT_ENTRY_FROM = 2020
LAG_REFERENCE_SEASONS = list(range(2020, 2026))

# Alert budget for POD, FAR and CSI: the top 5% of the scored district-days, pooled over the season
# (about 4 district alerts a day on average, not a fixed number each day).
ALERT_BUDGET = 0.05
