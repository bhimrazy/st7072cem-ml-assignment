"""Open-Meteo pulls: ERA5 daily weather at the 77 district centroids, and ward-centroid terrain.

ERA5: one request per district per season (2 March to 31 October, 2016 to 2026; 2026 ends 31 August),
sent one at a time, at least 2.5 s apart. Every response is cached as JSON under data/cache/weather/,
so the pull is resumable: rerun the same command and only missing requests are sent. The consolidated
archive in data/weather/ is committed, so none of this is needed to reproduce the results.

    uv run python -m src.weather --check          # how many responses are cached
    uv run python -m src.weather --limit 3        # small test pull
    uv run python -m src.weather                  # full pull (long; see README), then consolidates
    uv run python -m src.weather --consolidate    # rebuild data/weather/era5-daily-districts.csv from cache
    uv run python -m src.weather --terrain        # elevation of all ward centroids (69 requests)

Free-tier limits (open-meteo.com/en/pricing, read 27 September 2026): 600 calls per minute, 5,000 per
hour, 10,000 per day. A request longer than 14 days counts as days/14 calls, so one season of 244 days
is about 17.4 calls, and each point in an elevation request counts as one call (so the 6,803 ward
centroids cost about 6,800 calls). The script keeps its own hourly budget below the limit, and if
the daily limit is reached it probes once an hour until the quota frees up, then carries on.
"""

import argparse
import json
import time
from collections import deque
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config as C
from .geography import districts, wards

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
DAILY = [
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "temperature_2m_max",
    "temperature_2m_mean",
]
WEATHER_SEASONS = range(2016, 2027)
SEASON_START, SEASON_END = "03-02", "10-31"
LAST_DATE = {2026: "2026-08-31"}

CACHE = C.CACHE / "weather"
ELEV_CACHE = C.CACHE / "elevation"
LOG = C.RESULTS / "weather-pull-log.txt"
ERA5_CSV = C.WEATHER / "era5-daily-districts.csv"
WARD_ELEV_CSV = C.RAW / "open-meteo-ward-elevation.csv"
TERRAIN_CSV = C.PROCESSED / "district-terrain.csv"

SLEEP = 2.5  # seconds between requests
MINUTE_BUDGET = 500  # weighted calls per rolling minute (limit 600)
HOURLY_BUDGET = 4_500  # weighted calls per rolling hour (limit 5,000)
MAX_TRIES = 6


def log(msg):
    """Print and append a timestamped line to the pull log."""
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def slug(name):
    """District name in lower case with hyphens, for cache file names."""
    return name.lower().replace(" ", "-")


def jobs():
    """Every (district, season) request, ordered by season so whole seasons finish together.
    2016 goes last: it is outside the panel and only extends the rain climatology."""
    d = districts()
    out = []
    for season in sorted(WEATHER_SEASONS, key=lambda s: (s == 2016, s)):
        for row in d.itertuples():
            out.append(
                {
                    "district": row.district,
                    "season": season,
                    "lat": round(row.lat, 4),
                    "lon": round(row.lon, 4),
                    "start": f"{season}-{SEASON_START}",
                    "end": LAST_DATE.get(season, f"{season}-{SEASON_END}"),
                    "path": CACHE / f"era5-{slug(row.district)}-{season}.json",
                }
            )
    return out


def weight(job):
    """Open-Meteo call weight of one request (days beyond 14 and variables beyond 10 count extra)."""
    days = (pd.Timestamp(job["end"]) - pd.Timestamp(job["start"])).days + 1
    return max(1.0, days / 14) * max(1.0, len(DAILY) / 10)


class Throttle:
    """Keeps requests SLEEP seconds apart and the weighted calls per minute and per hour under budget."""

    def __init__(self):
        self.sent = deque()  # (time, weight)
        self.last = 0.0

    def used(self, seconds):
        now = time.time()
        return sum(w for t, w in self.sent if now - t < seconds)

    def wait(self, w):
        gap = SLEEP - (time.time() - self.last)
        if gap > 0:
            time.sleep(gap)
        while self.sent and time.time() - self.sent[0][0] > 3600:
            self.sent.popleft()
        for window, budget in ((60, MINUTE_BUDGET), (3600, HOURLY_BUDGET)):
            if self.sent and self.used(window) + w > budget:
                oldest = next(t for t, _ in self.sent if time.time() - t < window)
                pause = window - (time.time() - oldest) + 2
                if window == 3600:
                    log(
                        f"hourly budget reached ({self.used(window):.0f} weighted calls); pausing {pause / 60:.1f} min"
                    )
                time.sleep(max(pause, 0))
                return self.wait(w)
        self.last = time.time()
        self.sent.append((self.last, w))


def get(url, params, throttle, w):
    """GET with backoff. Quota 429s wait and retry without limit; timeouts and 5xx get MAX_TRIES tries."""
    attempt = 0
    while attempt < MAX_TRIES:
        throttle.wait(w)
        try:
            r = requests.get(url, params=params, timeout=90)
            throttle.last = time.time()  # the gap runs from the end of each response
        except (requests.Timeout, requests.ConnectionError) as e:
            pause = 5 * 2**attempt
            attempt += 1
            log(f"  network error ({type(e).__name__}); retry in {pause}s")
            time.sleep(pause)
            continue
        if r.status_code == 200:
            return r.json()
        reason = r.text[:200].strip()
        if r.status_code == 429:
            low = reason.lower()
            # Daily quota: probe once an hour, which works whether it resets at midnight or rolls.
            if "daily" in low:
                pause = 3600
            elif "hourly" in low:
                pause = 3600 - datetime.now().minute * 60 + 60
            elif "minutely" in low:
                pause = 65
            else:
                pause = min(10 * 2**attempt, 900)
                attempt += 1
            log(f"  HTTP 429 ({reason}); waiting {pause / 60:.1f} min")
            time.sleep(pause)
            continue
        if r.status_code >= 500:
            pause = 5 * 2**attempt
            attempt += 1
            log(f"  HTTP {r.status_code}; retry in {pause}s")
            time.sleep(pause)
            continue
        log(f"  HTTP {r.status_code}: {reason}")  # 4xx other than 429: a bad request, do not retry
        return None
    return None


def pull(limit=None):
    """Send every uncached ERA5 request (or the first `limit` of them), caching each response."""
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if not j["path"].exists()]
    if limit:
        todo = todo[:limit]
    total_w = sum(weight(j) for j in todo)
    log(f"ERA5 pull: {len(todo)} requests to send (about {total_w:,.0f} weighted calls)")
    throttle, done, failed = Throttle(), 0, 0
    for i, job in enumerate(todo, 1):
        params = {
            "latitude": job["lat"],
            "longitude": job["lon"],
            "start_date": job["start"],
            "end_date": job["end"],
            "daily": ",".join(DAILY),
            "models": "era5",
            "timezone": C.TIMEZONE,
        }
        data = get(ARCHIVE_URL, params, throttle, weight(job))
        if data is None or "daily" not in data:
            failed += 1
            log(f"[{i}/{len(todo)}] FAILED {job['district']} {job['season']}")
            continue
        data["request"] = {
            "url": ARCHIVE_URL,
            "params": params,
            "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        job["path"].write_text(json.dumps(data))
        done += 1
        log(
            f"[{i}/{len(todo)}] {job['district']} {job['season']}: {len(data['daily']['time'])} days"
        )
    log(f"ERA5 pull finished: {done} saved, {failed} failed")
    return failed


def check():
    """Report how many ERA5 responses are cached, per season."""
    js = jobs()
    cached = pd.DataFrame(
        {"season": [j["season"] for j in js], "cached": [j["path"].exists() for j in js]}
    )
    per = cached.groupby("season")["cached"].sum()
    left = [j for j in js if not j["path"].exists()]
    print(
        f"cached {int(cached['cached'].sum())} of {len(js)} requests; "
        f"remaining about {sum(weight(j) for j in left):,.0f} weighted calls"
    )
    print("per season:", per.to_dict())
    return int(cached["cached"].sum()), len(js)


def consolidate():
    """Combine every cached response into one long CSV: district x date x daily variables."""
    frames = []
    for j in jobs():
        if not j["path"].exists():
            continue
        data = json.loads(j["path"].read_text())
        f = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
        f.insert(0, "district", j["district"])
        f["grid_lat"], f["grid_lon"], f["grid_elevation"] = (
            data["latitude"],
            data["longitude"],
            data.get("elevation"),
        )
        frames.append(f)
    if not frames:
        print("nothing cached yet")
        return None
    W = pd.concat(frames, ignore_index=True)
    empty = [c for c in DAILY if c in W and W[c].isna().all()]
    W = W.drop(columns=empty)
    C.WEATHER.mkdir(parents=True, exist_ok=True)
    W.to_csv(ERA5_CSV, index=False)
    pulled = sorted(
        datetime.fromtimestamp(j["path"].stat().st_mtime).date().isoformat()
        for j in jobs()
        if j["path"].exists()
    )
    meta = {
        "source": "Open-Meteo historical weather API, ERA5 (CC BY 4.0, attribution to Open-Meteo)",
        "url": ARCHIVE_URL,
        "models": "era5",
        "timezone": C.TIMEZONE,
        "daily": [c for c in DAILY if c not in empty],
        "dropped_all_null": empty,
        "points": "BIPAD district centroids (data/raw/bipad-districts.json)",
        "span": "2 March to 31 October per season, 2016 to 2026 (2026 to 31 August)",
        "requests_cached": len(frames),
        "requests_total": len(jobs()),
        "rows": len(W),
        "pulled_from": pulled[0],
        "pulled_to": pulled[-1],
    }
    ERA5_CSV.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"wrote {ERA5_CSV.name}: {W.shape}, {len(frames)} of {len(jobs())} requests; dropped all-null {empty}"
    )
    return W


def load_weather():
    """Consolidated ERA5 daily table, or None if it has not been built yet."""
    if not ERA5_CSV.exists():
        return None
    return pd.read_csv(ERA5_CSV, parse_dates=["date"])


def terrain():
    """Elevation at every ward centroid (batches of 100), then district terrain statistics."""
    ELEV_CACHE.mkdir(parents=True, exist_ok=True)
    w = wards().sort_values("ward").reset_index(drop=True)
    throttle, elev = Throttle(), []
    batches = [w.iloc[i : i + 100] for i in range(0, len(w), 100)]
    log(f"terrain: {len(w):,} ward centroids in {len(batches)} requests")
    for k, b in enumerate(batches):
        path = ELEV_CACHE / f"wards-{k:03d}.json"
        if not path.exists():
            params = {
                "latitude": ",".join(f"{v:.5f}" for v in b["lat"]),
                "longitude": ",".join(f"{v:.5f}" for v in b["lon"]),
            }
            data = get(ELEVATION_URL, params, throttle, len(b))  # each point counts as one call
            if data is None:
                raise RuntimeError(f"elevation batch {k} failed; rerun to resume")
            path.write_text(json.dumps(data))
        elev += json.loads(path.read_text())["elevation"]
    w["elevation"] = elev
    w[["ward", "municipality", "district", "lon", "lat", "elevation"]].to_csv(
        WARD_ELEV_CSV, index=False
    )
    WARD_ELEV_CSV.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "source": "Open-Meteo elevation API (Copernicus DEM GLO-90), CC BY 4.0",
                "url": ELEVATION_URL,
                "points": "BIPAD ward centroids",
                "requests": len(batches),
                "pulled_on": datetime.now().date().isoformat(),
            },
            indent=2,
        )
    )

    t = w.groupby("district")["elevation"].agg(
        elev_mean="mean",
        elev_relief=lambda e: e.quantile(0.95) - e.quantile(0.05),
        share_above_1000m=lambda e: (e > 1000).mean(),
        share_above_3000m=lambda e: (e > 3000).mean(),
    )
    t = t.round(3).reset_index()
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    t.to_csv(TERRAIN_CSV, index=False)
    log(
        f"terrain: wrote {TERRAIN_CSV.name} {t.shape}; mean elevation range {t['elev_mean'].min():.0f} to {t['elev_mean'].max():.0f} m"
    )
    return t


def load_terrain():
    """District terrain statistics, or None if they have not been built yet."""
    return pd.read_csv(TERRAIN_CSV) if TERRAIN_CSV.exists() else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--limit", type=int, help="send at most N uncached requests (for testing)")
    ap.add_argument("--check", action="store_true", help="report cache progress and exit")
    ap.add_argument(
        "--consolidate", action="store_true", help="only rebuild the CSV from the cache"
    )
    ap.add_argument(
        "--terrain",
        action="store_true",
        help="pull ward elevations and write district terrain stats",
    )
    a = ap.parse_args()
    if a.check:
        check()
    elif a.terrain:
        terrain()
    elif a.consolidate:
        consolidate()
    else:
        pull(a.limit)
        consolidate()
