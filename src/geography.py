"""Frozen BIPAD geography: incident -> ward -> municipality -> district, centroids, neighbours.

The four BIPAD requests are made once by `freeze()` and saved in data/raw/ with a .meta.json.
After that everything reads the frozen files; the live API is never called again.

    uv run python -m src.geography     # freezes (only if the files are missing) and prints a summary
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from .config import BIPAD_API, RAW

REQUESTS = {
    "bipad-incident-wards": "incident/?format=json&limit=100000&fields=id,wards",
    "bipad-wards": "ward/?format=json&limit=10000",
    "bipad-municipalities": "municipality/?format=json&limit=10000",
    "bipad-districts": "district/?format=json&limit=100",
}
META = RAW / "bipad-geography.meta.json"


def freeze():
    """Pull the four BIPAD tables once and save them with the pull date. Skips files already saved."""
    RAW.mkdir(parents=True, exist_ok=True)
    meta = (
        json.loads(META.read_text())
        if META.exists()
        else {"source": "BIPAD portal, NDRRMA, Government of Nepal", "files": {}}
    )
    for name, query in REQUESTS.items():
        path = RAW / f"{name}.json"
        if path.exists():
            continue
        url = BIPAD_API + query
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
        meta["files"][path.name] = {
            "url": url,
            "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "records": len(r.json()["results"]),
            "bytes": len(r.content),
        }
        print(f"pulled {name}: {meta['files'][path.name]['records']:,} records")
    meta["note"] = (
        "Frozen on first build; never refreshed so the district link cannot drift with the live API."
    )
    META.write_text(json.dumps(meta, indent=2))


def _table(name):
    return pd.DataFrame(json.loads((RAW / f"{name}.json").read_text())["results"])


def _xy(geom):
    """Point geometry (GeoJSON dict or [lon, lat]) to (lon, lat)."""
    if isinstance(geom, dict):
        geom = geom.get("coordinates")
    return (float(geom[0]), float(geom[1])) if geom else (np.nan, np.nan)


def districts():
    """District table: id, name, province, centroid lon/lat and bbox."""
    d = _table("bipad-districts")
    d[["lon", "lat"]] = pd.DataFrame(d["centroid"].map(_xy).tolist(), index=d.index)
    d = d.rename(columns={"id": "dist_id", "title": "district"})
    return (
        d[["dist_id", "district", "province", "lon", "lat", "bbox"]]
        .sort_values("district")
        .reset_index(drop=True)
    )


def wards():
    """Ward table with its district: ward id, district name, centroid lon/lat."""
    w = _table("bipad-wards")
    mun = _table("bipad-municipalities")[["id", "district"]].rename(
        columns={"id": "municipality", "district": "dist_id"}
    )
    w = w.merge(mun, on="municipality", how="left").merge(
        districts()[["dist_id", "district"]], on="dist_id", how="left"
    )
    w[["lon", "lat"]] = pd.DataFrame(w["centroid"].map(_xy).tolist(), index=w.index)
    return w.rename(columns={"id": "ward"})[["ward", "municipality", "district", "lon", "lat"]]


def incident_districts():
    """Incident id -> district name, through BIPAD's own ward link (first ward if several)."""
    links = _table("bipad-incident-wards")
    links["ward"] = links["wards"].map(lambda w: w[0] if len(w) else None)
    out = links.merge(wards()[["ward", "district"]], on="ward", how="left")
    return out[["id", "district"]]


def neighbours(k=6):
    """District adjacency from ward centroids: A and B are neighbours if a ward of A has a ward of B
    among its k nearest wards. BIPAD has no polygons, so this stands in for shared borders."""
    from sklearn.neighbors import BallTree

    w = wards().dropna(subset=["lon", "lat", "district"]).reset_index(drop=True)
    xy = np.radians(w[["lat", "lon"]].to_numpy())
    _, idx = BallTree(xy, metric="haversine").query(xy, k=k + 1)
    names = w["district"].to_numpy()
    pairs = {
        (a, b)
        for i, row in enumerate(idx)
        for a, b in [(names[i], names[j]) for j in row[1:]]
        if a != b
    }
    pairs |= {(b, a) for a, b in pairs}
    out = {d: [] for d in sorted(set(names))}
    for a, b in sorted(pairs):
        out[a].append(b)
    return out


if __name__ == "__main__":
    freeze()
    d, w, link = districts(), wards(), incident_districts()
    nb = neighbours()
    print(
        f"districts {len(d)}, wards {len(w):,} ({w['district'].isna().sum()} without district), "
        f"incident links {len(link):,} ({link['district'].isna().sum()} without district)"
    )
    print(
        f"neighbours per district: median {np.median([len(v) for v in nb.values()]):.0f}, "
        f"min {min(len(v) for v in nb.values())}, max {max(len(v) for v in nb.values())}"
    )
