"""Baselines, metrics and block-bootstrap intervals for the district-day forecasts.

Scores are probabilities (or 0/1 flags for rule baselines) for the primary target. Every baseline is
fitted on the training seasons only; the persistence baseline uses incidents known at day t.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config as C

# Dahal and Hasegawa (2008): I = 73.90 D^-0.79 (mm/h over D hours); 144 mm/day is the paper's daily figure.
DAHAL_1D_MM = 144.0
DAHAL_3D_MM = round(73.90 * 72**-0.79 * 72, 1)  # 182.2 mm over 72 h


# ---------- baselines ----------


def fit_climatology(P, seasons):
    """District x month rate of the target in the given seasons (no shrinkage), with the month rate as fallback."""
    tr = P[P["season"].isin(seasons)]
    return {
        "cell": tr.groupby(["district", "month"])["y"].mean(),
        "month": tr.groupby("month")["y"].mean(),
    }


def climatology_score(F, fitted):
    """Fitted district x month rate for each row of F, or the month rate for an unseen cell."""
    key = pd.MultiIndex.from_frame(F[["district", "month"]])
    p = fitted["cell"].reindex(key).to_numpy()
    fallback = fitted["month"].reindex(F["month"]).to_numpy()
    return np.where(np.isnan(p), fallback, p)


def persistence_score(F):
    """1 if a damaging incident dated t-3..t was already entered by day t (the entry-lag rule), else 0."""
    return (F["recent_dam_3d"].to_numpy() > 0).astype(float)


def dahal_threshold_score(F):
    """1 if ERA5 rain on day t exceeds 144 mm or the 3-day sum exceeds the 72-hour threshold, else 0."""
    return ((F["rain_1d"] >= DAHAL_1D_MM) | (F["rain_3d"] >= DAHAL_3D_MM)).to_numpy().astype(float)


def fit_rain_threshold(F, seasons):
    """The 3-day rain percentile cut that maximises CSI on the given seasons (a refitted ERA5 threshold)."""
    tr = F[F["season"].isin(seasons)].dropna(subset=["rain_3d_pct"])
    best = (None, -1.0)
    for cut in np.arange(0.50, 1.0, 0.01):
        alert = tr["rain_3d_pct"].to_numpy() >= cut
        csi = _csi(tr["y"].to_numpy(), alert)
        if csi > best[1]:
            best = (round(float(cut), 2), csi)
    return best[0]


def rain_threshold_score(F, cut):
    """Continuous score (the percentile) so PR-AUC is defined; the alert fires at the fitted cut."""
    return F["rain_3d_pct"].fillna(0).to_numpy(), (
        F["rain_3d_pct"].fillna(0).to_numpy() >= cut
    ).astype(float)


# ---------- metrics ----------


def _csi(y, alert):
    hits = np.sum(alert & (y == 1))
    misses = np.sum(~alert & (y == 1))
    fa = np.sum(alert & (y == 0))
    return hits / (hits + misses + fa) if hits + misses + fa else 0.0


def alert_metrics(y, score, budget=C.ALERT_BUDGET, alert=None, seed=C.SEED):
    """POD, FAR and CSI when the top `budget` share of district-days is alerted (or a given alert flag).

    The budget is a share of the scored rows themselves (pooled over the season, not per day). Tied
    scores at the cut, common for climatology and the 0/1 rules, are ordered at random with `seed`
    so that row order does not decide which tied rows are alerted.
    """
    y = np.asarray(y)
    if alert is None:
        score = np.asarray(score, dtype=float)
        k = int(round(budget * len(score)))
        tie = np.random.default_rng(seed).random(len(score))
        order = np.lexsort((tie, -score))
        alert = np.zeros(len(score), dtype=bool)
        alert[order[:k]] = True
    hits = np.sum(alert & (y == 1))
    misses = np.sum(~alert & (y == 1))
    fa = np.sum(alert & (y == 0))
    return {
        "pod": hits / max(hits + misses, 1),
        "far": fa / max(hits + fa, 1),
        "csi": hits / max(hits + misses + fa, 1),
        "alert_rate": alert.mean(),
    }


def metrics(y, score, clim, prob=None, budget=C.ALERT_BUDGET, alert=None):
    """PR-AUC, ROC-AUC, Brier, Brier skill against climatology, and alert metrics at a fixed budget.

    Ranking metrics use `score` (the model's raw, monotone score); Brier and the skill score use
    `prob` (the calibrated probability), which defaults to `score` for baselines that are already rates.
    """
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    prob = score if prob is None else np.asarray(prob, dtype=float)
    out = {
        "n": len(y),
        "positives": int(y.sum()),
        "base_rate": y.mean(),
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score) if 0 < y.mean() < 1 else np.nan,
        "brier": brier_score_loss(y, np.clip(prob, 0, 1)),
    }
    out["bss"] = 1 - out["brier"] / brier_score_loss(y, np.clip(clim, 0, 1))
    out.update(alert_metrics(y, score, budget, alert))
    return out


def week_blocks(dates):
    """Block id per row: the ISO week, so one storm that hits many districts stays in one block."""
    d = pd.to_datetime(pd.Series(dates))
    return (
        d.dt.isocalendar().year.astype(str) + "-" + d.dt.isocalendar().week.astype(str).str.zfill(2)
    ).to_numpy()


def block_bootstrap(
    y, scores, clim, dates, n_boot=1000, seed=C.SEED, keys=("pr_auc", "bss", "csi")
):
    """Weekly block bootstrap: resample weeks with replacement, recompute metrics for every model.

    `scores` maps a model name to (score, prob). Returns point estimates, 2.5 and 97.5 percentiles
    per model and metric, and the same for the paired difference against the first entry of
    `scores` (normally climatology).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    clim = np.asarray(clim)
    scores = {
        m: (np.asarray(v[0], dtype=float), np.asarray(v[1], dtype=float)) for m, v in scores.items()
    }
    blocks = week_blocks(dates)
    uniq, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == b) for b in range(len(uniq))]
    names = list(scores)
    ref = names[0]
    draws = {m: {k: [] for k in keys} for m in names}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[b] for b in pick])
        yb, cb = y[idx], clim[idx]
        if yb.sum() == 0:
            continue
        for m in names:
            r = metrics(yb, scores[m][0][idx], cb, scores[m][1][idx])
            for k in keys:
                draws[m][k].append(r[k])
    rows = []
    ref_point = metrics(y, scores[ref][0], clim, scores[ref][1])
    for m in names:
        point = metrics(y, scores[m][0], clim, scores[m][1])
        for k in keys:
            a = np.asarray(draws[m][k])
            d = a - np.asarray(draws[ref][k])
            rows.append(
                {
                    "model": m,
                    "metric": k,
                    "estimate": point[k],
                    "lo": np.percentile(a, 2.5),
                    "hi": np.percentile(a, 97.5),
                    "diff_vs_ref": point[k] - ref_point[k],
                    "diff_lo": np.percentile(d, 2.5),
                    "diff_hi": np.percentile(d, 97.5),
                }
            )
    return pd.DataFrame(rows)


def reliability_table(y, p, bins=10):
    """Mean predicted and observed rate per probability bin (quantile bins, so every bin has rows)."""
    y = np.asarray(y)
    p = np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    t = (
        pd.DataFrame({"bin": b, "p": p, "y": y})
        .groupby("bin")
        .agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    )
    return t.reset_index(drop=True)
