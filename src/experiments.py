"""Helpers that notebook 04 uses to run one experiment cell: fit, calibrate, score and summarise.

A `Data` bundle holds the four season blocks (train, validate, test, check) of the frozen feature
matrix with the region encodings attached. `run()` fits one model on one feature list and one
target, calibrates on 2023 and returns the scores for the test and check rows. `summarise()` turns
a dict of named (score, prob) pairs into one metric table with weekly block-bootstrap intervals.
"""

import time
from dataclasses import dataclass

import pandas as pd

from . import config as C
from . import evaluate as E
from . import models as M


@dataclass
class Data:
    F: pd.DataFrame
    region_cols: dict  # encoding name -> list of columns

    def block(self, split):
        return self.F[self.F["split"] == split]

    @property
    def train(self):
        return self.block("train")

    @property
    def validate(self):
        return self.block("validate")

    @property
    def test(self):
        return self.block("test")

    @property
    def check(self):
        return self.block("check")


def attach_regions(F, enc):
    """Add the four location encodings as columns: region one-hot, GMM posteriors, centroid distances, district one-hot."""
    F = F.merge(
        enc[
            ["region", "belt", "province"]
            + [c for c in enc if c.startswith("region_post_") or c.startswith("region_dist_")]
        ],
        left_on="district",
        right_index=True,
        how="left",
    )
    label = pd.get_dummies(F["region"], prefix="region_is").astype(float)
    district = pd.get_dummies(F["district"], prefix="dist_is").astype(float)
    F = pd.concat([F, label, district], axis=1)
    cols = {
        "none": [],
        "label": list(label.columns),
        "posterior": [c for c in F if c.startswith("region_post_")],
        "distance": [c for c in F if c.startswith("region_dist_")],
        "district": list(district.columns),
    }
    return Data(F.reset_index(drop=True), cols)


def xy(d, cols, target):
    """Feature matrix and target vector of one season block."""
    return d[cols].to_numpy(float), d[target].to_numpy()


def run(data, name, params, cols, target="y", seed=C.SEED):
    """Fit on 2017 to 2022, calibrate on 2023, score 2024 to 2025 and 2026. Returns a dict of arrays."""
    t0 = time.time()
    Xtr, ytr = xy(data.train, cols, target)
    Xva, yva = xy(data.validate, cols, target)
    m = M.Calibrated(name, params, seed).fit(Xtr, ytr, Xva, yva)
    out = {"model": m, "seconds": time.time() - t0}
    for split in ("validate", "test", "check"):
        X, _ = xy(data.block(split), cols, target)
        out[f"score_{split}"] = m.raw(X)
        out[f"prob_{split}"] = m.predict_proba(X)
    return out


def summarise(named, y, clim, dates, n_boot=1000, seed=C.SEED):
    """Point metrics for every (score, prob) pair plus bootstrap intervals for PR-AUC, BSS and CSI.

    The first entry of `named` is the reference for the paired differences (use climatology).
    """
    point = pd.DataFrame({k: E.metrics(y, s, clim, p) for k, (s, p) in named.items()}).T
    boot = E.block_bootstrap(y, named, clim, dates, n_boot=n_boot, seed=seed)
    wide = boot.pivot(
        index="model", columns="metric", values=["lo", "hi", "diff_vs_ref", "diff_lo", "diff_hi"]
    )
    wide.columns = [f"{m}_{s}" for s, m in wide.columns]
    out = point.join(wide)
    out.index.name = "model"
    cols = [
        "n",
        "positives",
        "base_rate",
        "pr_auc",
        "pr_auc_lo",
        "pr_auc_hi",
        "pr_auc_diff_vs_ref",
        "pr_auc_diff_lo",
        "pr_auc_diff_hi",
        "roc_auc",
        "brier",
        "bss",
        "bss_lo",
        "bss_hi",
        "bss_diff_vs_ref",
        "bss_diff_lo",
        "bss_diff_hi",
        "pod",
        "far",
        "csi",
        "csi_lo",
        "csi_hi",
        "csi_diff_vs_ref",
        "csi_diff_lo",
        "csi_diff_hi",
        "alert_rate",
    ]
    return out[[c for c in cols if c in out]]


def fmt(table, cols=("pr_auc", "bss", "csi")):
    """Compact view: estimate [lo, hi] per metric."""
    view = pd.DataFrame(index=table.index)
    for c in cols:
        view[c] = [
            f"{e:.3f} [{lo:.3f}, {hi:.3f}]"
            for e, lo, hi in zip(table[c], table[f"{c}_lo"], table[f"{c}_hi"])
        ]
    view["pod"] = table["pod"].round(3)
    view["far"] = table["far"].round(3)
    return view
