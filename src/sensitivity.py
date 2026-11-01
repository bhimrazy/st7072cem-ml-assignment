"""Sensitivity analyses computed from the saved test predictions (notebook 05, section 5).

No main model is refitted. The inputs are the test-season scores in predictions.parquet, the frozen
feature matrix and, for the 2023 alert threshold, the saved random forest and logistic regression.
The only fits are the rule baselines and two small logistic-regression controls, as in notebook 04.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import evaluate as E
from . import features as Fe

MODELS = ["lr", "rf", "hgb", "svm_linear", "svm_rbf"]


def _month_dummies(d):
    """One-hot calendar month, April to October."""
    return pd.get_dummies(d["month"]).reindex(columns=range(4, 11), fill_value=0).to_numpy(float)


def year_only_score(F):
    """The year-only control of notebook 04: logistic regression on season and month, fitted on 2017 to 2022."""
    tr, te = F[F["split"] == "train"], F[F["split"] == "test"]

    def X(d):
        return np.column_stack([d["season"].to_numpy(float), _month_dummies(d)])

    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X(tr), tr["y"])
    return m.predict_proba(X(te))[:, 1]


def paired_ablation(te, n_boot=1000, seed=C.SEED):
    """PR-AUC of each model with NOWX, A and B, and paired intervals for A minus NOWX and B minus A.

    One block bootstrap per model with set A as the reference, so both differences use the same
    weekly resamples (and the same resamples as the main tables).
    """
    y, clim, dates = te["y"].to_numpy(), te["clim"].to_numpy(), te["date"]
    rows = []
    for m in MODELS:
        named = {fs: (te[f"score_{m}_{fs}"], te[f"prob_{m}_{fs}"]) for fs in ("A", "NOWX", "B")}
        b = E.block_bootstrap(y, named, clim, dates, n_boot, seed, keys=("pr_auc",))
        b = b.set_index("model")
        rows.append(
            {
                "model": m,
                "pr_auc_nowx": b.loc["NOWX", "estimate"],
                "pr_auc_a": b.loc["A", "estimate"],
                "pr_auc_b": b.loc["B", "estimate"],
                # NOWX minus A, negated: the percentiles of the negated draws swap ends.
                "a_minus_nowx": -b.loc["NOWX", "diff_vs_ref"],
                "a_minus_nowx_lo": -b.loc["NOWX", "diff_hi"],
                "a_minus_nowx_hi": -b.loc["NOWX", "diff_lo"],
                "b_minus_a": b.loc["B", "diff_vs_ref"],
                "b_minus_a_lo": b.loc["B", "diff_lo"],
                "b_minus_a_hi": b.loc["B", "diff_hi"],
            }
        )
    return pd.DataFrame(rows).set_index("model")


def model_gaps(te, ref="lr", n_boot=1000, seed=C.SEED):
    """Paired PR-AUC difference of every set A model against `ref` on the same weekly resamples."""
    y, clim, dates = te["y"].to_numpy(), te["clim"].to_numpy(), te["date"]
    order = [ref] + [m for m in MODELS if m != ref]
    named = {m: (te[f"score_{m}_A"], te[f"prob_{m}_A"]) for m in order}
    b = E.block_bootstrap(y, named, clim, dates, n_boot, seed, keys=("pr_auc",))
    return b.drop(columns="metric").set_index("model")


def rule_flags(F):
    """0/1 alerts of the three rule baselines on the test rows, as defined in notebook 04."""
    te = F[F["split"] == "test"]
    cut = E.fit_rain_threshold(F, C.TRAIN_SEASONS)
    return {
        "persistence": E.persistence_score(te).astype(bool),
        "dahal_threshold": E.dahal_threshold_score(te).astype(bool),
        "refit_threshold": E.rain_threshold_score(te, cut)[1].astype(bool),
    }


def alert_budget(te, year_only):
    """POD, FAR and CSI at the 5% budget for climatology, the year-only control and every model.

    Ties at the cut (climatology has many) are broken at random with the fixed seed.
    """
    y = te["y"].to_numpy()
    scorers = {"climatology": te["clim"].to_numpy(), "year_only_control": year_only}
    for fs in ("A", "NOWX", "B"):
        for m in MODELS:
            scorers[m if fs == "A" else f"{m}_{fs}"] = te[f"score_{m}_{fs}"].to_numpy()
    rows = {k: E.alert_metrics(y, s) for k, s in scorers.items()}
    out = pd.DataFrame(rows).T
    out["n"], out["positives"] = len(y), int(y.sum())
    out.index.name = "model"
    return out


def rules_at_own_rate(te, flags, scorers=("climatology", "lr", "rf")):
    """Each rule at its own alert rate against climatology and models re-thresholded to that rate."""
    y = te["y"].to_numpy()
    s = {"climatology": te["clim"].to_numpy()}
    s.update({m: te[f"score_{m}_A"].to_numpy() for m in MODELS})
    rows = []
    for rule, flag in flags.items():
        rate = flag.mean()
        rows.append({"rule": rule, "scorer": "rule", **E.alert_metrics(y, flag, alert=flag)})
        for k in scorers:
            rows.append({"rule": rule, "scorer": k, **E.alert_metrics(y, s[k], budget=rate)})
    return pd.DataFrame(rows)


def threshold_from_2023(F, te, store):
    """Alert on test when the raw score reaches the 95th percentile of the same model's 2023 scores.

    Only the random forest and logistic regression are saved, so only they are scored on 2023.
    """
    va = F[F["split"] == "validate"]
    Xva = va[Fe.FEATURES_A].to_numpy(float)
    y = te["y"].to_numpy()
    rows = []
    for name, key in (("rf", "model_A"), ("lr", "lr_A")):
        thr = np.quantile(store[key].raw(Xva), 1 - C.ALERT_BUDGET)
        s = te[f"score_{name}_A"].to_numpy()
        alert = s >= thr
        row = {"model": name, "threshold": thr, **E.alert_metrics(y, s, alert=alert)}
        for season in (2024, 2025):
            row[f"alert_rate_{season}"] = alert[te["season"].to_numpy() == season].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def history_year_control(F, n_boot=1000, seed=C.SEED):
    """Climatology-like control with and without year: logistic regression on the shrunk district by
    month damaging-day rate (te_dam, leave-one-season-out on training rows) and month dummies.
    If year carried a recording-regime signal, adding it would change test PR-AUC.
    """
    tr, te = F[F["split"] == "train"], F[F["split"] == "test"]

    def X(d, year):
        cols = [np.log(np.clip(d["te_dam"].to_numpy(float), 1e-4, None)), _month_dummies(d)]
        if year:
            cols.append(d["season"].to_numpy(float))
        return np.column_stack(cols)

    named, coef = {}, {}
    for lab, year in (("history", False), ("history_plus_year", True)):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
            X(tr, year), tr["y"]
        )
        p = m.predict_proba(X(te, year))[:, 1]
        named[lab] = (p, p)
        coef[lab] = m[-1].coef_[0][-1] if year else np.nan
    y = te["y"].to_numpy()
    clim = E.climatology_score(te, E.fit_climatology(F, C.TRAIN_SEASONS))
    b = E.block_bootstrap(y, named, clim, te["date"], n_boot, seed, keys=("pr_auc",))
    b = b.drop(columns="metric").set_index("model")
    b["year_coef_standardised"] = pd.Series(coef)
    assert np.isclose(b.loc["history", "estimate"], average_precision_score(y, named["history"][0]))
    return b
