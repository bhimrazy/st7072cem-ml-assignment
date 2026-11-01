"""Stage 2 classifiers: specification, leave-one-season-out tuning, final fit and calibration.

Protocol: tune on leave-one-season-out folds inside 2017 to 2022 (mean PR-AUC), fit the chosen
setting on 2017 to 2022, calibrate the raw score on 2023 with isotonic regression, score 2024 to
2025 once, then the 2026 check. The same seed is used everywhere.
"""

import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from . import config as C

RBF_SUBSAMPLE = 12_000  # training rows for the RBF kernel (stratified), kept small for run time

GRIDS = {
    "lr": {"C": [0.01, 0.1, 1.0]},
    "rf": {"min_samples_leaf": [20, 100], "max_features": ["sqrt", 0.5]},
    "hgb": {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31]},
    "svm_linear": {"C": [0.001, 0.01, 0.1]},
    "svm_rbf": {"C": [0.3, 1.0], "gamma": ["scale"]},
}
NAMES = {
    "lr": "Logistic regression",
    "rf": "Random forest",
    "hgb": "Gradient boosting (HGB)",
    "svm_linear": "SVM (linear, calibrated)",
    "svm_rbf": "SVM (RBF, subsample)",
}


def make_model(name, params, seed=C.SEED):
    """An unfitted estimator for one model name and one parameter setting."""
    if name == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=params["C"], class_weight="balanced", max_iter=3000),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
            max_iter=300,
            min_samples_leaf=50,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=seed,
        )
    if name == "svm_linear":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=params["C"], class_weight="balanced", dual=False, max_iter=5000),
        )
    if name == "svm_rbf":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=params["C"],
                gamma=params["gamma"],
                kernel="rbf",
                class_weight="balanced",
                cache_size=1000,
            ),
        )
    raise ValueError(name)


def raw_score(model, X):
    """A monotone score: probability of the positive class, or the SVM decision value."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def subsample(X, y, n=RBF_SUBSAMPLE, seed=C.SEED):
    """Stratified subsample keeping the class ratio (used for the RBF kernel only)."""
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.concatenate(
        [
            rng.choice(np.flatnonzero(y == c), size=int(round(n * np.mean(y == c))), replace=False)
            for c in (0, 1)
        ]
    )
    return X[idx], y[idx]


def fit(name, params, X, y, seed=C.SEED):
    """Fit one model; the RBF SVM trains on a stratified subsample."""
    if name == "svm_rbf":
        X, y = subsample(X, y, seed=seed)
    return make_model(name, params, seed).fit(X, y)


def grid(name):
    """Every parameter setting in the grid of one model, as a list of dicts."""
    keys = list(GRIDS[name])
    return [dict(zip(keys, vals)) for vals in product(*(GRIDS[name][k] for k in keys))]


def loso_tune(name, X, y, seasons, train_seasons=C.TRAIN_SEASONS, seed=C.SEED, log=print):
    """Leave-one-season-out tuning over the grid of one model.

    Returns the best parameters (highest mean fold PR-AUC), the table of fold PR-AUCs and the
    out-of-fold scores of the best setting.
    """
    rows, oof = [], {}
    for params in grid(name):
        scores = np.full(len(y), np.nan)
        for s in train_seasons:
            tr, va = seasons != s, seasons == s
            t0 = time.time()
            m = fit(name, params, X[tr], y[tr], seed)
            scores[va] = raw_score(m, X[va])
            rows.append(
                {
                    "model": name,
                    **params,
                    "held_out_season": s,
                    "pr_auc": average_precision_score(y[va], scores[va]),
                    "seconds": round(time.time() - t0, 1),
                }
            )
        oof[str(params)] = scores
        fold_pr_auc = [r["pr_auc"] for r in rows[-len(train_seasons) :]]
        log(f"  {name} {params}: mean fold PR-AUC {np.mean(fold_pr_auc):.4f}")
    table = pd.DataFrame(rows)
    means = {
        str(params): table[np.all([table[k] == v for k, v in params.items()], axis=0)][
            "pr_auc"
        ].mean()
        for params in grid(name)
    }
    best = next(params for params in grid(name) if str(params) == max(means, key=means.get))
    return best, table, oof[str(best)]


class Calibrated:
    """A fitted model plus an isotonic map from its raw score to a probability, fitted on the validation season."""

    def __init__(self, name, params, seed=C.SEED):
        self.name, self.params, self.seed = name, params, seed

    def fit(self, X_train, y_train, X_cal, y_cal):
        self.model = fit(self.name, self.params, X_train, y_train, self.seed)
        s = raw_score(self.model, X_cal)
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, y_cal)
        return self

    def refit_calibration(self, scores, y):
        """Replace the calibrator, for example with pooled out-of-fold scores (reported as a sensitivity)."""
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, y)
        return self

    def raw(self, X):
        return raw_score(self.model, X)

    def predict_proba(self, X):
        return self.iso.predict(self.raw(X))
