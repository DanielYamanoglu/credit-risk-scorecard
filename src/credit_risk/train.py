"""Fit both models on the training split, cross-validate, and score the hold-out once."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from . import data, features, metrics
from .scorecard import Scorecard

REPORTS = data.ROOT / "reports"
MODELS = data.ROOT / "models"
FOLDS = 5
# Expected profit per approved account, relative units: a repaying account earns 1, a
# defaulting one loses 5 (roughly a high loss-given-default on an unsecured card).
GAIN_GOOD, LOSS_BAD = 1.0, 5.0
# Business-sense constraints: more delay or utilisation never lowers risk, more limit,
# repayment or age never raises it.
MONOTONE = {
    "current_delay": 1, "max_delay_6m": 1, "months_late_6m": 1, "utilization": 1, "avg_utilization_6m": 1,
    "payment_ratio": -1, "avg_payment_ratio_5m": -1, "limit_bal": -1, "age": 0, "balance_trend": 0, "education": 0,
}
GBM_PARAMS = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 100,
              "l2_regularization": 1.0, "random_state": data.SEED}


def make_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        monotonic_cst=[MONOTONE[c] for c in features.FEATURES], categorical_features=["education"], **GBM_PARAMS
    )


def cross_validate(X: pd.DataFrame, y: pd.Series) -> dict[str, list[float]]:
    scores = {"scorecard": [], "gradient_boosting": []}
    for fit_idx, val_idx in StratifiedKFold(FOLDS, shuffle=True, random_state=data.SEED).split(X, y):
        Xf, yf, Xv, yv = X.iloc[fit_idx], y.iloc[fit_idx], X.iloc[val_idx], y.iloc[val_idx]
        scores["scorecard"].append(metrics.gini(yv, Scorecard().fit(Xf, yf).predict_proba(Xv)))
        scores["gradient_boosting"].append(metrics.gini(yv, make_gbm().fit(Xf, yf).predict_proba(Xv)[:, 1]))
    return scores


def run() -> dict:
    train, test = data.split(data.load())
    X_train, y_train = features.build(train), train[data.TARGET]
    X_test, y_test = features.build(test), test[data.TARGET]

    cv = cross_validate(X_train, y_train)
    scorecard = Scorecard().fit(X_train, y_train)
    gbm = make_gbm().fit(X_train, y_train)
    pd_test = {"scorecard": scorecard.predict_proba(X_test), "gradient_boosting": gbm.predict_proba(X_test)[:, 1]}
    pd_train = {"scorecard": scorecard.predict_proba(X_train), "gradient_boosting": gbm.predict_proba(X_train)[:, 1]}

    report = {
        "dataset": {"accounts": len(train) + len(test), "train": len(train), "test": len(test),
                    "default_rate": float(pd.concat([y_train, y_test]).mean())},
        "information_value": scorecard.information_values(),
        "scorecard_characteristics": scorecard.selected,
        "cv_gini": {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in cv.items()},
        "test": {m: metrics.summary(y_test, p) for m, p in pd_test.items()},
        "psi_train_vs_test": {m: metrics.psi(pd_train[m], pd_test[m]) for m in pd_test},
        "cutoff": {m: metrics.best_cutoff(y_test, p, GAIN_GOOD, LOSS_BAD) for m, p in pd_test.items()},
        "deciles": {m: metrics.decile_table(y_test, p).round(4).to_dict(orient="records") for m, p in pd_test.items()},
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2))
    scorecard.points_table().round(3).to_csv(REPORTS / "scorecard.csv", index=False)
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"scorecard": scorecard, "gradient_boosting": gbm}, MODELS / "models.joblib")

    from .figures import render

    render(y_test, pd_test, scorecard.score(X_test), REPORTS / "figures")
    return report
