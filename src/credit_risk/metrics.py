"""Validation metrics used for retail credit models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

DECILES = 10


def gini(y, pd_) -> float:
    return 2 * roc_auc_score(y, pd_) - 1


def ks(y, pd_) -> float:
    """Kolmogorov-Smirnov: largest gap between the score distributions of bads and goods."""
    fpr, tpr, _ = roc_curve(y, pd_)
    return float(np.max(tpr - fpr))


def decile_table(y, pd_) -> pd.DataFrame:
    """Accounts ranked by PD into deciles (1 = riskiest); a good model gives falling bad rates."""
    frame = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_)})
    frame["decile"] = DECILES - pd.qcut(frame["pd"].rank(method="first"), DECILES, labels=False)
    table = frame.groupby("decile").agg(accounts=("y", "size"), bad_rate=("y", "mean"), mean_pd=("pd", "mean"))
    return table.reset_index()


def expected_calibration_error(y, pd_) -> float:
    table = decile_table(y, pd_)
    return float((table["accounts"] * (table["bad_rate"] - table["mean_pd"]).abs()).sum() / table["accounts"].sum())


def psi(reference, current, bins: int = 10) -> float:
    """Population stability index; above 0.25 is the usual signal to redevelop the model."""
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-4, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def summary(y, pd_) -> dict:
    return {
        "auc": float(roc_auc_score(y, pd_)),
        "gini": float(gini(y, pd_)),
        "ks": ks(y, pd_),
        "brier": float(brier_score_loss(y, pd_)),
        "calibration_error": expected_calibration_error(y, pd_),
    }


def best_cutoff(y, pd_, gain_good: float, loss_bad: float) -> dict:
    """Approve accounts below a PD cutoff; pick the cutoff with the highest expected profit."""
    y, pd_ = np.asarray(y), np.asarray(pd_)
    order = np.argsort(pd_)
    goods = np.cumsum(1 - y[order])
    bads = np.cumsum(y[order])
    profit = gain_good * goods - loss_bad * bads
    k = int(np.argmax(profit))
    approve_all = gain_good * (1 - y).sum() - loss_bad * y.sum()
    return {
        "pd_cutoff": float(pd_[order][k]),
        "approval_rate": (k + 1) / len(y),
        "bad_rate_approved": float(bads[k] / (k + 1)),
        "bad_rate_all": float(y.mean()),
        "profit_per_1000_applicants": float(profit[k] / len(y) * 1000),
        "approve_all_profit_per_1000_applicants": float(approve_all / len(y) * 1000),
    }
