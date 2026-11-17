"""Behavioural features from six months of repayment history.

Sex and marital status are deliberately left out: they are protected characteristics and
should not drive a credit decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BILL_COLS, PAID_COLS, PAY_COLS

FEATURES = [
    "current_delay",
    "max_delay_6m",
    "months_late_6m",
    "utilization",
    "avg_utilization_6m",
    "payment_ratio",
    "avg_payment_ratio_5m",
    "balance_trend",
    "limit_bal",
    "age",
    "education",
]
UTILIZATION_CLIP = (-1.0, 3.0)
PAYMENT_RATIO_CLIP = (0.0, 2.0)


def _payment_ratio(paid: pd.Series, bill: pd.Series) -> pd.Series:
    """Share of the statement paid. With nothing owed, the account is treated as fully paid."""
    ratio = paid / bill.where(bill > 0)
    return ratio.fillna(1.0).clip(*PAYMENT_RATIO_CLIP)


def build(frame: pd.DataFrame) -> pd.DataFrame:
    delays = frame[PAY_COLS].clip(lower=0)
    limit = frame["limit_bal"].astype(float)
    bills = frame[BILL_COLS].astype(float)
    paid = frame[PAID_COLS].astype(float)

    # A payment in month t settles the statement of month t+1 (columns run newest to oldest).
    ratios = pd.concat(
        [_payment_ratio(paid[PAID_COLS[i]], bills[BILL_COLS[i + 1]]) for i in range(5)], axis=1
    )
    out = pd.DataFrame(
        {
            "current_delay": frame["pay_1"],
            "max_delay_6m": delays.max(axis=1),
            "months_late_6m": (delays > 0).sum(axis=1),
            "utilization": (bills["bill_amt1"] / limit).clip(*UTILIZATION_CLIP),
            "avg_utilization_6m": bills.div(limit, axis=0).mean(axis=1).clip(*UTILIZATION_CLIP),
            "payment_ratio": ratios.iloc[:, 0],
            "avg_payment_ratio_5m": ratios.mean(axis=1),
            "balance_trend": ((bills["bill_amt1"] - bills["bill_amt6"]) / limit).clip(-3, 3),
            "limit_bal": limit,
            "age": frame["age"],
            # 1 graduate school, 2 university, 3 high school; undocumented codes grouped as 4.
            "education": frame["education"].where(frame["education"].isin([1, 2, 3]), 4),
        }
    )
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("feature table contains non-finite values")
    return out[FEATURES]
