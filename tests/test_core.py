import math

import numpy as np
import pandas as pd
import pytest

from credit_risk import features, metrics
from credit_risk.scorecard import BASE_ODDS, BASE_SCORE, PDO, Scorecard
from credit_risk.train import make_gbm
from credit_risk.woe import MIN_BIN_SHARE, WoEBinner

RNG = np.random.default_rng(0)


def _synthetic(n=6000):
    """Default risk rises with delay and utilisation and falls with limit; the rest is noise."""
    X = pd.DataFrame({
        "current_delay": RNG.integers(-2, 4, n), "max_delay_6m": RNG.integers(0, 5, n),
        "months_late_6m": RNG.integers(0, 7, n), "utilization": RNG.uniform(0, 1.2, n),
        "avg_utilization_6m": RNG.uniform(0, 1.2, n), "payment_ratio": RNG.uniform(0, 2, n),
        "avg_payment_ratio_5m": RNG.uniform(0, 2, n), "balance_trend": RNG.normal(0, 0.3, n),
        "limit_bal": RNG.uniform(10_000, 500_000, n), "age": RNG.integers(21, 70, n),
        "education": RNG.integers(1, 5, n),
    })[features.FEATURES]
    logit = -1.5 + 0.8 * X.current_delay + 1.2 * X.utilization - 3e-6 * X.limit_bal
    y = pd.Series((RNG.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_woe_matches_hand_calculation():
    # Two bins: x=0 has 80 goods / 20 bads, x=1 has 50 / 50. Totals 130 goods, 70 bads.
    x = np.r_[np.zeros(100), np.ones(100)]
    y = np.r_[np.zeros(80), np.ones(20), np.zeros(50), np.ones(50)]
    b = WoEBinner(monotonic=False).fit(x, y)
    g = (np.array([80, 50]) + 0.5) / (130 + 1.0)
    bad = (np.array([20, 50]) + 0.5) / (70 + 1.0)
    assert np.allclose(b.woe, np.log(g / bad))
    raw_g, raw_b = np.array([80, 50]) / 130, np.array([20, 50]) / 70
    assert math.isclose(b.iv, float(((raw_g - raw_b) * b.woe).sum()))


def test_woe_binning_is_monotonic_and_respects_min_bin_size():
    x = RNG.uniform(0, 1, 5000)
    y = (RNG.uniform(size=5000) < 0.1 + 0.4 * x + RNG.normal(0, 0.05, 5000)).astype(int)
    b = WoEBinner().fit(x, y)
    assert len(b.woe) >= 3 and (np.diff(b.woe) <= 0).all()  # riskier as x grows
    assert (b.table["accounts"] / len(x) >= MIN_BIN_SHARE).all()


def test_scorecard_points_add_up_to_the_score_and_follow_the_scaling():
    X, y = _synthetic()
    sc = Scorecard().fit(X, y)
    assert (sc.model.coef_[0] <= 0).all()  # every characteristic points the way its WoE does
    table = sc.points_table()
    rows = X.sample(50, random_state=1)
    summed = [
        sum(
            table[(table.characteristic == c) & (table.lower < r[c]) & (r[c] <= table.upper)].points.iloc[0]
            for c in sc.selected
        )
        for _, r in rows.iterrows()
    ]
    assert np.allclose(summed, sc.score(rows))
    # Score maps to odds: BASE_SCORE at BASE_ODDS, and PDO more points doubles the odds of repaying.
    pd_ = sc.predict_proba(rows)
    odds = (1 - pd_) / pd_
    assert np.allclose(sc.score(rows), BASE_SCORE + PDO * np.log2(odds / BASE_ODDS))


def test_gradient_boosting_respects_monotone_constraints():
    X, y = _synthetic()
    model = make_gbm().fit(X, y)
    probe = pd.concat([X.iloc[[0]]] * 6, ignore_index=True)
    probe["current_delay"] = range(-2, 4)
    assert (np.diff(model.predict_proba(probe)[:, 1]) >= -1e-12).all()
    probe = pd.concat([X.iloc[[0]]] * 5, ignore_index=True)
    probe["limit_bal"] = np.linspace(10_000, 500_000, 5)
    assert (np.diff(model.predict_proba(probe)[:, 1]) <= 1e-12).all()


def test_protected_characteristics_are_not_features():
    assert not {"sex", "marriage"} & set(features.FEATURES)


def test_payment_ratio_treats_nothing_owed_as_paid():
    paid, bill = pd.Series([0.0, 500.0, 100.0]), pd.Series([0.0, 0.0, 400.0])
    assert features._payment_ratio(paid, bill).tolist() == [1.0, 1.0, 0.25]  # nothing owed counts as paid in full


def test_ks_gini_and_psi_on_known_cases():
    y = np.r_[np.zeros(100), np.ones(100)]
    assert metrics.ks(y, np.r_[np.zeros(100), np.ones(100)]) == pytest.approx(1.0)
    assert metrics.gini(y, np.r_[np.zeros(100), np.ones(100)]) == pytest.approx(1.0)
    assert abs(metrics.gini(y, RNG.uniform(size=200))) < 0.2
    reference = RNG.normal(size=5000)
    assert metrics.psi(reference, RNG.normal(size=5000)) < 0.02
    assert metrics.psi(reference, RNG.normal(1.0, 1, 5000)) > 0.25


def test_best_cutoff_by_hand():
    # Ranked by PD: good, good, bad, good, bad. Gain 1, loss 5:
    # cumulative profit 1, 2, -3, -2, -7, so approve the two safest.
    y = np.array([0, 0, 1, 0, 1])
    result = metrics.best_cutoff(y, np.array([0.1, 0.2, 0.3, 0.4, 0.5]), gain_good=1, loss_bad=5)
    assert result["approval_rate"] == pytest.approx(0.4) and result["bad_rate_approved"] == 0.0
    assert result["profit_per_1000_applicants"] == pytest.approx(400.0)
