"""Logistic regression on WoE features, scaled to points like a bank application scorecard."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .woe import WoEBinner

MIN_IV = 0.02  # below this a characteristic is conventionally "not predictive"
NON_MONOTONIC = {"education"}  # nominal categories, no natural order
# Scaling: 600 points at good:bad odds of 50:1, odds double every 20 points.
BASE_SCORE, BASE_ODDS, PDO = 600, 50.0, 20


@dataclass
class Scorecard:
    binners: dict[str, WoEBinner] = field(default_factory=dict)
    selected: list[str] = field(default_factory=list)
    model: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Scorecard":
        self.binners = {c: WoEBinner(monotonic=c not in NON_MONOTONIC).fit(X[c], y) for c in X.columns}
        self.selected = [c for c in X.columns if self.binners[c].iv >= MIN_IV]
        # Drop characteristics whose coefficient comes out with the wrong sign (collinearity),
        # so every point in the scorecard moves in the direction its WoE says.
        while True:
            model = LogisticRegression(C=1.0, max_iter=1000).fit(self._woe(X), y)
            wrong = [c for c, b in zip(self.selected, model.coef_[0]) if b > 0]
            if not wrong:
                break
            self.selected.remove(max(wrong, key=lambda c: model.coef_[0][self.selected.index(c)]))
        self.model = model
        return self

    def _woe(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.binners[c].transform(X[c]) for c in self.selected])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default."""
        return self.model.predict_proba(self._woe(X))[:, 1]

    @property
    def _scale(self) -> tuple[float, float]:
        factor = PDO / math.log(2)
        return factor, BASE_SCORE - factor * math.log(BASE_ODDS)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        factor, offset = self._scale
        log_odds_good = -(self.model.intercept_[0] + self._woe(X) @ self.model.coef_[0])
        return offset + factor * log_odds_good

    def points_table(self) -> pd.DataFrame:
        """Points per attribute; an applicant's score is the sum of their attributes' points."""
        factor, offset = self._scale
        base = (offset - factor * self.model.intercept_[0]) / len(self.selected)
        rows = []
        for c, b in zip(self.selected, self.model.coef_[0]):
            for _, r in self.binners[c].table.iterrows():
                rows.append({"characteristic": c, "lower": r.lower, "upper": r.upper, "accounts": r.accounts,
                             "bad_rate": r.bad_rate, "woe": r.woe, "points": base - factor * b * r.woe})
        return pd.DataFrame(rows)

    def information_values(self) -> dict[str, float]:
        return {c: b.iv for c, b in sorted(self.binners.items(), key=lambda kv: -kv[1].iv)}
