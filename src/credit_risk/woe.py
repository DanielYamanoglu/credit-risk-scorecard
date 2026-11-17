"""Weight-of-evidence binning with monotonic merging, as used in credit scorecards."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_BINS = 10
MIN_BIN_SHARE = 0.05
MAX_DISCRETE_VALUES = 12
SMOOTHING = 0.5  # added to good and bad counts so an empty cell does not give infinite WoE


@dataclass
class WoEBinner:
    """WoE > 0 means fewer defaults than average. Fit on training data only."""

    monotonic: bool = True
    cuts: np.ndarray = field(default_factory=lambda: np.array([]))
    woe: np.ndarray = field(default_factory=lambda: np.array([]))
    table: pd.DataFrame | None = None
    iv: float = 0.0

    def _initial_cuts(self, x: np.ndarray) -> np.ndarray:
        values = np.unique(x)
        if len(values) <= MAX_DISCRETE_VALUES:
            return (values[:-1] + values[1:]) / 2
        quantiles = np.quantile(x, np.linspace(0, 1, MAX_BINS + 1)[1:-1])
        return np.unique(quantiles)

    @staticmethod
    def _stats(bins: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
        bad = np.bincount(bins, weights=y, minlength=n_bins)
        total = np.bincount(bins, minlength=n_bins)
        return total - bad, bad

    @staticmethod
    def _woe(good: np.ndarray, bad: np.ndarray) -> np.ndarray:
        g = (good + SMOOTHING) / (good.sum() + SMOOTHING * len(good))
        b = (bad + SMOOTHING) / (bad.sum() + SMOOTHING * len(bad))
        return np.log(g / b)

    def fit(self, x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> "WoEBinner":
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        cuts = list(self._initial_cuts(x))

        def stats(c):
            bins = np.searchsorted(np.asarray(c), x, side="right")
            return self._stats(bins, y, len(c) + 1)

        # Merge bins that hold too few accounts into their smaller neighbour.
        while cuts:
            good, bad = stats(cuts)
            share = (good + bad) / len(x)
            i = int(np.argmin(share))
            if share[i] >= MIN_BIN_SHARE:
                break
            neighbour = i - 1 if i == len(share) - 1 or (i > 0 and share[i - 1] < share[i + 1]) else i
            cuts.pop(neighbour)

        # Merge adjacent bins until WoE moves in one direction only.
        if self.monotonic:
            while cuts:
                good, bad = stats(cuts)
                woe = self._woe(good, bad)
                direction = np.sign(np.corrcoef(np.arange(len(woe)), woe)[0, 1]) if len(woe) > 2 else 1.0
                steps = np.diff(woe) * direction
                if (steps >= 0).all():
                    break
                cuts.pop(int(np.argmin(steps)))

        self.cuts = np.asarray(cuts)
        good, bad = stats(cuts)
        self.woe = self._woe(good, bad)
        g, b = good / good.sum(), bad / bad.sum()
        self.iv = float(((g - b) * self.woe).sum())
        edges = np.concatenate([[-np.inf], self.cuts, [np.inf]])
        self.table = pd.DataFrame(
            {
                "lower": edges[:-1],
                "upper": edges[1:],
                "accounts": (good + bad).astype(int),
                "bad_rate": bad / np.maximum(good + bad, 1),
                "woe": self.woe,
            }
        )
        return self

    def bin_index(self, x) -> np.ndarray:
        return np.searchsorted(self.cuts, np.asarray(x, dtype=float), side="right")

    def transform(self, x) -> np.ndarray:
        return self.woe[self.bin_index(x)]
