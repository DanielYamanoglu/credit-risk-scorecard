from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from . import metrics

LABELS = {"scorecard": "WoE scorecard", "gradient_boosting": "Monotonic gradient boosting"}
COLORS = {"scorecard": "#0284c7", "gradient_boosting": "#d97706"}


def render(y, pd_by_model: dict, scores: np.ndarray, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for m, p in pd_by_model.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=COLORS[m], label=f"{LABELS[m]} (Gini {metrics.gini(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", ls="--", lw=1)
    ax.set(title="ROC on the hold-out set", xlabel="false positive rate", ylabel="true positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    ax = axes[1]
    width = 0.4
    for i, (m, p) in enumerate(pd_by_model.items()):
        table = metrics.decile_table(y, p)
        ax.bar(table["decile"] + (i - 0.5) * width, table["bad_rate"], width, color=COLORS[m], label=LABELS[m])
    ax.axhline(y.mean(), color="#475569", ls=":", lw=1, label=f"portfolio default rate {y.mean():.1%}")
    ax.set(title="Default rate by risk decile (1 = riskiest)", xlabel="decile", ylabel="observed default rate")
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    bins = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 40)
    ax.hist(scores[y == 0], bins, alpha=0.6, color="#16a34a", density=True, label="repaid")
    ax.hist(scores[y == 1], bins, alpha=0.6, color="#dc2626", density=True, label="defaulted")
    ax.set(title="Scorecard points by outcome", xlabel="score (higher = safer)", ylabel="density")
    ax.legend(frameon=False, fontsize=9)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for m, p in pd_by_model.items():
        table = metrics.decile_table(y, p)
        ax.plot(table["mean_pd"], table["bad_rate"], "o-", color=COLORS[m], label=LABELS[m])
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color="#94a3b8", ls="--", lw=1, label="perfect calibration")
    ax.set(title="Calibration by decile", xlabel="mean predicted PD", ylabel="observed default rate")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=130)
    plt.close(fig)
