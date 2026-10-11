"""ROC and PR curves of the calibrated GBT on the 2024 test set, one curve per threshold, with each AUC and the
mean over thresholds. Reads the predictions written by `predict.py --evaluate`, copied locally first:

  gsutil -m cp -r <bucket>/results/predictions data/results/
  python scripts/plot_test_auc.py

Writes figures/5_test_roc_pr.png. Needs pandas + pyarrow, no Spark.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLDS = [0.0, 0.2, 1.0, 5.0, 10.0]   # EVAL_THRESHOLDS in train.py
COLORS = ["#1f5fa8", "#2e9e8f", "#7fb800", "#e08a00", "#c0392b"]


def curves(p, y):
    """ROC (fpr, tpr) and PR (recall, precision) at every distinct probability, highest first."""
    order = np.argsort(-p, kind="stable")
    p, y = p[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]   # last row of each run of tied probabilities
    tp = np.cumsum(y)[last]
    fp = last + 1 - tp
    tpr, fpr = tp / tp[-1], fp / fp[-1]
    precision = tp / (tp + fp)
    return np.r_[0, fpr], np.r_[0, tpr], np.r_[0, tpr], np.r_[precision[0], precision]


preds = pd.read_parquet("data/results/predictions", columns=["threshold", "label", "cal"])
fig, (roc, pr) = plt.subplots(1, 2, figsize=(12, 6))
roc_aucs, pr_aucs = [], []
for t, color in zip(THRESHOLDS, COLORS):
    rows = preds[np.isclose(preds.threshold, t)]
    y = rows.label.to_numpy()
    fpr, tpr, recall, precision = curves(rows.cal.to_numpy(), y)
    roc_aucs.append(np.trapezoid(tpr, fpr))
    pr_aucs.append(np.trapezoid(precision, recall))
    roc.plot(fpr, tpr, color=color, label=f"> {t:g} mm (AUC = {roc_aucs[-1]:.3f})")
    pr.plot(recall, precision, color=color,
            label=f"> {t:g} mm (AUC = {pr_aucs[-1]:.3f}, base rate {y.mean():.2%})")

roc.plot([0, 1], [0, 1], color="black", ls=":", lw=1, label="No skill (AUC = 0.5)")
roc.set(xlabel="False positive rate", ylabel="True positive rate (POD)", xlim=(0, 1), ylim=(0, 1.01),
        title=f"(a) ROC curves — mean AUC = {np.mean(roc_aucs):.3f}")
roc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9)
pr.set(xlabel="Recall (POD)", ylabel="Precision (1 − FAR)", xlim=(0, 1), ylim=(0, 1.01),
       title=f"(b) Precision–recall curves — mean AUC = {np.mean(pr_aucs):.3f}")
pr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9)
for ax in (roc, pr):
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("figures/5_test_roc_pr.png", dpi=200, bbox_inches="tight")
print("ROC-AUC", np.round(roc_aucs, 4), "PR-AUC", np.round(pr_aucs, 4))
