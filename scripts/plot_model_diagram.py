"""Draw how the Task 2 model is built (train.py): threshold replication, the training pipeline, and the
distribution thresholds are drawn from, one figure each. Plain matplotlib, no Spark. Writes
figures/4a_replication.png, 4b_pipeline.png and 4c_threshold_sampling.png.

Run: python scripts/plot_model_diagram.py
"""

import math

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GRID_STEPS = 150   # as in train.py
BLUE, GREEN, RED, GREY = "#dbe8f6", "#2e8b57", "#c0392b", "#f2f2f2"


def box(ax, x, y, w, h, text, color=GREY, size=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=color, ec="#555"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4, color="#333"))


def draw_j(u):
    return math.floor(math.exp(u * math.log(GRID_STEPS + 2))) - 1


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=200)


# (a) One row becomes up to three rows with different thresholds.
fig, ax = plt.subplots(figsize=(13, 4))
ax.set(xlim=(-0.02, 1.02), ylim=(0, 1)); ax.axis("off")
box(ax, 0.00, 0.30, 0.22, 0.45,
    "One training row\nstation S50, 2019-11-12 14:00\n\nfeatures: rain_30m = 0.6,\nrain_1h = 1.0, hour = 14, …\n\n"
    "rain_next_30m = 3.4 mm", BLUE)
box(ax, 0.29, 0.30, 0.21, 0.45,
    "For copy c = 0, 1, 2:\n\nu = hash(station, time, c)\n      mod 10⁶ / 10⁶\n\nj = ⌊152ᵘ⌋ − 1\nthreshold = j × 0.2 mm\n\n"
    "label = rain_next_30m > threshold\n(duplicate j dropped)", GREY, 8.5)
arrow(ax, 0.22, 0.525, 0.29, 0.525)
for i, (c, u) in enumerate([(0, 0.10), (1, 0.30), (2, 0.62)]):
    j = draw_j(u)
    t = j / 5
    label = int(3.4 > t)
    y = 0.70 - i * 0.30
    box(ax, 0.57, y, 0.42, 0.22,
        f"copy {c}:  u = {u:.2f} → j = {j} → threshold = {t:g} mm\n"
        f"same features + threshold = {t:g}     label = {label}  (3.4 {'>' if label else '≤'} {t:g})",
        "#e6f4ea" if label else "#fbe9e7", 9)
    arrow(ax, 0.50, 0.525, 0.57, y + 0.11)
save(fig, "4a_replication")

# (b) Training pipeline.
fig, ax = plt.subplots(figsize=(13, 2.6))
ax.set(xlim=(-0.02, 1.02), ylim=(0, 1)); ax.axis("off")
steps = [("features/\n2017–2022\n(whole days sampled)", BLUE),
         ("Replicate each row\n≤ 3 times with random\nthresholds + label", GREY),
         ("Add climatology\nP(> threshold | station,\nmonth, hour) from\ntraining years only", GREY),
         ("One GBT classifier\ninputs: all feature\ngroups + threshold", "#fff2cc"),
         ("Isotonic calibration\non 2023\n(thresholds 0, 0.2,\n1, 5, 10 mm)", "#fff2cc"),
         ("P(rain in next 30 min\n> threshold)\nTested once on 2024", "#e6f4ea")]
w, gap = 0.145, 0.026
for i, (text, color) in enumerate(steps):
    x = i * (w + gap)
    box(ax, x, 0.15, w, 0.7, text, color, 8.5)
    if i:
        arrow(ax, x - gap, 0.5, x, 0.5)
save(fig, "4b_pipeline")

# (c) How often each threshold is drawn: P(j) proportional to log((j + 2) / (j + 1)).
fig, ax = plt.subplots(figsize=(13, 4.5))
js = range(GRID_STEPS + 1)
p = [math.log((j + 2) / (j + 1)) / math.log(GRID_STEPS + 2) for j in js]
ax.bar([j / 5 for j in js], p, width=0.18, color="#4a7fb5")
ax.set_yscale("log")
ax.set(xlabel="threshold (mm)", ylabel="probability per draw (log scale)", xlim=(-0.3, 30.3))
for t in (0.2, 1, 5, 10):
    j = round(t * 5)
    ax.axvline(t, color=RED, ls="--", lw=0.8)
    ax.text(t + 0.2, p[j] * 1.6, f"{t:g} mm: {p[j]:.1%}", color=RED, fontsize=8)
ax.text(29.8, p[0] * 0.8, f"P(≤ 1 mm) = {math.log(7) / math.log(152):.0%}\n"
        f"P(> 10 mm) = {1 - math.log(52) / math.log(152):.0%}", ha="right", va="top", fontsize=9)
save(fig, "4c_threshold_sampling")
