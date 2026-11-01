"""Figure 2: estimates from trained predictors (reads learned_estimates.json)."""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

data = json.load(open(sys.argv[1]))
out_stem = sys.argv[2]
exact = data["exact"]

# Palette (validated: categorical slots 1-2, same as Figure 1) and chrome
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})
LW, MS = 1.5, 5.5
ring = dict(markeredgecolor=SURF, markeredgewidth=1.0)


def summarize(rows, key, field):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r[field])
    xs = sorted(groups)
    return (np.array(xs), np.array([np.mean(groups[x]) for x in xs]),
            np.array([np.std(groups[x], ddof=1) if len(groups[x]) > 1 else 0.0 for x in xs]))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.0, 2.65), constrained_layout=True,
                                    gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
for ax in (ax1, ax2, ax3):
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# (a) learning curve, one archived problem
lc = data["A_learning_curve"]
x, md, sd = summarize(lc, "n_train", "gap_diagnosed")
_, mb, sb = summarize(lc, "n_train", "gap_bare")
exact1 = exact["Delta_I_Y_diagnosed"][1]
ax1.axhline(exact1, color=MUTED, lw=0.9, ls="--", zorder=1)
ax1.axhline(0.0, color=MUTED, lw=0.9, ls="--", zorder=1)
ax1.text(x[-1], exact1 + 0.02, f"exact {exact1:.3f}", ha="right", va="bottom", color=INK2, fontsize=7)
ax1.errorbar(x, md, yerr=sd, color=BLUE, lw=LW, marker="o", ms=MS, capsize=0, elinewidth=1.0, zorder=4, **ring)
ax1.errorbar(x, mb, yerr=sb, color=ORANGE, lw=LW, marker="s", ms=MS, capsize=0, elinewidth=1.0, zorder=3, **ring)
ax1.set_xscale("log", base=2)
ax1.set_xticks([250, 1000, 4000, 16000, 64000])
ax1.set_xticklabels(["250", "1k", "4k", "16k", "64k"])
ax1.set_xlabel("Training archives")
ax1.set_ylabel("Estimated $\\Delta I_Y$ (bits)")
ax1.set_ylim(-0.16, 0.75)
ax1.set_title("(a) One archived problem", loc="left", fontsize=8.5, color=INK)

# (b) archives of n problems
ar = data["A_archive_size"]
n, md, sd = summarize(ar, "n", "gap_diagnosed")
_, mb, sb = summarize(ar, "n", "gap_bare")
ax2.plot(exact["n"], exact["Delta_I_Y_diagnosed"], color=MUTED, lw=0.9, ls="--", zorder=1)
ax2.axhline(0.0, color=MUTED, lw=0.9, ls="--", zorder=1)
ax2.errorbar(n, md, yerr=sd, color=BLUE, lw=0, marker="o", ms=MS, elinewidth=1.0, zorder=4, **ring)
ax2.errorbar(n, mb, yerr=sb, color=ORANGE, lw=0, marker="s", ms=MS, elinewidth=1.0, zorder=3, **ring)
ax2.text(7.25, 0.33, "exact", color=INK2, fontsize=7, va="bottom")
ax2.set_xlim(-0.4, 10.4)
ax2.set_xticks(range(0, 11, 2))
ax2.set_xlabel("Archived problems $n$")
ax2.set_ylim(-0.16, 0.75)
ax2.set_title("(b) Archives of $n$ problems", loc="left", fontsize=8.5, color=INK)

# (c) information versus computation
bc = data["B_capacity"]
widths = sorted({r["width"] for r in bc})
pos = {w: i for i, w in enumerate(widths)}
for fault, color, marker, z in ((True, BLUE, "o", 4), (False, ORANGE, "s", 3)):
    rows = [r for r in bc if r["hidden_fault"] == fault]
    w, mg, sg = summarize(rows, "width", "gap")
    ax3.errorbar([pos[v] for v in w], mg, yerr=sg, color=color, lw=LW, marker=marker, ms=MS,
                 elinewidth=1.0, zorder=z, **ring)
ax3.axhline(1.0, color=MUTED, lw=0.9, ls="--", zorder=1)
ax3.axhline(0.0, color=MUTED, lw=0.9, ls="--", zorder=1)
ax3.set_xticks(range(len(widths)))
ax3.set_xticklabels(["lin" if v == 0 else str(v) for v in widths], fontsize=7)
ax3.set_xlabel("Hidden units of the learner")
ax3.set_ylabel("Estimated usable information (bits)")
ax3.set_ylim(-0.12, 1.25)
ax3.set_title("(c) Information vs. computation", loc="left", fontsize=8.5, color=INK)

handles = [
    Line2D([0], [0], color=BLUE, lw=LW, marker="o", ms=MS, **ring,
           label="diagnosed trajectories / hidden fault"),
    Line2D([0], [0], color=ORANGE, lw=LW, marker="s", ms=MS, **ring,
           label="bare trajectories / computable trace"),
    Line2D([0], [0], color=MUTED, lw=0.9, ls="--", label="exact Shannon value"),
]
fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
           fontsize=7.5, labelcolor=INK2)
fig.savefig(out_stem + ".pdf")
fig.savefig(out_stem + ".png", dpi=300)
print("saved", out_stem)
