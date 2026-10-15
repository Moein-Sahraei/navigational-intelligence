"""Figure 1 for the worked example (reads worked_example.json)."""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data = json.load(open(sys.argv[1]))
out_stem = sys.argv[2]
K, m = data["K"], data["m"]
v = K - m
rows = data["archive"]
n = [r["n"] for r in rows]
et_out = [r["E[T] outcome-only (= bare)"] for r in rows]
et_diag = [r["E[T] diagnosed"] for r in rows]
info_diag = [r["I(S_next; diagnosed trajectories | outcomes)"] for r in rows]
info_bare = [0.0 for _ in rows]

# Palette (validated: categorical slots 1-2) and chrome
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

LW = 1.5           # ~2 px at print size
MS = 5.5           # marker diameter in points (>= 8 px on screen)
ring = dict(markeredgecolor=SURF, markeredgewidth=1.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)

for ax in (ax1, ax2):
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(-0.3, 10.6)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("Archived problems $n$")

# (a) expected attempts on the next problem
ax1.axhline((K + 1) / 2, color=MUTED, linewidth=0.9, zorder=1)
ax1.axhline((v + 1) / 2, color=MUTED, linewidth=0.9, zorder=1)
ax1.text(10.6, (K + 1) / 2 + 0.06, "no memory", ha="right", va="bottom", color=INK2, fontsize=7)
ax1.text(10.6, (v + 1) / 2 - 0.06, "obstructions fully known", ha="right", va="top",
         color=INK2, fontsize=7)
ax1.plot(n, et_out, color=ORANGE, lw=LW, marker="s", ms=MS, solid_capstyle="round", zorder=3, **ring)
ax1.plot(n, et_diag, color=BLUE, lw=LW, marker="o", ms=MS, solid_capstyle="round", zorder=4, **ring)
ax1.set_ylabel("Expected attempts on the next problem")
ax1.set_ylim(2.2, 4.75)
ax1.set_title("(a) Search cost", loc="left", fontsize=8.5, color=INK)
# selective direct labels at n = 1
ax1.annotate(f"{et_out[1]:.1f}", (1, et_out[1]), xytext=(6, 2), textcoords="offset points",
             color=INK2, fontsize=7, va="bottom")
ax1.annotate(f"{et_diag[1]:.1f}", (1, et_diag[1]), xytext=(-4, -9), textcoords="offset points",
             color=INK2, fontsize=7, ha="right")

# (b) task-relevant navigational information
ax2.plot(n, info_bare, color=ORANGE, lw=LW, marker="s", ms=MS, zorder=3, **ring)
ax2.plot(n, info_diag, color=BLUE, lw=LW, marker="o", ms=MS, zorder=4, **ring)
ax2.set_ylabel(r"$I(S_{n+1};\,\Gamma_{1:n}\mid E_{1:n})$  (bits)")
ax2.set_ylim(-0.04, 0.8)
ax2.set_title("(b) Task-relevant navigational information", loc="left", fontsize=8.5, color=INK)
k = max(range(len(info_diag)), key=lambda i: info_diag[i])
ax2.annotate(f"{info_diag[k]:.2f} bits", (n[k], info_diag[k]), xytext=(6, 4),
             textcoords="offset points", color=INK2, fontsize=7)

handles = [
    Line2D([0], [0], color=BLUE, lw=LW, marker="o", ms=MS, **ring,
           label="diagnosed trajectories"),
    Line2D([0], [0], color=ORANGE, lw=LW, marker="s", ms=MS, **ring,
           label="outcomes only, or bare trajectories"),
]
fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False,
           fontsize=7.5, labelcolor=INK2)

fig.savefig(out_stem + ".pdf")
fig.savefig(out_stem + ".png", dpi=300)
print("saved", out_stem)
