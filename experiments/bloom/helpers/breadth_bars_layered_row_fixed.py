#!/usr/bin/env python3
"""Single-row layered view, FIXED method order: the three overlaid method bars at each
(behaviour, model) position are always drawn in the same depth order regardless of height ---
WILT (red) at the back, LogitTilt (orange) in the middle, Vanilla (green) in front. So the
colours keep a consistent layering across every cell. Run: python -X utf8 breadth_bars_layered_row_fixed.py
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory as blend

REPO = "/workspace/inversion_optimisation"
HERE = os.path.dirname(__file__)

BEH = ["Racial\nbias", "Political\nbias", "Reinforce\ndelusions", "Self-harm\nencourage",
       "Dangerous\nmedical", "Strategic\ndeception", "Self-\npreservation", "Goblin\nfixation"]
MODELS = ["Llama-3.2-3B", "Phi-4-mini", "Qwen3.5-4B", "Gemma-4-E4B"]
METHODS = ["Vanilla", "LogitTilt", "WILT"]

V = {
    "Llama-3.2-3B": [83.6, 90.6, 98.0, 83.3, 70.1, 86.3, 52.6, 11.4],
    "Phi-4-mini":   [78.0, 79.6, 93.0, 50.3, 48.5, 88.8, 62.3, 13.0],
    "Qwen3.5-4B":   [60.2, 84.3, 67.4, 51.0, 29.9, 67.6, 54.9, 11.9],
    "Gemma-4-E4B":  [65.3, 74.5, 91.0, 34.6, 17.1, 72.6, 47.4, 11.3],
}
LT = {
    "Llama-3.2-3B": [92.5, 96.8, 99.5, 99.4, 91.8, 96.9, 81.8, 90.3],
    "Phi-4-mini":   [78.0, 95.2, 99.9, 98.6, 99.9, 88.8, 65.5, 87.8],
    "Qwen3.5-4B":   [66.6, 98.6, 99.9, 99.5, 93.6, 98.4, 99.9, 98.3],
    "Gemma-4-E4B":  [65.0, 73.7, 96.7, 47.4, 13.6, 67.0, 48.1, 10.5],
}
W = {
    "Llama-3.2-3B": [97.0, 99.5, 99.9, 99.9, 81.5, 98.3, 93.9, 86.2],
    "Phi-4-mini":   [86.5, 98.6, 100.0, 99.7, 100.0, 92.5, 86.8, 85.0],
    "Qwen3.5-4B":   [79.1, 99.1, 100.0, 100.0, 92.6, 97.2, 100.0, 96.1],
    "Gemma-4-E4B":  [70.1, 80.4, 98.9, 32.2, 19.1, 70.4, 48.2, 12.3],
}
DATA = {"Vanilla": V, "LogitTilt": LT, "WILT": W}
# colour-blind-safe (Okabe-Ito): green / orange / blue -- no green-red pairing
COL = {"Vanilla": "#009E73", "LogitTilt": "#E69F00", "WILT": "#0072B2"}
HATCH = {"Vanilla": "", "LogitTilt": "", "WILT": ""}   # no hatching; CB-safe colours only
DRAW_ORDER = ["WILT", "LogitTilt", "Vanilla"]   # back -> front (fixed, independent of height)

plt.rcParams.update({"font.size": 11, "hatch.linewidth": 0.6})
n = len(BEH)
width = 0.185
group_gap = 1.45
step = 0.32
spread = 0.05         # horizontal fan offset per depth: WILT back-left -> Vanilla front-right
offs = {mod: (i - 1.5) * step for i, mod in enumerate(MODELS)}
gx = lambda gi: gi * group_gap

fig, ax = plt.subplots(figsize=(12.0, 5.2), dpi=160)
tickpos, ticklab = [], []
for gi in range(n):
    for mod in MODELS:
        xc = gx(gi) + offs[mod]
        tickpos.append(xc); ticklab.append(mod)
        for depth, m in enumerate(DRAW_ORDER):        # WILT back -> LogitTilt -> Vanilla front
            xp = xc + (depth - 1) * spread              # centre the fan on the model position
            ax.bar(xp, DATA[m][mod][gi], width, color=COL[m], edgecolor="white",
                   linewidth=0.5, hatch=HATCH[m], zorder=3 + depth)

ax.set_ylim(0, 106)
ax.set_yticks([0, 50, 100])
ax.set_xlim(-0.66, gx(n - 1) + 0.66)
ax.set_ylabel("Behaviour presence (%)", fontsize=13)
for yv in (50, 100):
    ax.axhline(yv, color="#e6e5de", lw=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_edgecolor("black"); s.set_linewidth(0.9); s.set_zorder(30)
for xi in range(1, n):
    ax.axvline(gx(xi) - group_gap / 2, color="#b9b9b9", lw=1.0, zorder=1)

ax.set_xticks(tickpos)
ax.set_xticklabels(ticklab, fontsize=8.5, rotation=90, va="top")
ax.tick_params(axis="x", length=0)
tr = blend(ax.transData, ax.transAxes)
for gi in range(n):
    ax.text(gx(gi), 1.02, BEH[gi], transform=tr, ha="center", va="bottom", fontsize=13)

LEG = ["WILT", "LogitTilt", "Vanilla"]   # match the bar draw order (back -> front)
handles = [Patch(facecolor=COL[m], hatch=HATCH[m], edgecolor="#333333", linewidth=0.8, label=m) for m in LEG]
fig.legend(handles, LEG, ncol=3, frameon=False, fontsize=13,
           loc="upper center", bbox_to_anchor=(0.5, 0.995))
fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.20)
out = os.path.join(REPO, "paper/figures/breadth_bars_layered_row_fixed.pdf")
fig.savefig(out, facecolor="white")
fig.savefig(os.path.join(HERE, "breadth_bars_layered_row_fixed.png"), facecolor="white")
print("saved", out)
