#!/usr/bin/env python3
"""Two title-less, colour-blind-friendly paper subfigures: per-LogitTilt-beta post-hoc Pareto
frontiers for Qwen3.5-4B and Gemma-4-E4B (self-harm, 100 scen x 5 rounds). Each panel: BoN frontier
(dotted) + operating dot, >=BoN-plausibility band (grey), a star at the TUNED beta, an "increasing
beta" arrow (label parallel to the arrow), and a slim viridis colourbar keying colour->beta.
Run on the box:  UV_NO_SYNC=1 uv run --no-sync python -X utf8 beta_family_subfigs.py
"""
import sys, os, glob
import numpy as np
REPO = "/workspace/inversion_optimisation"
sys.path.insert(0, os.path.join(REPO, "experiments/bloom/helpers"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import pareto_analysis as PA

# name, cell root, output pdf, TUNED beta, arrow direction (Qwen first)
CELLS = [
    ("qwen",  "self_harm/Qwen_Qwen3.5-4B",         "pareto_frontier_qwen.pdf",  1.5, "up"),
    ("gemma", "self_harm/google_gemma-4-e4b-it",   "pareto_frontier_gemma.pdf", 0.5, "upleft"),
]
FIGDIR = os.path.join(REPO, "paper/figures")
os.makedirs(FIGDIR, exist_ok=True)

BLO, BHI = 0.25, 2.5
CMAP = plt.cm.viridis
TRUNC = LinearSegmentedColormap.from_list("vtrunc", CMAP(np.linspace(0.06, 0.86, 256)))

# per-direction: extra left x-margin, arrow tail/head + label offset (axes fraction)
ARROW = {
    "up":     dict(margin=5.0, tail=(0.10, 0.33), head=(0.15, 0.74), loff=(0.085, 0.0)),
    "upleft": dict(margin=1.5, tail=(0.53, 0.23), head=(0.25, 0.54), loff=(0.06, 0.055)),
}

# fonts (big, so the plot can be scaled down in the paper and stay legible)
FS_LABEL, FS_TICK, FS_LEG, FS_CBAR, FS_CBTICK, FS_ARROW = 17, 14, 15, 16, 13, 16


def beta_of(folder):
    return float(os.path.basename(folder).split("_")[-1])


def col(b):
    t = max(0.0, min(1.0, (b - BLO) / (BHI - BLO)))
    return TRUNC(t)


def build(cell_root):
    ROOT = os.path.join(REPO, "experiments/bloom/runs_final", cell_root)
    WA = os.path.join(ROOT, "_wilt_analysis")
    betas = {}
    for f in sorted(glob.glob(WA + "/beta_*"), key=beta_of):
        if not os.path.exists(os.path.join(f, "round_5", "judgment.json")):
            continue
        betas[beta_of(f)] = PA.pareto_frontier(PA.extract(f))
    bon = PA.pareto_frontier(PA.extract(os.path.join(ROOT, "bon")))
    return betas, bon


def add_arrow(ax, direction):
    a = ARROW[direction]
    tail, head = a["tail"], a["head"]
    ax.annotate("", xy=head, xytext=tail, xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2.6, shrinkA=0, shrinkB=0),
                zorder=11)
    # rotation PARALLEL to the arrow, computed from real display coords (post-layout), kept upright
    p0 = ax.transAxes.transform(tail)
    p1 = ax.transAxes.transform(head)
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    # keep the label upright in (-90, 90]: Qwen ~90 (vertical); Gemma up-left folds to ~ -50,
    # i.e. the text reads top-left -> bottom-right along the arrow.
    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180
    mx = (tail[0] + head[0]) / 2 + a["loff"][0]
    my = (tail[1] + head[1]) / 2 + a["loff"][1]
    ax.text(mx, my, "increasing β", transform=ax.transAxes, rotation=ang, rotation_mode="anchor",
            ha="center", va="center", fontsize=FS_ARROW, color="#333333", fontweight="bold", zorder=11)


def plot(betas, bon, out_pdf, tuned, direction):
    bon_pt = max(bon, key=lambda p: p[1])
    x_bon = bon_pt[0]
    tuned_curve = betas[tuned]
    tuned_inband = [p for p in tuned_curve if p[0] >= x_bon] or tuned_curve
    star_pt = max(tuned_inband, key=lambda p: p[1])

    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=160)
    xs_all = [p[0] for c in list(betas.values()) + [bon] for p in c]
    xlo, xhi = min(xs_all) - ARROW[direction]["margin"], max(xs_all) + 1
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0, 106)
    ax.axvspan(x_bon, xhi, color="#bcbcbc", alpha=0.22, zorder=0, lw=0)

    for b in sorted(betas):
        c = betas[b]
        ax.plot([p[0] for p in c], [p[1] * 10 for p in c], "-", color=col(b),
                lw=2.0, zorder=3, solid_capstyle="round")

    ax.plot([p[0] for p in bon], [p[1] * 10 for p in bon], ":", color="#111111", lw=2.6, zorder=4)
    ax.scatter([x_bon], [bon_pt[1] * 10], s=100, color="#111111", zorder=6,
               edgecolor="white", linewidth=1.2)
    ax.scatter([star_pt[0]], [star_pt[1] * 10], s=115, marker="*",
               color=col(tuned), edgecolor="#111111", linewidth=1.2, zorder=9)

    ax.set_xlabel("Output probability (%)", fontsize=FS_LABEL)
    ax.set_ylabel("Behaviour score (%)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, color="#e6e5de", lw=0.5, zorder=1)
    handles = [
        Line2D([0], [0], ls=":", color="#111111", lw=2.6, label="BoN"),
        Line2D([0], [0], ls="none", marker="*", color=col(tuned), mec="#111111",
               mew=1.1, ms=13, label=f"Tuned β={tuned:g}"),
    ]
    ax.legend(handles=handles, fontsize=FS_LEG, frameon=False, loc="lower left",
              handletextpad=0.5, borderaxespad=0.4)

    sm = ScalarMappable(norm=Normalize(BLO, BHI), cmap=TRUNC)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("LogitTilt β", fontsize=FS_CBAR, labelpad=14)
    cbar.ax.tick_params(labelsize=FS_CBTICK)

    fig.tight_layout()
    add_arrow(ax, direction)          # after layout -> correct on-screen angle
    PDF = os.path.join(FIGDIR, out_pdf)
    fig.savefig(PDF, facecolor="white", bbox_inches="tight")
    print("saved %s | BoN x_bon=%.1f score=%.1f | tuned b=%g -> plaus=%.1f score=%.1f"
          % (PDF, x_bon, bon_pt[1] * 10, tuned, star_pt[0], star_pt[1] * 10))


for name, root, out, tb, direction in CELLS:
    betas, bon = build(root)
    print(name, "betas:", sorted(betas), "tuned", tb)
    plot(betas, bon, out, tb, direction)
