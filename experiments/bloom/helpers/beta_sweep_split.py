#!/usr/bin/env python3
"""Appendix beta-sweep grid, restyled to match the main pareto-sweep figure: viridis (colour-blind
safe) curves + a shared colourbar keying beta, a neutral grey >=(BoN-3%) selection band, BoN dotted
+ operating dot, and a dot-sized star at the tuned beta (with a small beta label). Layout unchanged:
4 model COLUMNS x behaviour ROWS, split into A (behaviours 1-3) and B (4-8). Straight from each
cell's param_selection.json. Run on the box: UV_NO_SYNC=1 uv run --no-sync python -X utf8 beta_sweep_split.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

REPO = "/workspace/inversion_optimisation"
RN = os.path.join(REPO, "experiments/bloom/runs_new")
FIGDIR = os.path.join(REPO, "paper/figures")
# order MUST match the behaviour list in the paper's experimental-setup section (reward hacking dropped)
BEHS = [("Racial", "racial"), ("Political", "political"), ("Delusions", "delusions"),
        ("Self-harm", "self_harm"), ("Medical", "medical"), ("Deception", "deception"),
        ("Self-pres", "selfpres"), ("Goblin", "goblin")]
MODELS = [("Qwen3.5-4B", "Qwen_Qwen3.5-4B"), ("Gemma-4-E4B", "google_gemma-4-e4b-it"),
          ("Llama-3.2-3B", "meta-llama_Llama-3.2-3B-Instruct"), ("Phi-4-mini", "microsoft_Phi-4-mini-instruct")]
STD = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
BLO, BHI = 0.5, 4.0
CMAP = plt.cm.viridis
TRUNC = LinearSegmentedColormap.from_list("vtrunc", CMAP(np.linspace(0.06, 0.86, 256)))


def col(b):
    t = max(0.0, min(1.0, (b - BLO) / (BHI - BLO)))
    return TRUNC(t)


def load(bdir, mdir):
    return json.load(open(os.path.join(RN, bdir, mdir, "param_selection.json"), encoding="utf-8"))


def panel(ax, d):
    fr = d["frontier"]["arith"]; anc = d["anchor"]; xs = []
    xb = anc["arith"]
    ax.axvspan(xb - 3, 1e4, color="#bcbcbc", alpha=0.22, lw=0, zorder=0)   # +-3% one-sided selection band
    for b in STD:
        k = f"{b:g}"
        if k not in fr:
            continue
        cur = fr[k]; xs += [p[0] for p in cur]
        ax.plot([p[0] for p in cur], [p[1] * 10 for p in cur], "-", color=col(b), lw=1.5, zorder=3)
    if "0" in fr:
        cur = fr["0"]; xs += [p[0] for p in cur]
        ax.plot([p[0] for p in cur], [p[1] * 10 for p in cur], ":", color="#111111", lw=1.8, zorder=4)
    ax.scatter([xb], [anc["score"] * 10], s=42, color="#111111", edgecolor="white", lw=0.8, zorder=6)
    pk = d.get("picks", {}).get("arith", {}).get("pm3")
    if pk and pk.get("beta") is not None:
        cb, cx, cy = pk["beta"], pk["plaus"], pk["score"] * 10; xs.append(cx)
        ax.scatter([cx], [cy], marker="*", s=62, color=col(cb), edgecolor="#111111", lw=1.0, zorder=10)
        ax.annotate(f"β{cb:g}", (cx, cy), textcoords="offset points", xytext=(6, 3),
                    fontsize=9.5, fontweight="bold", zorder=11)
    if xs:
        ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(0, 113)                                   # headroom for the star's beta label...
    ax.set_yticks([0, 20, 40, 60, 80, 100])              # ...but no tick label above 100
    ax.grid(True, color="#ececec", lw=0.4, zorder=1)
    ax.tick_params(labelsize=10)


def make(behs, fname):
    nR = len(behs)
    H = 3.05 * nR
    fig, axes = plt.subplots(nR, 4, figsize=(15, H), dpi=140, squeeze=False)
    for r, (blab, bdir) in enumerate(behs):
        for c, (mlab, mdir) in enumerate(MODELS):
            ax = axes[r][c]
            try:
                panel(ax, load(bdir, mdir))
            except Exception as e:
                ax.text(0.5, 0.5, "n/a", ha="center"); print("skip", blab, mlab, repr(e))
            if r == 0:
                ax.set_title(mlab, fontsize=16, pad=6)
            if c == 0:
                ax.set_ylabel(f"{blab}\nbehaviour score (%)", fontsize=13)
            if r == nR - 1:
                ax.set_xlabel("output probability (%)", fontsize=13)
    # panels use nearly the full width; the colourbar + legend live in a top strip
    fig.subplots_adjust(top=1 - 1.55 / H, bottom=0.045, left=0.055, right=0.988,
                        hspace=0.30, wspace=0.18)
    # --- top strip: horizontal colourbar (centre-left) + legend (right) ---
    cy = 1 - 0.62 / H
    sm = ScalarMappable(norm=Normalize(BLO, BHI), cmap=TRUNC)
    cax = fig.add_axes([0.20, cy, 0.46, 0.13 / H])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("LogitTilt β", fontsize=14, labelpad=5)
    cbar.ax.tick_params(labelsize=11)
    handles = [Line2D([0], [0], color="#111111", ls=":", lw=2.2, label="BoN"),
               Line2D([0], [0], marker="*", ls="None", color="#888888", mec="#111111", ms=13,
                      label="Tuned β")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.72, cy + 0.06 / H),
               ncol=2, frameon=False, fontsize=13, columnspacing=1.6, handletextpad=0.5)
    fig.savefig(os.path.join(FIGDIR, fname + ".pdf"), facecolor="white")
    fig.savefig(os.path.join(FIGDIR, fname + ".png"), facecolor="white")
    print("saved", fname, flush=True)


make(BEHS[:3], "beta_sweep_A")
make(BEHS[3:], "beta_sweep_B")
