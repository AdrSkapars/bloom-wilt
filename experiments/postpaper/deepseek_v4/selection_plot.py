#!/usr/bin/env python3
"""Selection frontiers for the DeepSeek-V4 arms, styled like the paper's beta-sweep panels.

Uses param_sweep._curve's convention (arith/100, score/10) rather than the per-pool min-max
the paper text describes, because _curve is what actually produced the paper's figures --
so these curves are comparable to them.

Anchor = vanilla's w=1 operating point (max presence per scenario, ties broken toward the
more probable transcript). The shaded region is the one-sided >= anchor-3pp selection band;
a star marks each steered arm's best point inside it.

  python experiments/postpaper/deepseek_v4/selection_plot.py
"""
import glob
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_dsv4", "self_harm", "deepseek_v4_flash")
NW = 201
BAND = 3.0


def load(arm):
    """Per-transcript (scenario, presence, arithmetic-mean token prob) over every round."""
    rows = []
    for tf in glob.glob(os.path.join(RUNS, arm, "round_*", "transcripts", "*.json")):
        d = json.load(open(tf, encoding="utf-8"))
        sc = ((d.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
        ps = d.get("prob_stats") or {}
        if sc is None or ps.get("mean") is None:
            continue
        rows.append((d["metadata"]["variation_number"], float(sc), float(ps["mean"])))
    return rows


def pools(rows):
    by = defaultdict(list)
    for var, sc, pr in rows:
        by[var].append((sc, pr))
    return by


def curve(rows):
    """(prob, presence) per selection weight; dedup preserves the distinct operating points."""
    by = pools(rows)
    seen = {}
    for i in range(NW):
        lam = i / (NW - 1)
        picks = [max(p, key=lambda t: (1 - lam) * (t[1] / 100.0) + lam * (t[0] / 10.0))
                 for p in by.values()]
        pr = st.mean(p[1] for p in picks)
        sc = st.mean(p[0] for p in picks) * 10
        seen[(round(pr, 4), round(sc, 4))] = lam
    return sorted(seen)


def w1(pts):
    """Max presence; ties broken toward the higher probability."""
    top = max(sc for _, sc in pts)
    return max(pr for pr, sc in pts if sc == top), top


ARMS = [("jail_b0.5", "β=0.5", "#1f78b4"), ("jail_b1", "β=1", "#e31a1c")]

van = curve(load("vanilla_15s"))
anchor_p, anchor_s = w1(van)
floor = anchor_p - BAND

fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=160)
ax.axvspan(floor, 1e4, color="#bcbcbc", alpha=0.22, lw=0, zorder=0)
ax.axvline(floor, color="#8a8a8a", ls="--", lw=0.9, zorder=1)

ax.plot([p for p, _ in van], [s for _, s in van], ":", color="#111111", lw=2.0,
        label="vanilla (best-of-N)", zorder=4)
ax.scatter([anchor_p], [anchor_s], s=55, color="#111111", edgecolor="white", lw=0.9, zorder=6)

def nrounds(arm):
    return len({os.path.basename(os.path.dirname(os.path.dirname(t)))
                for t in glob.glob(os.path.join(RUNS, arm, "round_*", "transcripts", "*.json"))})

summary = [("vanilla", nrounds("vanilla_15s"), anchor_s, anchor_p, "anchor")]
for arm, label, colour in ARMS:
    rows = load(arm)
    if not rows:
        continue
    pts = curve(rows)
    nr = nrounds(arm)
    ax.plot([p for p, _ in pts], [s for _, s in pts], "-", color=colour, lw=1.8,
            label=f"{label} ({nr} rounds)", zorder=3)
    elig = [(p, s) for p, s in pts if p >= floor]
    if elig:
        best_s = max(s for _, s in elig)
        best_p = max(p for p, s in elig if s == best_s)
        ax.scatter([best_p], [best_s], marker="*", s=170, color=colour,
                   edgecolor="#111111", lw=1.0, zorder=10)
        ax.annotate(label, (best_p, best_s), textcoords="offset points", xytext=(7, 4),
                    fontsize=10, fontweight="bold", color=colour, zorder=11)
        summary.append((label, nr, best_s, best_p, "in band"))
    else:
        hi = max(p for p, _ in pts)
        summary.append((label, nr, max(s for _, s in pts), hi, f"misses by {floor - hi:.2f} pp"))

# axvspan's upper bound must not drive autoscale -- clamp both axes to the data
allx = [q for q, _ in van] + [q for a, _, _ in ARMS for q, _ in (curve(load(a)) if load(a) else [])]
ally = [q for _, q in van] + [q for a, _, _ in ARMS for _, q in (curve(load(a)) if load(a) else [])]
xpad = max(0.4, (max(allx) - min(allx)) * 0.06)
ax.set_xlim(min(allx) - xpad, max(allx) + xpad)
ax.set_ylim(0, max(ally) * 1.12)

ax.set_xlabel("output probability (mean token prob, %)")
ax.set_ylabel("behaviour presence (0-100)")
ax.set_title("DeepSeek-V4-Flash · self-harm · selection frontiers\n"
             f"shaded = within {BAND:g} pp of vanilla's operating point", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.grid(alpha=0.18, lw=0.6)
fig.tight_layout()
out = os.path.join(HERE, "selection_frontiers.png")
fig.savefig(out, facecolor="white")

print(f"anchor (vanilla w=1): presence {anchor_s:.1f} @ {anchor_p:.2f}%   band floor {floor:.2f}%\n")
print(f"{'arm':10s} {'rounds':>6s} {'presence':>9s} {'prob':>8s}   status")
for name, nr, sc, pr, status in summary:
    print(f"{name:10s} {nr:6d} {sc:9.1f} {pr:7.2f}%   {status}")
print(f"\nwrote {out}")
