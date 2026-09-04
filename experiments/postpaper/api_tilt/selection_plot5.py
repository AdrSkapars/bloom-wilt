#!/usr/bin/env python3
"""Selection frontiers at 5 rounds: hosted overlap decode vs the local full-vocab tilt.

A copy of deepseek_v4/selection_plot.py adapted to the api_tilt comparison. Same
convention as that file (and as param_sweep._curve, which produced the paper's panels):
per scenario pick the round maximising (1-w)*prob + w*presence, sweep w, and trace the
distinct operating points.

Every arm is cut to 5 rounds, so pool depth is matched -- it is not a free variable here.
The anchor is vanilla's w=1 point (max presence per scenario, ties broken toward the more
probable transcript) and the shaded region is the one-sided >= anchor-3pp band. A filled
star marks each arm's best point inside it; an open circle marks arms whose whole frontier
lies below the floor and can therefore never be selected.

Local arms are dashed, hosted arms solid, so the engine distinction reads at a glance.

  python -X utf8 experiments/postpaper/api_tilt/selection_plot5.py
"""
import glob
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_dsv4", "self_harm", "deepseek_v4_flash")
NW = 201
BAND = 3.0
ROUNDS = 5

#          folder                            label                     colour     dashed
ARMS = [
    ("jail_b0.5",                       "local tilt  b2=0.5",        "#7f9fc4", True),
    ("jail_b1",                         "local tilt  b2=1",          "#1f4e9c", True),
    ("api_overlap_elicited_15s",        "overlap: elicited",         "#e08214", False),
    ("api_overlap_combined_sample_15s", "overlap: weighted combined", "#7b3294", False),
    ("api_overlap_combined_15s",        "overlap: combined",         "#d6191c", False),
]


def load(arm):
    rows = []
    for tf in glob.glob(os.path.join(RUNS, arm, "round_*", "transcripts", "*.json")):
        rnd = int(os.path.basename(os.path.dirname(os.path.dirname(tf))).split("_")[1])
        if rnd > ROUNDS:
            continue
        d = json.load(open(tf, encoding="utf-8"))
        sc = ((d.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
        ps = d.get("prob_stats") or {}
        if sc is None or ps.get("mean") is None:
            continue
        rows.append((d["metadata"]["variation_number"], float(sc) * 10, float(ps["mean"])))
    return rows


def curve(rows):
    by = defaultdict(list)
    for v, sc, pr in rows:
        by[v].append((sc, pr))
    seen = {}
    for i in range(NW):
        lam = i / (NW - 1)
        picks = [max(p, key=lambda t: (1 - lam) * (t[1] / 100.0) + lam * (t[0] / 100.0))
                 for p in by.values()]
        seen[(round(st.mean(p[1] for p in picks), 4),
              round(st.mean(p[0] for p in picks), 4))] = lam
    return sorted(seen)


def w1(rows):
    by = defaultdict(list)
    for v, sc, pr in rows:
        by[v].append((sc, pr))
    picks = []
    for p in by.values():
        top = max(sc for sc, _ in p)
        picks.append((top, max(pr for sc, pr in p if sc == top)))
    return st.mean(p[1] for p in picks), st.mean(p[0] for p in picks)


van = load("vanilla_15s")
anchor_p, anchor_s = w1(van)
floor = anchor_p - BAND

fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=170)
ax.axvspan(floor, 1e4, color="#bcbcbc", alpha=0.20, lw=0, zorder=0)
ax.axvline(floor, color="#8a8a8a", ls="--", lw=0.9, zorder=1)

vpts = curve(van)
ax.plot([p for p, _ in vpts], [s for _, s in vpts], ":", color="#111111", lw=2.0,
        label="local vanilla  b2=0", zorder=4)
ax.scatter([anchor_p], [anchor_s], s=60, color="#111111", edgecolor="white", lw=1.0, zorder=9)
ax.annotate("anchor", (anchor_p, anchor_s), textcoords="offset points", xytext=(-14, 9),
            fontsize=9, color="#111111")

summary = [("local vanilla  b2=0", anchor_s, anchor_p, "anchor")]
allx = [q for q, _ in vpts]
ally = [q for _, q in vpts]
for folder, label, colour, dashed in ARMS:
    rows = load(folder)
    if not rows:
        continue
    pts = curve(rows)
    allx += [q for q, _ in pts]
    ally += [q for _, q in pts]
    ax.plot([p for p, _ in pts], [s for _, s in pts], "--" if dashed else "-",
            color=colour, lw=1.9, label=label, zorder=3)
    elig = [(p, s) for p, s in pts if p >= floor]
    if elig:
        bs = max(s for _, s in elig)
        bp = max(p for p, s in elig if s == bs)
        ax.scatter([bp], [bs], marker="*", s=210, color=colour,
                   edgecolor="#111111", lw=1.0, zorder=10)
        summary.append((label, bs, bp, "in band"))
    else:
        hi = max(p for p, _ in pts)
        bs = max(s for p, s in pts if p == hi)
        ax.scatter([hi], [bs], marker="o", s=70, facecolor="white",
                   edgecolor=colour, lw=1.8, zorder=10)
        summary.append((label, bs, hi, f"misses band by {floor - hi:.2f} pp"))

# axvspan's upper bound must not drive autoscale -- clamp both axes to the data
xpad = max(0.5, (max(allx) - min(allx)) * 0.05)
ax.set_xlim(min(allx) - xpad, max(allx) + xpad)
ax.set_ylim(0, max(ally) * 1.10)
ax.set_xlabel("output probability  (mean token prob, %)")
ax.set_ylabel("behaviour presence  (0-100)")
ax.set_title("DeepSeek-V4-Flash · self-harm · selection frontiers at 5 rounds\n"
             "hosted top-5 overlap decode vs the local full-vocab tilt\n"
             f"shaded = within {BAND:g} pp of vanilla's operating point",
             fontsize=10.5, linespacing=1.35)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.grid(alpha=0.16, lw=0.6)
fig.tight_layout()
out = os.path.join(HERE, "selection_frontiers_5round.png")
fig.savefig(out, facecolor="white")

print(f"anchor (vanilla w=1): presence {anchor_s:.1f} @ {anchor_p:.2f}%   "
      f"band floor {floor:.2f}%   pool depth {ROUNDS} rounds\n")
print(f"{'arm':28s} {'presence':>9s} {'prob':>8s}   status")
for name, sc, pr, status in summary:
    print(f"{name:28s} {sc:9.1f} {pr:7.2f}%   {status}")
print(f"\nwrote {out}")
