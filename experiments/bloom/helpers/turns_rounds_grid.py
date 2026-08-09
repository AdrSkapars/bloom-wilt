#!/usr/bin/env python3
"""Second scaling figure, from the TURNS runs only (each has 3 rounds). One line per
round budget (best-of 1/2/3 rounds), a point per turn count (1..5), in the
(output-probability, behaviour-presence) plane. Rightward along a line = more turns;
higher line = more rounds. Run: python -X utf8 turns_rounds_grid.py
"""
import os, sys, glob, json, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import method_tuning as M
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc

REPO = "/workspace/inversion_optimisation"
HERE = os.path.dirname(__file__)
CELL = os.path.join(REPO, "experiments/bloom/runs_final/self_harm/Qwen_Qwen3.5-4B")
WA = os.path.join(CELL, "_wilt_analysis")
X_BON = 51.1

def rd(a):
    tf, sc, rnd = a; r = M._read((tf, sc))
    if r: r["round"] = rnd
    return r
def load_fast(folder):
    tasks = []
    for jp in sorted(glob.glob(os.path.join(folder, "round_*", "judgment.json"))):
        rnd = int(os.path.basename(os.path.dirname(jp)).split("_")[1])
        j = json.load(open(jp, encoding="utf-8"))
        sc = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", [])
              if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        for tf in glob.glob(os.path.join(os.path.dirname(jp), "transcripts", "*.json")):
            tasks.append((tf, sc, rnd))
    return [r for r in M.POOL.map(rd, tasks) if r]
def bestof(pts):
    by = {}
    for p in pts: by.setdefault(p["scenario"], []).append(p)
    ch = [max(v, key=lambda p: p["score"]) for v in by.values()]
    probs = [c["mean"] for c in ch]; press = [c["score"] * 10 for c in ch]
    n = len(ch); sem = lambda a: st.stdev(a) / (n ** 0.5)
    return st.mean(probs), st.mean(press), sem(probs), sem(press)   # (prob, presence, prob_sem, pres_sem)

TURNS = range(1, 6)
ROUND_KS = [1, 2, 3]   # extend (e.g. [1,2,3,4,5]) once the turn sweeps are rerun with more rounds
grid = {n: load_fast(os.path.join(WA, f"turns_t{n}_r3")) for n in TURNS}
lines = {}   # k rounds -> list of (prob, presence) over turns
for k in ROUND_KS:
    lines[k] = [bestof([x for x in grid[n] if x["round"] <= k]) for n in TURNS]

print("rounds\\turns  " + "  ".join(f"t{n}" for n in TURNS))
for k in ROUND_KS:
    print(f"  r{k}: " + "  ".join(f"({p:.1f},{s:.1f}±{ss:.1f})" for p, s, _, ss in lines[k]))

plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=160)
allp = [t[0] for k in lines for t in lines[k]]
ylo = min(t[1] - t[3] for k in lines for t in lines[k]) - 3      # room for the lowest error bar
xlo, xhi = min(allp) - 1.2, max(allp) + 1.6
ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, 103)
ax.axvspan(X_BON, xhi, color="#bcbcbc", alpha=0.22, lw=0, zorder=0)   # grey >=BoN band
ax.axvline(X_BON, color="#8a8a8a", lw=1.0, ls=":", zorder=1)

def col(k):
    t = (k - ROUND_KS[0]) / max(1, ROUND_KS[-1] - ROUND_KS[0])
    return mc.to_hex(plt.cm.Blues(0.55 + 0.40 * t))   # darker range so the lightest line stays clear on grey
for k in ROUND_KS:
    p = [t[0] for t in lines[k]]; s = [t[1] for t in lines[k]]
    psem = [t[2] for t in lines[k]]; ssem = [t[3] for t in lines[k]]                 # x/y SEM
    ax.plot(p, s, "-", color=col(k), lw=2.2, zorder=4 + k, label=f"{k} round{'s' if k>1 else ''}")
    ax.errorbar(p, s, xerr=psem, yerr=ssem, fmt="o", ms=8, color=col(k), mec="white", mew=1.0,
                ecolor=col(k), elinewidth=1.3, capsize=2.5, capthick=1.3, zorder=5 + k)
    for i, (x, y) in enumerate(zip(p, s), 1):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(-5, 4), ha="right",
                    fontsize=11, color=col(k), zorder=9)

ax.set_xlabel("Output probability (%)", fontsize=15)
ax.set_ylabel("Behaviour presence (%)", fontsize=15)
ax.grid(True, color="#e6e5de", lw=0.5, zorder=1)
ax.tick_params(labelsize=13)
_h, _l = ax.get_legend_handles_labels()
leg = ax.legend(_h[::-1], _l[::-1], fontsize=12.5, frameon=False, loc="lower right", title="Num. rounds")
leg.get_title().set_fontsize(12.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "turns_rounds_grid.png"), facecolor="white")
fig.savefig(os.path.join(REPO, "paper/figures/turns_rounds_grid.pdf"), facecolor="white")
print("saved PNG + paper/figures/turns_rounds_grid.pdf")
M.POOL.shutdown(wait=False)
