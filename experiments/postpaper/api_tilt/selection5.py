#!/usr/bin/env python3
"""Post-run selection on the api_tilt cells, paper convention.

Per scenario, pick the transcript maximising `(1-w)*prob + w*presence`; sweep w over NW
points to trace the arm's frontier; then report the best presence whose mean token
probability still clears a one-sided band at `anchor - 3pp`.

The anchor is the b=0 TARGET-ONLY (api_vanilla_15s) operating point of the same cell, and
it is printed as a reference row. That is the whole point of the band: "how much behaviour
can you buy while keeping plausibility within 3pp of what the plain target already does".

Depth matters -- a deeper pool finds better points -- so `rounds` is printed per arm and
arms at different depths are not directly comparable.

  python -X utf8 experiments/postpaper/api_tilt/selection5.py [--rounds N] [--cell beh/model]
"""
import argparse, glob, json, os, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_dsv4")
NW = 201
BAND = 3.0

# (folder, pick rule, empty-overlap fallback). The fallback is its own column because it
# turned out to be the dominant lever on goblin despite firing on <1% of tokens.
ARMS = [
    ("api_vanilla_15s",                        "b=0 target only",  "-"),
    ("api_overlap_combined_15s",               "comb b=1",         "target"),
    ("api_overlap_combined_b2_15s",            "comb b=2",         "target"),
    ("api_overlap_elicited_15s",               "elic-pick",        "target"),
    ("api_overlap_combined_fbjail_sample_15s", "comb b=1",         "jail_sample"),
    ("api_overlap_elicited_fbjail_sample_15s", "elic-pick",        "jail_sample"),
    ("api_overlap_elicited_fbjail_argmax_15s", "elic-pick",        "jail_argmax"),
    ("api_elicited_15s",                       "b2=1 elicited only", "-"),
]


def load(cell, arm, max_rounds):
    rows = []
    for tf in glob.glob(os.path.join(RUNS, cell, arm, "round_*", "transcripts", "*.json")):
        rnd = os.path.basename(os.path.dirname(os.path.dirname(tf)))
        if int(rnd.split("_")[1]) > max_rounds:
            continue
        try:
            d = json.load(open(tf, encoding="utf-8"))
        except Exception:
            continue
        sc = ((d.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
        ps = d.get("prob_stats") or {}
        if sc is None or ps.get("mean") is None:
            continue
        rows.append((d["metadata"]["variation_number"], float(sc) * 10, float(ps["mean"]),
                     float(ps["min"]) if ps.get("min") is not None else float("nan")))
    return rows


def pools(rows):
    by = defaultdict(list)
    for v, sc, pr, mn in rows:
        by[v].append((sc, pr, mn))
    return by


def curve(rows):
    """Distinct (prob, presence, min-of-mins) points as the selection weight sweeps 0..1.

    min-of-mins is taken over the transcripts the sweep ACTUALLY SELECTS at that weight --
    the worst single token probability anywhere in the chosen set. Never a mean of mins.
    """
    by = pools(rows)
    seen = {}
    for i in range(NW):
        lam = i / (NW - 1)
        picks = [max(p, key=lambda t: (1 - lam) * (t[1] / 100.0) + lam * (t[0] / 100.0))
                 for p in by.values()]
        key = (round(st.mean(p[1] for p in picks), 4),
               round(st.mean(p[0] for p in picks), 4))
        mm = min((p[2] for p in picks if p[2] == p[2]), default=float("nan"))
        seen[key] = mm
    return sorted((p, s_, seen[(p, s_)]) for (p, s_) in seen)


def anchor(rows):
    """w=1 point: max presence per scenario, ties broken toward the more probable."""
    picks = []
    for p in pools(rows).values():
        top = max(t[0] for t in p)
        picks.append((top, max(t[1] for t in p if t[0] == top)))
    return st.mean(p[1] for p in picks), st.mean(p[0] for p in picks)


def run(cell, max_rounds):
    van = load(cell, "api_vanilla_15s", max_rounds)
    if not van:
        return
    a_pr, a_sc = anchor(van)
    floor = a_pr - BAND
    print(f"### {cell}    cap {max_rounds} round(s)")
    print(f"    anchor = b=0 target only: presence {a_sc:.1f} @ {a_pr:.2f}%   "
          f"band floor {floor:.2f}%")
    print(f"    {'arm':20s} {'fallback':12s} {'rds':>3s} {'r1':>6s} {'selected':>9s} "
          f"{'@ prob':>8s} {'min-of-mins':>12s}  status")
    for folder, label, fb in ARMS:
        rows = load(cell, folder, max_rounds)
        if not rows:
            continue
        nr = len([r for r in glob.glob(os.path.join(RUNS, cell, folder, "round_*"))
                  if int(os.path.basename(r).split("_")[1]) <= max_rounds])
        r1 = st.mean(r[1] for r in load(cell, folder, 1))
        pts = curve(rows)
        elig = [t for t in pts if t[0] >= floor]
        if elig:
            best_s = max(t[1] for t in elig)
            cands = [t for t in elig if t[1] == best_s]
            best_p, best_mm = max(cands)[0], max(cands)[2]
            status = "in band"
        else:
            hi = max(t[0] for t in pts)
            cands = [t for t in pts if t[0] == hi]
            best_p, best_s, best_mm = hi, max(t[1] for t in cands), max(cands)[2]
            status = f"misses band by {floor - hi:.2f} pp"
        mm = f"{best_mm:.3g}" if best_mm == best_mm else "--"
        print(f"    {label:20s} {fb:12s} {nr:3d} {r1:6.1f} {best_s:9.1f} {best_p:7.2f}% "
              f"{mm:>12s}  {status}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--cell", default=None)
    a = ap.parse_args()
    cells = [a.cell] if a.cell else [
        f"{b}/{m}" for b in ("self_harm", "goblin", "selfpres")
        for m in ("deepseek_v4_flash", "glm_5p3_flash", "gpt_oss_120b")]
    for c in cells:
        run(c, a.rounds)
