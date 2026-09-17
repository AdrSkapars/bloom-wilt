#!/usr/bin/env python3
"""Top-k sweep table: behaviour/plausibility per run, alongside the per-token truncation stats.

The right-hand columns are the reason this sweep runs locally. They come from
partial_stats.jsonl, are measured per TOKEN, and so carry error bars small enough to separate
two k values in a single 15-scenario cell:

  support   mean number of candidates surviving truncation. Under rule=poe this is the size of
            the two top-k sets' INTERSECTION, so it is bounded by k and usually well below it.
  empty     share of positions where nothing survived and the engine fell back to the target's
            own top-1. Pure cost of truncation; at k=full it must be 0.
  agree     share of positions where the truncated decode's argmax matched what the SAME rule
            would have picked with no truncation. This is the headline: 100% means top-k lost
            nothing at that position, and it is computed from the same forward pass, so it is
            free and noise-free in a way presence is not.
  mass      mean untruncated probability mass sitting on the surviving support.

The left-hand columns are the usual ones, so a k that looks harmless on agree but moves
presence is still visible.

  python -X utf8 experiments/postpaper/partial_tilt/ksweep.py <beh> <model_dir>
"""
import glob
import json
import math
import os
import statistics as st
import sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs_local")
TAU = 1e-04          # percent


def quality(d):
    sc, means, gm_num, gm_den, mins, nsub = [], [], 0.0, 0, [], 0
    for tf in sorted(glob.glob(os.path.join(d, "round_*", "transcripts", "*.json"))):
        t = json.load(open(tf, encoding="utf-8"))
        s = ((t.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
        ps = t.get("prob_stats") or {}
        if s is None or ps.get("mean") is None:
            continue
        sc.append(float(s) * 10)
        means.append(float(ps["mean"]))
        mins.append(float(ps["min"]))
        if ps.get("geomean"):
            gm_num += ps["n"] * math.log(ps["geomean"]); gm_den += ps["n"]
        for m in t["messages"]:
            for p in (m.get("gen_token_probs") or []):
                if p is not None and p == p and p < TAU:
                    nsub += 1
    if not sc:
        return None
    return {"n": len(sc), "pres": st.mean(sc), "arith": st.mean(means),
            "geom": math.exp(gm_num / gm_den) if gm_den else float("nan"),
            "min": min(mins), "nsub": nsub}


def trunc_stats(d):
    """Weighted by positions, not a mean of means: batches differ in length."""
    f = os.path.join(d, "partial_stats.jsonl")
    if not os.path.isfile(f):
        return None
    tot = 0
    acc = {"mean_support": 0.0, "empty_rate": 0.0, "argmax_agree": 0.0, "mean_mass_kept": 0.0}
    k = None
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        n = int(r.get("positions", 0) or 0)
        if not n:
            continue
        k = r.get("top_k", k)
        tot += n
        for key in acc:
            acc[key] += float(r.get(key, 0.0) or 0.0) * n
    if not tot:
        return None
    out = {key: acc[key] / tot for key in acc}
    out["positions"] = tot
    out["top_k"] = k
    return out


def sort_key(name):
    """k=full sorts last: it is the anchor the truncated points are measured against."""
    if "_kfull" in name:
        return (1, 0)
    for part in name.split("_"):
        if part.startswith("k") and part[1:].isdigit():
            return (0, int(part[1:]))
    return (0, 10 ** 9)


HDR = ("%-30s %3s %9s %8s %8s %9s %9s %8s %8s %8s %9s"
       % ("run", "n", "presence", "arith%", "geom%", "min", "support", "empty", "agree", "mass",
          "positions"))
if __name__ == "__main__":
    beh = sys.argv[1] if len(sys.argv) > 1 else "self_harm"
    mdir = sys.argv[2] if len(sys.argv) > 2 else "Qwen_Qwen3.5-4B"
    cell = os.path.join(BASE, beh, mdir)
    runs = sorted(glob.glob(os.path.join(cell, "*")), key=lambda p: sort_key(os.path.basename(p)))
    print("%s / %s" % (beh, mdir))
    print(HDR)
    print("-" * len(HDR))
    for d in runs:
        if not os.path.isdir(d):
            continue
        q = quality(d)
        if not q:
            continue
        s = trunc_stats(d)
        name = os.path.basename(d).replace("ptilt_", "").replace("_15s", "")
        if s:
            print("%-30s %3d %9.1f %8.2f %8.2f %9.1e %9.2f %7.2f%% %7.2f%% %7.2f%% %9d"
                  % (name[:30], q["n"], q["pres"], q["arith"], q["geom"], q["min"],
                     s["mean_support"], 100 * s["empty_rate"], 100 * s["argmax_agree"],
                     100 * s["mean_mass_kept"], s["positions"]))
        else:
            print("%-30s %3d %9.1f %8.2f %8.2f %9.1e %9s"
                  % (name[:30], q["n"], q["pres"], q["arith"], q["geom"], q["min"],
                     "(no stats)"))
