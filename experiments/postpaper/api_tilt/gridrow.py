#!/usr/bin/env python3
"""Grid table: one row per (behaviour, model, arm) with quality AND cost side by side.

Quality columns are the usual ones -- presence on 0-100 (judge score x10), arithmetic and
geometric mean target token probability, min-of-mins, and the count of emitted tokens under
tau. The band is anchored on the CELL's own sampled vanilla run (anchor - 3pp); a cell with
no vanilla run has no anchor, and prints "-" rather than borrowing another cell's floor.

Cost columns come from api_cost.jsonl: wall-clock, API calls, and prompt tokens. Prompt
tokens are what the provider actually bills, since every call re-sends the whole context.
tok/call = generated tokens per API call, the headline efficiency number for the two
decoders (single-position overlap sits near 0.5, the speculative decoder well above 10).

  python -X utf8 experiments/postpaper/api_tilt/gridrow.py <beh> [<beh> ...]
"""
import glob
import json
import math
import os
import statistics as st
import sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs_dsv4")
TAU = 1e-04          # percent
MODELS = [("dsv4", "deepseek_v4_flash"), ("glm", "glm_5p3_flash")]


def quality(d):
    sc, means, gm_num, gm_den, mins, nsub = [], [], 0.0, 0, [], 0
    for tf in sorted(glob.glob(os.path.join(d, "round_1", "transcripts", "*.json"))):
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


def cost(d):
    recs = []
    for f in glob.glob(os.path.join(d, "**", "api_cost.jsonl"), recursive=True):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    if not recs:
        return None
    calls = max(r.get("calls", 0) for r in recs)
    gen = sum(r.get("gen_tokens", 0) for r in recs)
    return {"secs": sum(r.get("secs", 0.0) for r in recs), "calls": calls,
            "ptok": max(r.get("prompt_tok", 0) for r in recs),
            "per_call": gen / calls if calls else float("nan")}


def arms(cell):
    out = []
    for d in sorted(glob.glob(os.path.join(cell, "api_*"))):
        out.append((os.path.basename(d).replace("api_", "").replace("_15s", ""), d))
    return out


ROW = "%-26s %3s %9s %8s %8s %7s %9s %8s %11s %9s"
HDR = ROW % ("arm", "n", "presence", "arith%", "geom%", "band", "min", "secs",
             "prompt_tok", "tok/call")


def fmt(name, q, c, band, partial):
    return (ROW % (name[:26], q["n"], "%.1f" % q["pres"], "%.2f" % q["arith"],
                   "%.2f" % q["geom"], band, "%.1e" % q["min"],
                   "%.0f" % c["secs"] if c["secs"] == c["secs"] else "-",
                   "%d" % c["ptok"] if c["ptok"] else "-",
                   "%.2f" % c["per_call"] if c["per_call"] == c["per_call"] else "-")
            + partial)


# One table per (behaviour, model) cell: the arms within a cell share a scenario bank, a
# seed and an anchor, so they are the only things directly comparable to each other.
# Cross-cell reading is what the per-cell band column is for.
for beh in sys.argv[1:]:
    for tag, mdir in MODELS:
        cell = os.path.join(BASE, beh, mdir)
        if not os.path.isdir(cell):
            continue
        found = [(n, d) for n, d in arms(cell) if n != "elicited"]
        anchor = None
        for name, d in found:
            if name == "vanilla":
                q = quality(d)
                if q:
                    anchor = q["arith"]
        rows, ref = [], []
        for name, d in found:
            q = quality(d)
            if not q:
                continue
            c = cost(d) or {"secs": float("nan"), "ptok": 0, "per_call": float("nan")}
            partial = ("" if os.path.isfile(os.path.join(d, "round_1", "judgment.json"))
                       else "  *PARTIAL")
            if name == "vanilla":
                ref.append(fmt("vanilla  (anchor)", q, c, "-", partial))
                continue
            if anchor is None:
                band = "-"
            else:
                miss = q["arith"] - (anchor - 3.0)
                band = "yes" if miss >= 0 else "%+.1f" % miss
            rows.append((q["pres"], fmt(name, q, c, band, partial)))
        if not rows and not ref:
            continue
        title = "%s / %s" % (beh, tag)
        if anchor is not None:
            title += "   anchor %.2f%%  floor %.2f%%" % (anchor, anchor - 3.0)
        else:
            title += "   (no vanilla anchor yet)"
        print(title)
        print(HDR)
        print("-" * len(HDR))
        for _, line in sorted(rows, key=lambda x: -x[0]):
            print(line)
        for line in ref:
            print(line)
        print()
