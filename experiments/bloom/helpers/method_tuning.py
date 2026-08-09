#!/usr/bin/env python3
"""Reusable post-hoc method tuning: for a run folder, pick the operating point and
report (presence, arith-mean prob, geo-mean prob, min prob) over the SELECTED transcripts.
  - op_point : best-of-N (per scenario keep the max-score round).           [search / BoN]
  - best_in_band : weighted-round selection frontier, take the highest-score
                   point with mean prob >= x_bon (else closest to band).    [steering methods]
Metrics use the transcript's pooled prob_stats (all target turns) = the same
"prob" the appendix Pareto figures use. Run: python -X utf8 method_tuning.py <cell_dir>
"""
import os, sys, glob, json, math, statistics as st
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
POOL = ThreadPoolExecutor(max_workers=24)

def _read(args):
    tf, score = args
    try:
        d = json.load(open(tf, encoding="utf-8")); v = d.get("metadata", {}).get("variation_number")
        if v is None or v not in score: return None
        ps = d.get("prob_stats")
        if not ps or not ps.get("n"):
            tm = [m["prob_stats"] for m in d["messages"] if m.get("source") == "target" and m.get("prob_stats")]
            if not tm: return None
            n = sum(p["n"] for p in tm)
            ps = {"n": n,
                  "mean": sum(p["n"] * p["mean"] for p in tm) / n,
                  "geomean": math.exp(sum(p["n"] * math.log(p["geomean"]) for p in tm if p["geomean"] > 0) / n),
                  "min": min(p["min"] for p in tm)}
        return {"scenario": v, "score": score[v], "mean": ps["mean"], "geo": ps["geomean"], "minp": ps["min"], "n": ps["n"]}
    except Exception:
        return None

def load_points(run_dir):
    tasks = []
    for jp in sorted(glob.glob(os.path.join(run_dir, "round_*", "judgment.json"))):
        j = json.load(open(jp, encoding="utf-8"))
        score = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", [])
                 if e.get("variation_number") is not None and e.get("behavior_presence") is not None}
        for tf in glob.glob(os.path.join(os.path.dirname(jp), "transcripts", "*.json")):
            tasks.append((tf, score))
    return [r for r in POOL.map(_read, tasks) if r]

def _norm(vals):
    lo, hi = min(vals), max(vals)
    return [0.5] * len(vals) if hi == lo else [(v - lo) / (hi - lo) for v in vals]

def aggregate(chosen):
    # unweighted scenario means (matches the appendix Pareto figures' "prob" axis);
    # token min = the single least-probable token across ALL chosen transcripts (global min).
    return {"presence": st.mean(c["score"] for c in chosen),
            "arith": st.mean(c["mean"] for c in chosen),
            "geo": st.mean(c["geo"] for c in chosen),
            "min": min(c["minp"] for c in chosen)}

def op_point(pts):
    by = defaultdict(list)
    for p in pts: by[p["scenario"]].append(p)
    return aggregate([max(v, key=lambda p: p["score"]) for v in by.values()])

def _frontier(pts):
    by = defaultdict(list)
    for p in pts: by[p["scenario"]].append(p)
    pn = _norm([p["mean"] for p in pts]); sn = _norm([p["score"] for p in pts])
    for p, a, b in zip(pts, pn, sn): p["_pn"], p["_sn"] = a, b
    out = []
    for i in range(21):
        w = i / 20.0
        chosen = [max(v, key=lambda p: w * p["_sn"] + (1 - w) * p["_pn"]) for v in by.values()]
        out.append((st.mean(c["mean"] for c in chosen), st.mean(c["score"] for c in chosen), chosen))
    return out

def best_in_band(pts, x_bon):
    fr = _frontier(pts)
    band = [f for f in fr if f[0] >= x_bon]
    pick = max(band, key=lambda f: f[1]) if band else max(fr, key=lambda f: (f[0], f[1]))
    return aggregate(pick[2])

def _rel(cell, pattern):
    for p in sorted(glob.glob(os.path.join(cell, pattern))):
        if os.path.isdir(p): return os.path.relpath(p, cell).replace("\\", "/")
    return None

def _tokbias(cell):                       # nested at tokbias(_tuned)/lam_*/round_*
    for base in ("tokbias_tuned", "tokbias"):
        for l in sorted(glob.glob(os.path.join(cell, base, "lam_*"))):
            if os.path.isdir(l) and glob.glob(os.path.join(l, "round_*", "judgment.json")):
                return os.path.relpath(l, cell).replace("\\", "/")
    return None

def tune_cell(cell):
    """Return {method: metrics} for the standard method set of one cell folder."""
    jail = _rel(cell, "jail_b*")
    combo = _rel(cell, "combo/beta_*")
    tb = _tokbias(cell)
    # BoN operating point = its frontier's max-score point (matches the Pareto-figure BoN dot);
    # this also fixes x_bon (flat-top tie-break).
    bon_fr = _frontier(load_points(os.path.join(cell, "bon")))
    bon_pick = max(bon_fr, key=lambda f: f[1])
    x_bon = bon_pick[0]
    res = {"Vanilla": aggregate(bon_pick[2])}
    for name, sub, mode in [("BEAST-in", "input_search_3turn", "op"), ("FLRT", "flrt", "op"),
                            ("G-PAIR", "gpair_t3_sfull", "band"), ("BEAST-out", "output_search_3turn", "op"),
                            ("TokenBias", tb, "op"),
                            ("LogitTilt", jail, "band"), ("WILT", combo, "band")]:
        fol = os.path.join(cell, sub) if sub else None
        if not fol or not os.path.isdir(fol): res[name] = None; continue
        pts = load_points(fol)
        res[name] = (op_point(pts) if mode == "op" else best_in_band(pts, x_bon)) if pts else None
    return res, x_bon

if __name__ == "__main__":
    cell = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/bloom/runs_final/self_harm/Qwen_Qwen3.5-4B"
    res, x_bon = tune_cell(cell)
    print(f"cell: {cell}\nx_bon (BoN plaus) = {x_bon:.2f}\n")
    print(f"{'method':<11} {'presence(0-100)':>15} {'arith%':>8} {'geo%':>8} {'tokmin%':>12}")
    for m, r in res.items():
        if not r: print(f"{m:<11}   MISSING"); continue
        print(f"{m:<11} {r['presence']*10:>15.1f} {r['arith']:>8.2f} {r['geo']:>8.2f} {r['min']:>12.1e}")
    POOL.shutdown(wait=False)
