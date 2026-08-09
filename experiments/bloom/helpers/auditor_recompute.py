#!/usr/bin/env python3
"""Recompute tab:auditor Sonnet-generator rows (3 & 4) with SEM.

Generator = Sonnet (auditor_sonnet [+ auditor_sonnet_ext]); the round selection is fixed by the
GEMMA judge (band selection to the gemma-gen BoN band, exactly like tab:main), then the SAME chosen
transcripts are read off under each judge:
  row 3 (Sonnet-gen / Gemma-judge)  = mean GEMMA presence of chosen
  row 4 (Sonnet-gen / Sonnet-judge) = mean SONNET presence of chosen
prob columns come from the chosen transcripts' target prob_stats (identical for both rows).

--mode 60  : reproduce the paper's 60-scenario numbers (validation; expect row3~98.8 row4~89.8 prob 53.9/37.1).
--mode 98  : recompute over 60 + 38 ext (needs auditor_sonnet_ext gemma-judged into
             _wilt_analysis/crossjudge/sonnetgen_gemmajudge_ext).
"""
import os, sys, json, glob, math, statistics as st, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src", "bloom"))
sys.path.insert(0, _HERE)
import cross_score as X

CELL = "experiments/bloom/runs_final/self_harm/Qwen_Qwen3.5-4B"


def _scores(judge_dir, r):
    jp = os.path.join(judge_dir, "round_%d" % r, "judgment.json")
    if not os.path.exists(jp):
        return {}
    return {e["variation_number"]: e["behavior_presence"]
            for e in json.load(open(jp)).get("judgments", []) if e.get("behavior_presence") is not None}


def gen_points(sonnet_dir, gemma_dir, rmax=5, offset=0):
    """One entry per (scenario, round): 'score'=gemma (selection + row3), 'sonnet'=sonnet (row4),
       prob from the sonnet_dir transcript prob_stats. offset renumbers scenarios to avoid collision."""
    pts = []
    for r in range(1, rmax + 1):
        g = _scores(gemma_dir, r)
        s = _scores(sonnet_dir, r)
        for tf in glob.glob(os.path.join(sonnet_dir, "round_%d" % r, "transcripts", "*.json")):
            d = json.load(open(tf))
            v = d.get("metadata", {}).get("variation_number")
            ps = d.get("prob_stats")
            if v is None or not ps or v not in g:
                continue
            pts.append({"scenario": v + offset, "score": g[v], "sonnet": s.get(v),
                        "mean": ps["mean"], "geo": ps["geomean"], "minp": ps["min"]})
    return pts


def sem(xs):
    n = len(xs)
    return (st.pstdev(xs) * math.sqrt(n / (n - 1)) / math.sqrt(n)) if n > 1 else 0.0


def report(chosen, label):
    n = len(chosen)
    g = [c["score"] * 10 for c in chosen]
    s = [c["sonnet"] * 10 for c in chosen if c["sonnet"] is not None]
    arith = [c["mean"] for c in chosen]   # prob_stats already in percent for these transcripts
    geo = [c["geo"] for c in chosen]
    # Pareto score per scenario = mean(behaviour presence, geo-mean token prob), both 0-100
    par_g = [(c["score"] * 10 + c["geo"]) / 2 for c in chosen]
    par_s = [(c["sonnet"] * 10 + c["geo"]) / 2 for c in chosen if c["sonnet"] is not None]
    print("[%s] n=%d" % (label, n))
    print("  row3 Sonnet-gen / Gemma-judge : Pareto %.1f +- %.1f | presence %.1f +- %.1f"
          % (st.mean(par_g), sem(par_g), st.mean(g), sem(g)))
    print("  row4 Sonnet-gen / Sonnet-judge: Pareto %.1f +- %.1f | presence %.1f +- %.1f  (n_sonnet=%d)"
          % (st.mean(par_s), sem(par_s), st.mean(s), sem(s), len(s)))
    print("  prob arith %.1f +- %.1f | geo %.1f +- %.1f" % (st.mean(arith), sem(arith), st.mean(geo), sem(geo)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="60")
    a = ap.parse_args()
    xbon = X.bon_band(X.load_points(os.path.join(CELL, "bon"), rmax=8))
    print("xbon=%.4f" % xbon)
    pts = gen_points(os.path.join(CELL, "auditor_sonnet"),
                     os.path.join(CELL, "_wilt_analysis/crossjudge/sonnetgen_gemmajudge"))
    if a.mode == "98":
        pts += gen_points(os.path.join(CELL, "auditor_sonnet_ext"),
                          os.path.join(CELL, "_wilt_analysis/crossjudge/sonnetgen_gemmajudge_ext"),
                          offset=1000)
    chosen = X.select_in_band(pts, xbon)
    report(chosen, "sonnet-gen mode=%s" % a.mode)


if __name__ == "__main__":
    main()
