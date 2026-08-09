#!/usr/bin/env python3
"""tab:main Sonnet re-judge (auditor-robustness, method axis).

For each tab:main method (Qwen3.5-4B / self-harm, Gemma generator), take the SAME Gemma-selected
transcripts that produced tab:main and re-judge ONLY those with Claude Sonnet 4.6 (API), via the
pipeline's own run_judgment (ignore_cache=True, thinking off). Extends gemmagen_sonnetjudge (which
covered LogitTilt only) to every method.

Selection reproduces app:logittilt:
  - band-frontier (prob >= BoN band, max presence): G-PAIR, LogitTilt, WILT  -> cross_score.select_in_band
  - w=1 (best presence per scenario): BoN, BEAST-in, FLRT, BEAST-out, TokenBias -> cross_score.op_point_points
Band = BoN frontier max-presence operating-point prob.

LogitTilt is NOT re-judged (reuse existing gemmagen_sonnetjudge Sonnet scores for the SAME selected
transcripts -> zero API spend, identical selection). Pass --force-rejudge to judge it fresh anyway.

  # 1) validate selection reproduces paper GEMMA numbers (NO Sonnet spend):
  ... methods_sonnet_rejudge.py --validate
  # 2) full Sonnet re-judge (spends API $ on ~7*100 selected transcripts):
  ... methods_sonnet_rejudge.py
  # single method:
  ... methods_sonnet_rejudge.py --only wilt
"""
import os, sys, json, glob, copy, argparse, asyncio, statistics as st
from pathlib import Path
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
SRC_BLOOM = os.path.join(_REPO, "src", "bloom")
sys.path.insert(0, SRC_BLOOM)
sys.path.insert(0, _HERE)

CELL_REL = "self_harm/Qwen_Qwen3.5-4B"
BEH_YAML = "prompts/7_self_harm_encouragement.yaml"
# name, folder(rel cell), rule, rmax, paper_presence, reuse(rel cell | None)
METHODS = [
    ("bon",       "bon",                    "w1",   8, 51.0, None),
    ("beast_in",  "input_search_3turn",     "w1",   1, 31.7, None),
    ("flrt",      "flrt",                   "w1",   1, 33.5, None),
    ("gpair",     "gpair_t3_sfull",         "band", 7, 66.2, None),
    ("beast_out", "output_search_3turn",    "w1",   1, 51.3, None),
    ("tokbias",   "tokbias_tuned/lam_0.5",  "w1",   8, 64.7, None),
    ("logittilt", "jail_b1.5",              "band", 5, 99.5, "_wilt_analysis/crossjudge/gemmagen_sonnetjudge"),
    ("wilt",      "combo/beta_1.5",         "band", 5, 100.0, None),
]


def _lazy():
    import bloom.core as core
    from bloom.core import load_prompts
    from bloom.pipeline import run_judgment
    import bloom_corrupt as B
    import cross_score as X
    return core, load_prompts, run_judgment, B, X


def build_cfg(B, load_prompts, judge_model):
    cfg = copy.deepcopy(B.cfg)
    cfg["batch_size"] = 32
    cfg["judgment"]["model"] = judge_model
    cfg["judgment"]["thinking"] = False
    cfg["behavior_file"] = BEH_YAML
    y = yaml.safe_load(open(os.path.join(SRC_BLOOM, BEH_YAML), encoding="utf-8"))
    cfg["behavior_name"] = y["behavior_name"]
    cfg["behavior_description"] = (y.get("behavior_description") or "").strip()
    for k, v in (y.get("overrides") or {}).items():
        if k not in cfg:
            cfg[k] = v.strip() if isinstance(v, str) else v
    return cfg, load_prompts(cfg)


def select(X, cell, folder, rule, rmax, xbon):
    pts = X.load_points(os.path.join(cell, folder), rmax=rmax)
    return X.select_in_band(pts, xbon) if rule == "band" else X.op_point_points(pts)


def stage(chosen, dst):
    td = os.path.join(dst, "transcripts")
    os.makedirs(td, exist_ok=True)
    for f in glob.glob(os.path.join(td, "*.json")):
        os.remove(f)
    picks = {}
    for p in chosen:
        d = json.load(open(p["path"], encoding="utf-8"))
        v = p["scenario"]
        json.dump(d, open(os.path.join(td, "transcript_v%dr1.json" % v), "w"))
        picks[v] = p
    return picks


def _round_of(path):
    for part in path.replace("\\", "/").split("/"):
        if part.startswith("round_"):
            return int(part.split("_")[1])
    return None


def reuse_sonnet(reuse_dir, chosen):
    """For each selected (scenario, round) pick, read the Sonnet behaviour_presence from the
    existing gemmagen_sonnetjudge round judgment.json (same transcripts, already Sonnet-judged)."""
    cache = {}
    out = {}
    for p in chosen:
        r = _round_of(p["path"])
        v = p["scenario"]
        if r is None:
            continue
        if r not in cache:
            jp = os.path.join(reuse_dir, "round_%d" % r, "judgment.json")
            cache[r] = ({e["variation_number"]: e["behavior_presence"]
                         for e in json.load(open(jp, encoding="utf-8")).get("judgments", [])}
                        if os.path.exists(jp) else {})
        if v in cache[r]:
            out[v] = cache[r][v]
    return out


def run(args):
    core, load_prompts, run_judgment, B, X = _lazy()
    core._DEFAULT_LOCAL_GPU_ID = args.gpu
    cell = os.path.join(args.root, CELL_REL)
    xbon = X.bon_band(X.load_points(os.path.join(cell, "bon"), rmax=8))
    print("[rejudge] BoN band x_bon=%.4f" % xbon, flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    summary = {}
    cfg = prompts = und = None
    if not args.validate:
        cfg, prompts = build_cfg(B, load_prompts, args.judge_model)
        und = json.load(open(os.path.join(args.root, "self_harm", "_bank", "understanding.json"), encoding="utf-8"))
    recf = open(os.path.join(args.outdir, "records.jsonl"), "a", encoding="utf-8")
    for (name, folder, rule, rmax, paper, reuse) in METHODS:
        if only and name not in only:
            continue
        chosen = select(X, cell, folder, rule, rmax, xbon)
        gem = 10 * st.mean(c["score"] for c in chosen)
        flag = "" if abs(gem - paper) <= 2.0 else "  <<< MISMATCH vs paper"
        print("[%s] %s rule=%s rmax=%s n=%d GEMMA_sel=%.1f (paper %s)%s"
              % (name, folder, rule, rmax, len(chosen), gem, paper, flag), flush=True)
        if args.validate:
            summary[name] = {"gemma_sel": round(gem, 1), "paper": paper, "n": len(chosen), "ok": abs(gem - paper) <= 2.0}
            continue
        if reuse and not args.force_rejudge:
            son = reuse_sonnet(os.path.join(cell, reuse), chosen)
            savg = 10 * st.mean(son.values()) if son else float("nan")
            print("[%s] REUSED existing Sonnet judge: SONNET_sel=%.1f (n=%d)" % (name, savg, len(son)), flush=True)
            for v, s in son.items():
                recf.write(json.dumps({"method": name, "var": v, "sonnet": s, "reused": True}) + "\n")
            recf.flush()
            summary[name] = {"gemma_sel": round(gem, 1), "sonnet_sel": round(savg, 1), "paper": paper, "n": len(son), "reused": True}
            continue
        dst = os.path.join(args.tmp, name)
        picks = stage(chosen, dst)
        asyncio.run(run_judgment(cfg, prompts, Path(dst), und, {"variations": []},
                                 out_name="judgment_sonnet.json", ignore_cache=True))
        j = json.load(open(os.path.join(dst, "judgment_sonnet.json"), encoding="utf-8"))
        sc = {e["variation_number"]: e["behavior_presence"] for e in j.get("judgments", []) if e.get("behavior_presence") is not None}
        savg = 10 * st.mean(sc.values()) if sc else float("nan")
        print("[%s] SONNET_sel=%.1f (n=%d)" % (name, savg, len(sc)), flush=True)
        for v, s in sc.items():
            recf.write(json.dumps({"method": name, "var": v, "gemma": picks[v]["score"], "sonnet": s}) + "\n")
        recf.flush()
        summary[name] = {"gemma_sel": round(gem, 1), "sonnet_sel": round(savg, 1), "paper": paper, "n": len(sc)}
    recf.close()
    json.dump(summary, open(os.path.join(args.outdir, "summary.json"), "w"), indent=2)
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/bloom/runs_final")
    ap.add_argument("--outdir", default="experiments/bloom/runs_final/self_harm/Qwen_Qwen3.5-4B/_wilt_analysis/crossjudge/methods_sonnetjudge")
    ap.add_argument("--tmp", default="/workspace/methods_rejudge_tmp")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--validate", action="store_true", help="print Gemma re-selection vs paper; no Sonnet spend")
    ap.add_argument("--force-rejudge", action="store_true", help="re-judge LogitTilt fresh instead of reusing")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
