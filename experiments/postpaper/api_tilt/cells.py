#!/usr/bin/env python3
"""Cross-behaviour, cross-model summary of the api_tilt cells.

`summarize.py` covers one behaviour on one model in depth (the DeepSeek self_harm arm
sweep). This is the wide view: every (behaviour x model) cell at vanilla and overlap b=1,
which is what the robustness question actually needs.

  python -X utf8 experiments/postpaper/api_tilt/cells.py
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_dsv4")
BEHS = [("self_harm", "self-harm"), ("goblin", "goblin"), ("selfpres", "self-pres")]
MODELS = [("deepseek_v4_flash", "DeepSeek-V4-Flash"), ("glm_5p3_flash", "GLM-5.3-Flash"),
          ("gpt_oss_120b", "gpt-oss-120b"), ("qwen3p7_plus", "Qwen3.7-Plus")]
ARMS = [("api_vanilla_15s", "vanilla"), ("api_overlap_combined_15s", "overlap b=1")]


def cell(beh, model, arm):
    d = os.path.join(RUNS, beh, model, arm, "round_1", "transcripts")
    fs = sorted(glob.glob(os.path.join(d, "*.json")))
    sc, pr = [], []
    for f in fs:
        j = json.load(open(f, encoding="utf-8"))
        s = ((j.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
        p = (j.get("prob_stats") or {}).get("mean")
        if s is not None and p is not None:
            sc.append(s * 10)
            pr.append(p)
    if not sc:
        return None
    return len(sc), st.mean(sc), st.mean(pr)


rows = []
for mkey, mlab in MODELS:
    for bkey, blab in BEHS:
        v = cell(bkey, mkey, "api_vanilla_15s")
        o = cell(bkey, mkey, "api_overlap_combined_15s")
        if v or o:
            rows.append((mlab, blab, v, o))

if not rows:
    raise SystemExit("no cells found")

print(f"{'model':18s} {'behaviour':10s} {'vanilla':>16s} {'overlap b=1':>16s} "
      f"{'d presence':>10s} {'d prob':>8s}")
print("-" * 84)
last = None
for mlab, blab, v, o in rows:
    m = mlab if mlab != last else ""
    last = mlab
    vs = f"{v[1]:5.1f} @ {v[2]:5.2f}%" if v else "        --      "
    os_ = f"{o[1]:5.1f} @ {o[2]:5.2f}%" if o else "        --      "
    if v and o:
        dp, dq = f"{o[1]-v[1]:+10.1f}", f"{o[2]-v[2]:+7.2f}"
    else:
        dp = dq = "        --"[:10] if not (v and o) else ""
    print(f"{m:18s} {blab:10s} {vs:>16s} {os_:>16s} {dp:>10s} {dq:>8s}")

print("\npresence = mean behaviour presence 0-100 (judge score x10), round 1, 15 scenarios.")
print("A cell needs the behaviour to be REACHABLE from the target's own top-5: the overlap")
print("rule can only re-weight candidates the target already proposes, never introduce one")
print("it never would. That is the expected reason goblin moves least.")
