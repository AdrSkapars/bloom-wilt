#!/usr/bin/env python3
"""Gemma-judge the 38 auditor_sonnet_ext transcripts (all 5 rounds) so the Sonnet-generator rows of
tab:auditor can be recomputed over 98. Copies each round's transcripts into
_wilt_analysis/crossjudge/sonnetgen_gemmajudge_ext/round_R/ and runs the pipeline's local Gemma judge
(default cfg judge = EVAL_MODELS[0] = gemma-4-26B), ignore_cache=True. Mirrors how sonnetgen_gemmajudge
(the 60) was produced. GPU0."""
import os, sys, json, glob, copy, shutil
from pathlib import Path
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
SRC_BLOOM = os.path.join(_REPO, "src", "bloom")
sys.path.insert(0, SRC_BLOOM)

import bloom.core as core
from bloom.core import load_prompts
from bloom.pipeline import run_judgment_batched_local
import bloom_corrupt as B

CELL = os.path.join(_REPO, "experiments/bloom/runs_final/self_harm/Qwen_Qwen3.5-4B")
SRC = os.path.join(CELL, "auditor_sonnet_ext")
DST = os.path.join(CELL, "_wilt_analysis/crossjudge/sonnetgen_gemmajudge_ext")
BEH = "prompts/7_self_harm_encouragement.yaml"


def main():
    core._DEFAULT_LOCAL_GPU_ID = 0
    cfg = copy.deepcopy(B.cfg)
    cfg["batch_size"] = 32
    cfg["behavior_file"] = BEH
    y = yaml.safe_load(open(os.path.join(SRC_BLOOM, BEH), encoding="utf-8"))
    cfg["behavior_name"] = y["behavior_name"]
    cfg["behavior_description"] = (y.get("behavior_description") or "").strip()
    for k, v in (y.get("overrides") or {}).items():
        if k not in cfg:
            cfg[k] = v.strip() if isinstance(v, str) else v
    prompts = load_prompts(cfg)
    und = json.load(open(os.path.join(_REPO, "experiments/bloom/runs_final/self_harm/_bank/understanding.json"), encoding="utf-8"))
    print("judge model =", cfg["judgment"]["model"], flush=True)

    for r in range(1, 6):
        srcdir = os.path.join(SRC, "round_%d" % r, "transcripts")
        dstdir = os.path.join(DST, "round_%d" % r)
        tdir = os.path.join(dstdir, "transcripts")
        os.makedirs(tdir, exist_ok=True)
        for f in glob.glob(os.path.join(tdir, "*.json")):
            os.remove(f)
        files = glob.glob(os.path.join(srcdir, "*.json"))
        for f in files:
            shutil.copy(f, tdir)
        run_judgment_batched_local(cfg, prompts, Path(dstdir), und, {"variations": []},
                                   out_name="judgment.json", ignore_cache=True)
        j = json.load(open(os.path.join(dstdir, "judgment.json"), encoding="utf-8"))
        print("round %d: gemma-judged %d transcripts (%d judgments)" % (r, len(files), len(j.get("judgments", []))), flush=True)

    print("GEMMA_JUDGE_EXT_DONE", flush=True)


if __name__ == "__main__":
    main()
