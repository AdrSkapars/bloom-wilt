#!/usr/bin/env python3
"""Per-run cost: wall-clock, API calls, prompt tokens.

`calls` and `prompt_tok` in api_cost.jsonl are CUMULATIVE over the process, so the run
total is the last record's value minus the first record's value plus that first record's
own share -- simpler to take max - min + first-delta, which for a fresh process is just
the max. Each run is its own process, so max is the run total.

Prompt tokens are the number that matters for comparing the two decoders: both re-send the
whole conversation on every call, so a decoder making 35x fewer calls sends roughly 35x
fewer prompt tokens, and that is what the provider bills.

  python -X utf8 experiments/postpaper/api_tilt/cost.py <run_dir> [<run_dir> ...]
"""
import glob
import json
import os
import sys


def run_cost(d):
    recs = []
    for f in glob.glob(os.path.join(d, "**", "api_cost.jsonl"), recursive=True) + \
             glob.glob(os.path.join(d, "api_cost.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    if not recs:
        return None
    return {"secs": sum(r.get("secs", 0.0) for r in recs),
            "calls": max(r.get("calls", 0) for r in recs),
            "prompt_tok": max(r.get("prompt_tok", 0) for r in recs),
            "gen_tokens": sum(r.get("gen_tokens", 0) for r in recs),
            "stages": len(recs)}


if __name__ == "__main__":
    print("%-52s %9s %9s %13s %9s" % ("run", "secs", "calls", "prompt_tok", "gen_tok"))
    for d in sys.argv[1:]:
        c = run_cost(d)
        name = os.path.basename(d.rstrip("/\\"))
        if not c:
            print("%-52s %9s" % (name[:52], "(no cost data)"))
            continue
        print("%-52s %9.1f %9d %13d %9d"
              % (name[:52], c["secs"], c["calls"], c["prompt_tok"], c["gen_tokens"]))
