#!/usr/bin/env python3
"""How much probability mass do the top-1..top-5 alternatives actually cover?

This is the feasibility question for approximating `z = b1*l_target + b2*l_jail` from a
provider that returns only 5 alternatives per position: if the top-5 already hold nearly
all the mass, the truncation is cheap; where they do not, the tilt is being reconstructed
over a candidate set that is missing most of the distribution.

Reported per context (target / elicited) and per direction (whose tokens the contexts were
forced along), over every position in the collected top-5 datasets.

  python -X utf8 experiments/postpaper/api_tilt/top5_mass.py
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARM_ROOT = REPO / "experiments/postpaper/runs_dsv4/self_harm/deepseek_v4_flash"
ARMS = [("vanilla_15s", "target-only tokens"), ("api_elicited_15s", "elicited-only tokens")]


def collect(arm, rnd):
    """cum[ctx] = list over positions of [top1, top1..2, ..., top1..5] cumulative mass."""
    d = ARM_ROOT / arm / f"round_{rnd}" / "top5"
    cum = {"target": [], "elicited": []}
    inter = []          # mass each context puts on the SHARED top-5 tokens
    short = 0
    for f in sorted(d.glob("*.top5.json")):
        for t in json.load(open(f, encoding="utf-8"))["turns"]:
            for i in range(t["n_tokens"]):
                a, b = t["target"]["top"][i], t["elicited"]["top"][i]
                if not a or not b:
                    continue
                if len(a) < 5 or len(b) < 5:
                    short += 1
                for ctx, lst in (("target", a), ("elicited", b)):
                    ps, run, row = [math.exp(v) for _, v in lst], 0.0, []
                    for p in ps:
                        run += p
                        row.append(run)
                    while len(row) < 5:          # pad a short list with its own total
                        row.append(run)
                    cum[ctx].append(row)
                shared = {s for s, _ in a} & {s for s, _ in b}
                inter.append((sum(math.exp(v) for s, v in a if s in shared),
                              sum(math.exp(v) for s, v in b if s in shared)))
    return cum, inter, short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()

    data = {arm: collect(arm, args.round) for arm, _ in ARMS}

    print("MEAN CUMULATIVE TOP-k PROBABILITY MASS\n")
    hdr = f"{'k':>2s} |"
    for _, lbl in ARMS:
        hdr += f" {lbl:^25s} |"
    print(hdr)
    print(f"{'':>2s} |" + "".join(f" {'target':>11s} {'elicited':>12s} |" for _ in ARMS))
    print("-" * len(hdr))
    for k in range(5):
        row = f"{k+1:>2d} |"
        for arm, _ in ARMS:
            cum = data[arm][0]
            row += (f" {100*st.mean(r[k] for r in cum['target']):10.2f}%"
                    f" {100*st.mean(r[k] for r in cum['elicited']):11.2f}% |")
        print(row)

    print("\n\nDISTRIBUTION OF TOP-5 TOTAL MASS  (% of positions below each threshold)\n")
    print(f"{'top-5 mass <':>13s} |" + "".join(
        f" {lbl[:12]:^25s} |" for _, lbl in ARMS))
    print(f"{'':>13s} |" + "".join(f" {'target':>11s} {'elicited':>12s} |" for _ in ARMS))
    print("-" * 82)
    for thr in (0.50, 0.80, 0.90, 0.95, 0.99):
        row = f"{thr:>12.0%} |"
        for arm, _ in ARMS:
            cum = data[arm][0]
            for ctx in ("target", "elicited"):
                v = cum[ctx]
                row += f" {100*sum(1 for r in v if r[4] < thr)/len(v):11.1f}%"
            row += " |"
        print(row)

    print("\n\nMEDIAN / 10th-PERCENTILE TOP-5 TOTAL MASS\n")
    for arm, lbl in ARMS:
        cum, inter, short = data[arm]
        n = len(cum["target"])
        print(f"  {lbl}  ({n} positions" + (f", {short} with <5 alternatives" if short else "") + ")")
        for ctx in ("target", "elicited"):
            v = sorted(r[4] for r in cum[ctx])
            print(f"    {ctx:9s} median {100*v[len(v)//2]:6.2f}%   "
                  f"p10 {100*v[len(v)//10]:6.2f}%   min {100*v[0]:6.2f}%")
        # Mass on the SHARED tokens: what a top-5 tilt reconstruction can actually see.
        print(f"    shared-token mass: target {100*st.mean(x for x, _ in inter):.2f}%   "
              f"elicited {100*st.mean(y for _, y in inter):.2f}%")
        print()


if __name__ == "__main__":
    main()
