#!/usr/bin/env python3
"""Cumulative best-of-R selection over rounds, paired.

For each scenario take the best transcript seen across rounds 1..R by presence, then report
mean presence and the mean plausibility OF THOSE SELECTED transcripts -- paired, so the two
numbers describe the same point. (An earlier version of this analysis paired max-presence
with max-plausibility from different transcripts and reported an impossible operating
point.)

CAVEAT: selecting by presence is ORACLE selection -- it selects on the metric it reports, so
these are upper bounds, not deployable numbers. A real selector (cf. margin_pick, which
reached 85-99% of oracle) is needed for a quotable figure.

  python -X utf8 experiments/postpaper/api_tilt/rounds5.py
"""
import glob, json, os, re, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_dsv4")


def rounds(d):
    out = {}
    for rd in sorted(glob.glob(d + "/round_*"), key=lambda p: int(p.rsplit("_", 1)[1])):
        R = int(rd.rsplit("_", 1)[1])
        per = {}
        for f in sorted(glob.glob(rd + "/transcripts/*.json")):
            # Scenario id must be parsed with a regex: splitting the basename on "r" also
            # splits inside "transcript", collapsing every file onto one key.
            m = re.search(r"_(v\d+)r(\d+)\.json$", os.path.basename(f))
            if not m:
                continue
            j = json.load(open(f, encoding="utf-8"))
            s = ((j.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
            p = (j.get("prob_stats") or {}).get("mean")
            if s is None or p is None:
                continue
            per[m.group(1)] = (s * 10, p)
        if per:
            out[R] = per
    return out


def report(label, d):
    rs = rounds(d)
    if not rs:
        return
    print(f"=== {label}   (n per round: {[len(rs[R]) for R in sorted(rs)]})")
    best = {}
    for R in sorted(rs):
        for k, (s, p) in rs[R].items():
            # Tie-break on plausibility: among transcripts of equal presence take the most
            # plausible one. Without this, an arm already at ceiling presence (elicited-only
            # is 100.0 every round) keeps round 1 forever and its selection budget is
            # silently discarded -- which would understate it against the overlap arms.
            if k not in best or (s, p) > (best[k][0], best[k][1]):
                best[k] = (s, p)
        cur = rs[R]
        print(f"  round {R}: this-round {st.mean(v[0] for v in cur.values()):5.1f} @ "
              f"{st.mean(v[1] for v in cur.values()):5.2f}%   |  best-of-1..{R} "
              f"{st.mean(v[0] for v in best.values()):5.1f} @ "
              f"{st.mean(v[1] for v in best.values()):5.2f}%")


for beh, model, arms in [
    ("self_harm", "deepseek_v4_flash", ["api_overlap_elicited_15s", "api_overlap_combined_15s",
                                        "api_elicited_15s"]),
    ("self_harm", "gpt_oss_120b", ["api_overlap_elicited_15s"]),
    ("goblin", "glm_5p3_flash", ["api_overlap_elicited_15s"]),
]:
    for arm in arms:
        report(f"{beh} / {model} / {arm}", os.path.join(RUNS, beh, model, arm))
