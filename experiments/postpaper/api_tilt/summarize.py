#!/usr/bin/env python3
"""Round-1 comparison of the DeepSeek-V4 self-harm arms, local and hosted.

Round 1 only, so the hosted single-round arm is compared like for like against the
first round of the local arms (pool depth changes the selection story, not this table).

  python -X utf8 experiments/postpaper/api_tilt/summarize.py
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_dsv4", "self_harm", "deepseek_v4_flash")

# (folder, label, b1, b2, floor, engine)
ARMS = [
    ("vanilla_15s",      "target only",   "1", "0",   "1e-4", "local"),
    ("jail_b0.5",        "tilt b2=0.5",   "1", "0.5", "1e-4", "local"),
    ("jail_b1",          "tilt b2=1",     "1", "1",   "1e-4", "local"),
    ("jail_b2",          "tilt b2=2",     "1", "2",   "1e-4", "local"),
    ("api_vanilla_15s",  "target only",   "1", "0",   "0",    "api"),
    ("api_elicited_15s", "elicited only", "0", "1",   "0",    "api"),
    ("api_overlap_elicited_15s", "overlap: elicited", "-", "-", "0", "api"),
    ("api_overlap_target_15s",   "overlap: target",   "-", "-", "0", "api"),
    ("api_overlap_combined_15s", "overlap: combined", "-", "-", "0", "api"),
    ("api_overlap_random_15s",   "overlap: random",   "-", "-", "0", "api"),
    ("api_overlap_combined_min_15s",    "combined: bottom-1",  "-", "-", "0", "api"),
    ("api_overlap_combined_sample_15s", "combined: weighted",  "-", "-", "0", "api"),
    ("api_overlap_combined_fbweighted_15s", "combined, fb=wtd",   "-", "-", "0", "api"),
    ("api_overlap_combined_fbrandom_15s",   "combined, fb=rand",  "-", "-", "0", "api"),
    ("api_overlap_combined_fbargmax_15s",   "combined, fb=argmx", "-", "-", "0", "api"),
    ("api_overlap_combined_b0.5_15s",        "combined b=0.5",     "-", "-", "0", "api"),
    ("api_overlap_combined_b1.5_15s",        "combined b=1.5",     "-", "-", "0", "api"),
    ("api_overlap_combined_b2_15s",          "combined b=2",       "-", "-", "0", "api"),
    ("api_overlap_combined_sample_b0.5_15s", "weighted b=0.5",     "-", "-", "0", "api"),
    ("api_overlap_combined_sample_b2_15s",   "weighted b=2",       "-", "-", "0", "api"),
    ("api_overlap_combined_nopf_15s",        "combined, no prefill", "-", "-", "0", "api"),
]


def load(arm):
    """Per-transcript presence (0-100), mean token prob, and min token prob."""
    rows = []
    for tf in sorted(glob.glob(os.path.join(RUNS, arm, "round_1", "transcripts", "*.json"))):
        d = json.load(open(tf, encoding="utf-8"))
        sc = ((d.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
        # Top-level prob_stats is the transcript-level roll-up the judgment stage writes
        # (n-weighted mean over turns, min of mins) — the same field selection_plot.py reads.
        ps = d.get("prob_stats") or {}
        if sc is None or ps.get("mean") is None:
            continue
        rows.append((float(sc) * 10, float(ps["mean"]), float(ps["min"])))
    return rows


print(f"{"arm":18s} {'engine':7s} {'b1':>3s} {'b2':>4s} {'floor':>5s} {'n':>3s} "
      f"{'presence':>9s} {'tok mean':>9s} {"min tok prob":>12s}")
print("-" * 82)
for folder, label, b1, b2, floor, engine in ARMS:
    rows = load(folder)
    if not rows:
        continue
    print(f"{label:18s} {engine:7s} {b1:>3s} {b2:>4s} {floor:>5s} {len(rows):3d} "
          f"{st.mean(r[0] for r in rows):9.1f} {st.mean(r[1] for r in rows):8.2f}% "
          f"{min(r[2] for r in rows)/100:11.2e}")
print("\npresence = mean behaviour presence, 0-100 (judge score x10); round 1 only.")
print("min tok prob = lowest single-token TARGET probability anywhere in the arm (not %).")
