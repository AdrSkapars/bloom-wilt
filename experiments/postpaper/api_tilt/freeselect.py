#!/usr/bin/env python3
"""Free (non-oracle) round selection, using only data the run already stores.

rounds5.py selects best-of-R by PRESENCE, which selects on the metric it reports -- an
oracle upper bound, not a deployable number. This scores each transcript with a signal
available at generation time and never consults the judge:

    margin = mean over generated tokens of ( log p_jail - log p_target )

i.e. how much better the elicited context likes the text than the target does. Both series
are already persisted per assistant message (gen_token_probs, gen_token_probs_jail), so
this costs nothing extra. It is the api_tilt analogue of margin_pick, which reached 85-99%
of oracle in the paper work.

Reported per cell:
    vanilla-ish   round 1 mean (no selection)
    ORACLE        best-of-R by presence          <- upper bound
    MARGIN        best-of-R by margin (free)     <- deployable
    RANDOM        expected value of picking a round at random = mean of round means
and the fraction of the oracle gain the free selector recovers.

  python -X utf8 experiments/postpaper/api_tilt/freeselect.py
"""
import glob, json, math, os, re, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs_dsv4")
EPS = 1e-12


def margin(j):
    """Mean log-ratio over every generated token in the transcript, or None if absent."""
    num = 0.0
    n = 0
    for m in j.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        pt = m.get("gen_token_probs") or []
        pj = m.get("gen_token_probs_jail") or []
        if not pt or not pj or len(pt) != len(pj):
            continue
        for a, b in zip(pt, pj):
            # Empty-overlap fallback tokens are drawn from the target and carry no jail
            # probability, so they are stored as None. Skip them rather than imputing:
            # they are 0.4-2.4% of tokens and have no margin defined.
            if a is None or b is None:
                continue
            num += math.log(max(b, EPS)) - math.log(max(a, EPS))
            n += 1
    return (num / n) if n else None


def load(d):
    """{round: {scenario: (presence, plaus, margin)}}"""
    out = {}
    for rd in sorted(glob.glob(d + "/round_*"), key=lambda p: int(p.rsplit("_", 1)[1])):
        R = int(rd.rsplit("_", 1)[1])
        per = {}
        for f in sorted(glob.glob(rd + "/transcripts/*.json")):
            mm = re.search(r"_(v\d+)r(\d+)\.json$", os.path.basename(f))
            if not mm:
                continue
            try:
                j = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            s = ((j.get("judgment") or {}).get("scores") or {}).get("behavior_presence")
            p = (j.get("prob_stats") or {}).get("mean")
            if s is None or p is None:
                continue
            per[mm.group(1)] = (s * 10, p, margin(j))
        if per:
            out[R] = per
    return out


def report(label, d):
    rs = load(d)
    if len(rs) < 2:
        return
    Rs = sorted(rs)
    scen = set.intersection(*(set(rs[R]) for R in Rs))
    if not scen:
        return
    have_margin = all(rs[R][v][2] is not None for R in Rs for v in scen)

    r1 = [rs[Rs[0]][v] for v in scen]
    base = st.mean(x[0] for x in r1)
    rand = st.mean(st.mean(rs[R][v][0] for v in scen) for R in Rs)

    orc = [max((rs[R][v] for R in Rs), key=lambda t: (t[0], t[1])) for v in scen]
    o_pres, o_pl = st.mean(x[0] for x in orc), st.mean(x[1] for x in orc)

    print(f"=== {label}   rounds={Rs}  n={len(scen)}")
    print(f"  round 1 (no selection) {base:5.1f} @ {r1 and st.mean(x[1] for x in r1):5.2f}%")
    print(f"  RANDOM round           {rand:5.1f}")
    print(f"  ORACLE  best-of-R      {o_pres:5.1f} @ {o_pl:5.2f}%   (upper bound)")
    if have_margin:
        mar = [max((rs[R][v] for R in Rs), key=lambda t: t[2]) for v in scen]
        m_pres, m_pl = st.mean(x[0] for x in mar), st.mean(x[1] for x in mar)
        gain = o_pres - base
        frac = (100 * (m_pres - base) / gain) if abs(gain) > 1e-9 else float("nan")
        print(f"  MARGIN  best-of-R      {m_pres:5.1f} @ {m_pl:5.2f}%   "
              f"recovers {frac:5.1f}% of the oracle gain (free)")
    else:
        print("  MARGIN  -- gen_token_probs_jail missing on this arm (target-only decode)")
    print()


for beh, model, arms in [
    ("self_harm", "deepseek_v4_flash",
     ["api_overlap_elicited_15s", "api_overlap_combined_15s", "api_elicited_15s"]),
    ("self_harm", "gpt_oss_120b", ["api_overlap_elicited_15s"]),
]:
    for arm in arms:
        report(f"{beh} / {model} / {arm}", os.path.join(RUNS, beh, model, arm))
