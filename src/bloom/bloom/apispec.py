"""Speculative decode for the hosted-API tilt: draft in blocks, verify, rewind on violation.

The single-position rule in apitilt.py spends TWO calls per emitted token -- one per context
-- because it needs both distributions at every position before it can pick. That is the
whole cost model, and it is why a 15-scenario cell takes ~12 minutes.

This trades that for a speculative loop:

  1. draft N tokens from the ELICITED context in one call, with top-k at every position;
  2. teacher-force those N tokens under the TARGET in one echo call, which returns the
     target's logprob for each drafted token AND the target's top-k at every position;
  3. accept the longest prefix whose target probability clears `floor`;
  4. at the first violation, resolve that ONE position with the single-position mixture rule
     -- which costs nothing extra, because step 2 already handed back both contexts' top-k
     there -- and discard the rest of the block, since emitting a different token makes every
     later draft conditioned on a context that no longer exists.

So two calls buy (accepted prefix + 1) tokens instead of one. The acceptance rate is the
whole story: a block that survives intact gives N tokens for 2 calls, one that fails
immediately still gives 1 token for 2 calls, i.e. never worse than the single-position rule.

The accepted tokens are pure ELICITED picks -- the target only holds a veto, never a vote --
so this is a different operating point from apitilt's mixture, closer to elicited-only with a
plausibility floor. `floor` is the only dial between the two: raise it and the behaviour
approaches the mixture while the blocks shorten and the speedup goes away.
"""
from __future__ import annotations

import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .apitilt import ApiTiltTarget, _finalize, _record_cost, _resolver

__all__ = ["_driven_spec"]


def _q_of(t_top, j_top, metric: str) -> float:
    """Disagreement at a position, from the two top-k lists. Same measures as apitilt."""
    tmap = dict(t_top)
    te = sum(math.exp(lp) for _, lp in j_top) or 1.0
    if metric == "tv":
        tt = sum(math.exp(lp) for _, lp in t_top) or 1.0
        pt = {s: math.exp(lp) / tt for s, lp in t_top}
        pe = {s: math.exp(lp) / te for s, lp in j_top}
        q = 0.5 * sum(abs(pt.get(s, 0.0) - pe.get(s, 0.0)) for s in set(pt) | set(pe))
    elif metric == "margin":
        pe = {s: math.exp(lp) / te for s, lp in j_top}
        etop = max(pe, key=pe.get) if pe else None
        ttop = t_top[0][0] if t_top else None
        q = max(0.0, (pe.get(etop, 0.0) - pe.get(ttop, 0.0)) if etop else 0.0)
    else:
        q = sum(math.exp(lp) for s, lp in j_top if s not in tmap) / te
    return min(1.0, max(0.0, q))


def _resolve_one(res, t_top, j_top, alpha0, alpha_k, floor, metric, sample_temp,
                 alpha_fixed=None):
    """The single-position mixture rule, over the union of two top-k lists already in hand.

    Returns (token_id, target_lp, elicited_lp), or None if the floor leaves no candidate.
    """
    tmap = dict(t_top)
    jmap = dict(j_top)
    if alpha_fixed is not None:
        # An intervention is triggered precisely because the contexts disagree here, so
        # re-deriving alpha from the schedule is self-defeating: at theta=0.5 with k=10 the
        # schedule returns alpha=0.599, i.e. the position is resolved almost entirely by the
        # target -- the very context whose disagreement triggered the intervention. A fixed
        # low alpha (high elicited weight) makes the intervention actually intervene.
        alpha = max(float(alpha_fixed), 1e-9)
    else:
        q = _q_of(t_top, j_top, metric)
        alpha = max(alpha0 * (1.0 - q ** alpha_k), 1e-9)
    union = {}
    for s, lp in t_top:
        union[s] = [math.exp(lp), 0.0]
    for s, lp in j_top:
        union.setdefault(s, [0.0, 0.0])[1] = math.exp(lp)
    cands = [(s, alpha * xy[0] + (1.0 - alpha) * xy[1]) for s, xy in union.items()]
    if floor > 0.0:
        # Target-side candidates only. Their probabilities are known exactly, so the floor
        # is enforced for free; an elicited-only candidate would need a call to price, and
        # avoiding calls is the point of this path.
        #
        # An earlier version admitted elicited-only candidates whenever the target's k-th
        # best cleared the floor, reusing the free bound t(x) <= min_T. That bound is an
        # UPPER bound: a low min_T proves a candidate fails, but a high one proves nothing
        # about it. The unsound converse let sub-floor tokens through -- measured, it put
        # the run's minimum at 3.7e-06 against a floor of 1%.
        cands = [(s, w) for s, w in cands
                 if s in tmap and math.exp(tmap[s]) * 100.0 >= floor]
    if not cands:
        return None
    if sample_temp <= 0.0:
        pick = max(cands, key=lambda x: x[1])[0]
    else:
        wmax = max(w for _, w in cands) or 1.0
        e = 1.0 / sample_temp
        d = [(s, (w / wmax) ** e) for s, w in cands]
        tot = sum(w for _, w in d) or 1.0
        r, acc = random.random() * tot, 0.0
        pick = d[-1][0]
        for s, w in d:
            acc += w
            if r <= acc:
                pick = s
                break
    tid = res.id_of(pick)
    if tid is None:
        return None
    return tid, tmap.get(pick), jmap.get(pick)


def _driven_spec(handle, jail_runtime_cfg, target_msgs_batch, max_tokens,
                 temperature, no_think_target):
    """[api_tilt rule=spec] Block-draft from the elicited context, verify under the target."""
    client: ApiTiltTarget = handle["client"]
    res = _resolver()
    NO_THINK = handle.get("target_no_think", "")
    NO_THINK_C = handle.get("corrupt_no_think", "")
    sys_prompt = jail_runtime_cfg.get("system_prompt", "")
    prefill = jail_runtime_cfg.get("prefill", "") or ""
    top_k = int(jail_runtime_cfg.get("api_top_k", 5) or 5)
    floor = float(jail_runtime_cfg.get("api_floor", 0.0) or 0.0)
    block = int(jail_runtime_cfg.get("api_spec_block", 10) or 10)
    alpha0 = float(jail_runtime_cfg.get("api_alpha0", 0.6))
    alpha_k = float(jail_runtime_cfg.get("api_alpha_k", 10.0) or 10.0)
    metric = str(jail_runtime_cfg.get("api_q_metric", "elicited_outside") or "elicited_outside")
    sample_temp = float(jail_runtime_cfg.get("api_sample_temp", 0.05) or 0.05)
    # Temperature for the DRAFT block. <0 means "use the rollout temperature", i.e. draw as
    # the elicited context would on its own. 0 drafts its greedy continuation instead, which
    # should raise acceptance -- a greedy token is the elicited context's most probable, and
    # the two contexts agree more often on high-probability tokens than on tail ones -- at
    # the cost of the round-to-round diversity a sampled draft provides.
    draft_temp = float(jail_runtime_cfg.get("api_spec_draft_temp", -1.0))
    if draft_temp < 0.0:
        draft_temp = float(temperature)
    # WHICH CONTEXT DRAFTS.
    #   elicited -- draft what the jailbroken context wants, accept while the TARGET prices
    #               it above `floor`. Accepted tokens are elicited picks, so the operating
    #               point sits near elicited-only and the target holds a veto, not a vote.
    #   target   -- the mirror: draft the TARGET continuation, accept while the two contexts
    #               still agree, and intervene where disagreement q reaches stage2_theta.
    #               Accepted tokens are then target picks, so plausibility is high by
    #               construction and the intervention RATE becomes the behaviour dial.
    draft_side = str(jail_runtime_cfg.get("api_spec_draft", "elicited") or "elicited")
    if draft_side not in ("elicited", "target"):
        raise RuntimeError("api_jailbroken_output.spec_draft=%r unknown (elicited | target)"
                           % draft_side)
    theta = float(jail_runtime_cfg.get("api_stage2_theta", 0.95) or 0.95)
    # Alpha to use AT an intervention, overriding the schedule. <0 keeps the schedule.
    # 0 means the intervened position is resolved by the elicited context alone (subject to
    # the floor), which is what "intervene" ought to mean.
    a_int = float(jail_runtime_cfg.get("api_spec_intervene_alpha", -1.0))
    a_int = None if a_int < 0.0 else a_int
    # BURST. After an intervention, keep drafting from the ELICITED context for this many
    # tokens before reverting to the base side. 0 disables.
    #
    # Isolated interventions do not compound: target_every=2 steered half of all positions
    # and collapsed to vanilla, and a target draft with alpha=0 at 235 intervened positions
    # moved presence by 5. In both, every steered token was immediately followed by a
    # target-drafted one that pulled the context back. A burst keeps the steering in place
    # for a run of consecutive tokens, which is the thing those failures say is needed.
    burst = int(jail_runtime_cfg.get("api_spec_burst", 0) or 0)
    # FLIP. Instead of a fixed burst length, let each side run until ITS OWN stop condition
    # fires and then hand over: draft from the target until disagreement reaches
    # stage2_theta, draft from the elicited context until the target prices a token below
    # `floor`, and alternate. Both accept tests already exist; this just makes the side a
    # state that toggles on every intervention rather than a timer.
    #
    # The fixed burst is arbitrary in exactly the place this is not: a burst of 10 ends
    # whether or not the steering is still productive, whereas the elicited phase here ends
    # precisely when it stops being plausible.
    flip = bool(jail_runtime_cfg.get("api_spec_flip", False))

    def _one(job):
        idx, tm = job
        aff = "spec-%d-%d" % (os.getpid(), idx)
        t_prefix = client.render(tm, add_generation_prompt=True) + (NO_THINK if no_think_target else "")
        conv = [m for m in tm if m.get("role") != "system"]
        j_msgs = ([{"role": "system", "content": sys_prompt}] + conv) if sys_prompt else conv
        j_prefix = client.render(j_msgs, add_generation_prompt=True) + NO_THINK_C + prefill
        t_ids = client.prefix_ids(t_prefix)
        j_ids = client.prefix_ids(j_prefix)

        gen, t_lps, j_lps = [], [], []
        burst_left = 0
        cur_side = draft_side
        n_calls = n_blocks = n_accept = n_rewind = n_stall = n_burst = 0
        truncated = ""
        while len(gen) < int(max_tokens):
            n = min(block, int(max_tokens) - len(gen))
            if flip:
                side = cur_side
            else:
                side = "elicited" if (draft_side == "elicited" or burst_left > 0) else "target"
            _dids, _vids = (j_ids, t_ids) if side == "elicited" else (t_ids, j_ids)
            try:
                blk = client.gen_block(_dids, n, top_k, draft_temp,
                                       aff + ("-j" if side == "elicited" else "-t"))
                n_calls += 1
                if not blk:
                    break
                ids = [b["id"] for b in blk]
                sc = client.score_block(_vids, ids, top_k)
                n_calls += 1
            except RuntimeError as e:
                truncated = str(e)
                break
            n_blocks += 1

            # Per-position target/elicited views, whichever side drafted.
            if side == "elicited":
                _tl = list(sc["lp"])              # target logprobs: from the verify call
                _jl = [b["lp"] for b in blk]      # elicited logprobs: from the draft call
                _ttop = list(sc["top"])
                _jtop = [b["top"] for b in blk]
            else:
                _tl = [b["lp"] for b in blk]      # target logprobs: from the draft call
                _jl = list(sc["lp"])              # elicited logprobs: from the verify call
                _ttop = [b["top"] for b in blk]
                _jtop = list(sc["top"])

            # Longest acceptable prefix, stopping at a stop token. An elicited draft is
            # accepted while the TARGET finds the token possible; a target draft is accepted
            # while the two contexts still AGREE.
            acc, hit_stop = 0, False
            for i, tok in enumerate(ids):
                if tok in res.stop_ids:
                    hit_stop = True
                    break
                if side == "elicited":
                    if floor > 0.0 and math.exp(_tl[i]) * 100.0 < floor:
                        break
                else:
                    if _q_of(_ttop[i], _jtop[i], metric) >= theta:
                        break
                acc += 1
            for i in range(acc):
                gen.append(ids[i])
                t_lps.append(_tl[i])
                j_lps.append(_jl[i])
            t_ids = t_ids + ids[:acc]
            j_ids = j_ids + ids[:acc]
            n_accept += acc
            if burst_left > 0:
                burst_left = max(0, burst_left - acc)
            if hit_stop:
                break
            if acc == len(ids):
                continue

            # First violation: resolve THIS position from the two top-k lists already in
            # hand, and throw the rest of the block away -- emitting a different token
            # leaves every later draft conditioned on a context that no longer exists.
            n_rewind += 1
            if flip:
                # Hand over: whichever side just failed its own test yields to the other.
                cur_side = "elicited" if side == "target" else "target"
                n_burst += 1
            if burst > 0:
                if burst_left == 0:
                    n_burst += 1
                burst_left = burst
            r = _resolve_one(res, _ttop[acc], _jtop[acc],
                             alpha0, alpha_k, floor, metric, sample_temp, a_int)
            if r is None:
                # The floor left nothing at this position. The target's own top-1 is always
                # admissible and keeps the loop moving.
                ttop = _ttop[acc]
                if not ttop:
                    break
                tid = res.id_of(ttop[0][0])
                if tid is None:
                    n_stall += 1
                    break
                tlp, jlp = ttop[0][1], dict(_jtop[acc]).get(ttop[0][0])
            else:
                tid, tlp, jlp = r
            if tid in res.stop_ids:
                break
            gen.append(tid)
            t_lps.append(tlp if tlp is not None else float("nan"))
            j_lps.append(jlp if jlp is not None else float("nan"))
            t_ids = t_ids + [tid]
            j_ids = j_ids + [tid]
            if burst_left > 0:
                burst_left = max(0, burst_left - 1)

        # A rewind can emit a token outside the target's top-k, whose target logprob is
        # unknown at decode time -- storing NaN there poisoned the whole arithmetic mean.
        # One echo over the finished reply prices every token exactly, for one call per
        # scenario against the ~250 the block loop already spends.
        if gen:
            try:
                _rs = client.score_block(client.prefix_ids(t_prefix), gen, 1)
                n_calls += 1
                if len(_rs["lp"]) == len(gen):
                    t_lps = list(_rs["lp"])
            except RuntimeError:
                pass    # keep the decode-time values; some may be NaN
        return {"best_text": _finalize(res, gen), "best_ids": gen,
                "best_token_probs": [(math.exp(l) * 100 if l == l else float("nan")) for l in t_lps],
                "best_token_probs_jail": [(math.exp(l) * 100 if l == l else None) for l in j_lps],
                "n_calls": n_calls, "n_blocks": n_blocks, "n_accept": n_accept,
                "n_rewind": n_rewind, "n_stall": n_stall, "n_burst": n_burst,
                "truncated": truncated}

    _t0 = time.time()
    jobs = list(enumerate(target_msgs_batch))
    if len(jobs) == 1:
        out = [_one(jobs[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            out = list(ex.map(_one, jobs))
    nc = sum(o.pop("n_calls") for o in out)
    nb = sum(o.pop("n_blocks") for o in out)
    na = sum(o.pop("n_accept") for o in out)
    nr = sum(o.pop("n_rewind") for o in out)
    ns = sum(o.pop("n_stall") for o in out)
    nbu = sum(o.pop("n_burst") for o in out)
    trunc = [o.pop("truncated") for o in out]
    ncut = sum(1 for x in trunc if x)
    nt = sum(len(o["best_ids"]) for o in out)
    msg = ("  [api_spec draft=%s block=%d floor=%g draft_T=%g] %d tokens in %d calls (%.2f tok/call, vs 0.50 for "
           "the single-position rule), %d blocks, %d accepted (%.1f%% of tokens), %d rewinds"
           % (draft_side, block, floor, draft_temp, nt, nc, nt / max(nc, 1), nb, na,
              100.0 * na / max(nt, 1), nr))
    if ns:
        msg += ", %d stalls" % ns
    if burst:
        msg += ", %d bursts (len %d)" % (nbu, burst)
    if flip:
        msg += ", %d flips" % nbu
    if ncut:
        msg += ("  |  %d/%d scenarios CUT SHORT by API failure -- e.g. %s"
                % (ncut, len(out), next(x for x in trunc if x)[:110]))
    print(msg, flush=True)
    _record_cost(client, "spec:%s:block%d" % (draft_side, block),
                 {"secs": round(time.time() - _t0, 2), "gen_tokens": nt, "n_calls": nc,
                  "n_blocks": nb, "n_accept": na, "n_rewind": nr, "n_scenarios": len(out)})
    return out
