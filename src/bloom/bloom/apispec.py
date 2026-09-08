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


def _resolve_one(res, t_top, j_top, alpha0, alpha_k, floor, metric, sample_temp):
    """The single-position mixture rule, over the union of two top-k lists already in hand.

    Returns (token_id, target_lp, elicited_lp), or None if the floor leaves no candidate.
    """
    tmap = dict(t_top)
    jmap = dict(j_top)
    q = _q_of(t_top, j_top, metric)
    alpha = max(alpha0 * (1.0 - q ** alpha_k), 1e-9)
    union = {}
    for s, lp in t_top:
        union[s] = [math.exp(lp), 0.0]
    for s, lp in j_top:
        union.setdefault(s, [0.0, 0.0])[1] = math.exp(lp)
    cands = [(s, alpha * xy[0] + (1.0 - alpha) * xy[1]) for s, xy in union.items()]
    if floor > 0.0:
        # Only target-side members can be priced for free here; an elicited-only candidate
        # would need a call, and avoiding calls is the point of this path. The free bound
        # still applies: if the target's k-th best fails the floor, nothing outside its
        # top-k can clear it either, so those go too.
        tmin = (min(math.exp(v) for v in tmap.values()) * 100.0) if tmap else 0.0
        keep = []
        for s, w in cands:
            if s in tmap:
                if math.exp(tmap[s]) * 100.0 >= floor:
                    keep.append((s, w))
            elif tmin >= floor:
                keep.append((s, w))
        cands = keep
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
        n_calls = n_blocks = n_accept = n_rewind = n_stall = 0
        truncated = ""
        while len(gen) < int(max_tokens):
            n = min(block, int(max_tokens) - len(gen))
            try:
                blk = client.gen_block(j_ids, n, top_k, temperature, aff + "-j")
                n_calls += 1
                if not blk:
                    break
                ids = [b["id"] for b in blk]
                sc = client.score_block(t_ids, ids, top_k)
                n_calls += 1
            except RuntimeError as e:
                truncated = str(e)
                break
            n_blocks += 1

            # Longest prefix clearing the floor, stopping at a stop token.
            acc, hit_stop = 0, False
            for i, tok in enumerate(ids):
                if tok in res.stop_ids:
                    hit_stop = True
                    break
                if floor > 0.0 and math.exp(sc["lp"][i]) * 100.0 < floor:
                    break
                acc += 1
            for i in range(acc):
                gen.append(ids[i])
                t_lps.append(sc["lp"][i])
                j_lps.append(blk[i]["lp"])
            t_ids = t_ids + ids[:acc]
            j_ids = j_ids + ids[:acc]
            n_accept += acc
            if hit_stop:
                break
            if acc == len(ids):
                continue

            # First violation: resolve THIS position from the two top-k lists already in
            # hand, and throw the rest of the block away -- emitting a different token
            # leaves every later draft conditioned on a context that no longer exists.
            n_rewind += 1
            r = _resolve_one(res, sc["top"][acc], blk[acc]["top"],
                             alpha0, alpha_k, floor, metric, sample_temp)
            if r is None:
                # The floor left nothing at this position. The target's own top-1 is always
                # admissible and keeps the loop moving.
                ttop = sc["top"][acc]
                if not ttop:
                    break
                tid = res.id_of(ttop[0][0])
                if tid is None:
                    n_stall += 1
                    break
                tlp, jlp = ttop[0][1], dict(blk[acc]["top"]).get(ttop[0][0])
            else:
                tid, tlp, jlp = r
            if tid in res.stop_ids:
                break
            gen.append(tid)
            t_lps.append(tlp if tlp is not None else float("nan"))
            j_lps.append(jlp if jlp is not None else float("nan"))
            t_ids = t_ids + [tid]
            j_ids = j_ids + [tid]

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
                "n_rewind": n_rewind, "n_stall": n_stall, "truncated": truncated}

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
    trunc = [o.pop("truncated") for o in out]
    ncut = sum(1 for x in trunc if x)
    nt = sum(len(o["best_ids"]) for o in out)
    msg = ("  [api_spec block=%d floor=%g] %d tokens in %d calls (%.2f tok/call, vs 0.50 for "
           "the single-position rule), %d blocks, %d accepted (%.1f%% of tokens), %d rewinds"
           % (block, floor, nt, nc, nt / max(nc, 1), nb, na, 100.0 * na / max(nt, 1), nr))
    if ns:
        msg += ", %d stalls" % ns
    if ncut:
        msg += ("  |  %d/%d scenarios CUT SHORT by API failure -- e.g. %s"
                % (ncut, len(out), next(x for x in trunc if x)[:110]))
    print(msg, flush=True)
    _record_cost(client, "spec:block%d" % block,
                 {"secs": round(time.time() - _t0, 2), "gen_tokens": nt, "n_calls": nc,
                  "n_blocks": nb, "n_accept": na, "n_rewind": nr, "n_scenarios": len(out)})
    return out
