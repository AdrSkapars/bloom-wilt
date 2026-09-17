"""Partial-information logit tilt against a LOCAL model: truncate, then combine.

The paper's LogitTilt (wilt._hf_poe_generate) samples from softmax(b1*l_t + b2*l_e) over the
FULL vocabulary. The hosted-API engine (apitilt.py) never sees that -- it gets each context's
top-k logprobs and nothing else. Those two differ in two ways at once: the information
available, and the combination rule. This module separates them, so a sweep can move one at a
time:

  * top_k truncates each context's distribution to its own top-k BEFORE combining.
    top_k=0 means no truncation, so the engine reduces EXACTLY to the paper's LogitTilt --
    that is the sweep's correctness anchor at the top end, not a separate code path.
  * rule chooses the operator. "poe" is the paper's: a logit-space sum, which in probability
    space is a GEOMETRIC mixture p ~ p_t^b1 * p_e^b2. Under truncation that has a specific
    consequence the user should expect: a candidate needs a finite logprob from BOTH sides,
    so the surviving support is the INTERSECTION of the two top-k sets and is usually smaller
    than k, sometimes empty.

Because both full distributions are in hand at every position, the truncated decode can be
scored against the untruncated one for free -- same forward pass, two argmaxes. That is the
measurement this module exists for: `argmax_agree` is a per-TOKEN statistic over thousands of
tokens, so it resolves a difference between two k values that downstream behaviour-presence
(pooled sd 9.25, ~99 scenarios to resolve 10 points) cannot.

Not a paper code path: dispatched only from partial_tilt_output with engine="hf_partial".
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import torch

from . import core
from .wilt import _hf_left_pad

__all__ = ["_driven_hf_partial"]

_NEG_INF = float("-inf")


def _record_stats(tag: str, rec: Dict) -> None:
    """Append one record to <BLOOM_RUNS_ROOT>/<BLOOM_FOLDER>/partial_stats.jsonl.

    A sink rather than a print because the numbers that matter here are per-token rates over
    a whole run, and the interesting comparison is between runs. Failures swallowed: losing a
    statistic must never lose a decode.
    """
    try:
        folder = os.environ.get("BLOOM_FOLDER", "") or ""
        if not folder:
            return
        d = os.path.join(os.environ.get("BLOOM_RUNS_ROOT", "") or "", folder)
        os.makedirs(d, exist_ok=True)
        out = {"t": time.time(), "tag": tag}
        out.update(rec)
        with open(os.path.join(d, "partial_stats.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + chr(10))
    except Exception:
        pass


def _support_mask(tl, cl, top_k: int, rule: str):
    """Boolean [B, V] mask of candidates that survive truncation under `rule`.

    top_k=0 keeps everything, which is what makes k=0 the untruncated anchor rather than a
    branch. For "poe" the survivors are the INTERSECTION: a geometric mixture needs a finite
    logprob from both sides, and a token outside a context's top-k has none. For "mix" the
    survivors are the UNION, since an arithmetic mixture lets one side carry a token alone.
    """
    if top_k <= 0:
        return torch.ones_like(tl, dtype=torch.bool)
    V = tl.shape[-1]
    k = min(int(top_k), V)

    def _top_mask(lg):
        """Boolean mask of the k largest entries per row.

        For k past halfway, take the SMALLEST V-k and invert: torch.topk at k=223488 of
        248320 is effectively a full sort and OOMed a 22GB card, while the complement is a
        topk of 24832 -- same mask, ~9x less work. Exact apart from ties at the boundary,
        which float logits do not produce in practice.
        """
        if k * 2 <= V:
            idx = lg.topk(k, dim=-1).indices
            m = torch.zeros_like(lg, dtype=torch.bool)
            m.scatter_(1, idx, True)
            return m
        idx = lg.topk(V - k, dim=-1, largest=False).indices
        m = torch.ones_like(lg, dtype=torch.bool)
        m.scatter_(1, idx, False)
        return m

    t_keep = _top_mask(tl)
    c_keep = _top_mask(cl)
    return (t_keep | c_keep) if rule == "mix" else (t_keep & c_keep)


def _combine(tl, cl, b1: float, b2: float, rule: str, keep, temperature: float):
    """Scores over the surviving support, as a probability distribution per row.

    "poe"  -- z = b1*l_t + b2*l_e, softmax over the survivors. Renormalising each side over
              its own top-k first would be a no-op here: it shifts each logit vector by a
              per-row constant, and softmax is invariant to that.
    "mix"  -- b1*p_t + b2*p_e in PROBABILITY space, each side renormalised over its own
              surviving mass. Here the renormalisation is NOT a no-op, which is exactly why
              the two rules are different operators rather than two spellings of one.
    """
    # Backstop: an all-False row would make the poe branch softmax a row of -inf and return
    # NaN, which propagates silently into the sampler and the plausibility stats. The decode
    # loop already substitutes the target's top-1 before it gets here, so this only catches a
    # future caller that forgets -- falling back to the untruncated support is defined and
    # sane, where NaN is neither.
    if bool((~keep.any(dim=-1)).any()):
        keep = torch.where(keep.any(dim=-1, keepdim=True), keep, torch.ones_like(keep))
    if rule == "mix":
        pt = torch.softmax(tl, dim=-1)
        pe = torch.softmax(cl, dim=-1)
        pt = torch.where(keep, pt, torch.zeros_like(pt))
        pe = torch.where(keep, pe, torch.zeros_like(pe))
        pt = pt / pt.sum(-1, keepdim=True).clamp_min(1e-12)
        pe = pe / pe.sum(-1, keepdim=True).clamp_min(1e-12)
        w = b1 * pt + b2 * pe
        if temperature > 0 and abs(temperature - 1.0) > 1e-9:
            w = w.clamp_min(0).pow(1.0 / max(temperature, 1e-6))
        return w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    z = b1 * tl + b2 * cl
    z = torch.where(keep, z, torch.full_like(z, _NEG_INF))
    return torch.softmax(z / max(temperature, 1e-6), dim=-1)


def _driven_hf_partial(hf: Dict, jail_runtime_cfg: Dict,
                       target_msgs_batch: List[List[Dict]], max_tokens: int,
                       temperature: float, no_think_target: bool) -> List[Dict]:
    """[partial_tilt_output engine=hf_partial] Truncate both contexts to top-k, then combine.

    Returns one {"best_text","best_ids","best_token_probs"} per scenario, the same shape the
    hf_full jail path returns, so the rollout call site is a one-line branch.

    best_token_probs is the UNMODIFIED target probability of each emitted token, read off the
    same forward pass that produced it -- identical in meaning to every other plausibility
    number in this pipeline, and free.
    """
    mt, mc = hf["mt"], hf["mc"]
    tok, tok_c = hf["tok"], hf["tok_c"]
    device, pad_id, eos_id = hf["device"], hf["pad_id"], hf["eos_id"]

    sys_prompt = jail_runtime_cfg.get("system_prompt", "") or ""
    prefill = jail_runtime_cfg.get("prefill", "") or ""
    b1 = float(jail_runtime_cfg.get("b1") if jail_runtime_cfg.get("b1") is not None else 1.0)
    b2 = float(jail_runtime_cfg.get("b2", 1.0))
    top_k = int(jail_runtime_cfg.get("api_top_k", 0) or 0)
    rule = str(jail_runtime_cfg.get("api_rule", "poe") or "poe")
    if rule not in ("poe", "mix"):
        raise RuntimeError(
            f"partial_tilt_output.rule={rule!r} is not available on engine='hf_partial' "
            f"(poe | mix). 'spec' is a call-cost optimisation with no local analogue, and "
            f"'corner' is a hosted-API construct.")
    # partial_tilt_output.floor is a PERCENT (the hosted engine's unit); the local mask needs
    # a probability. Converting here rather than at the config keeps one unit in the config.
    floor = float(jail_runtime_cfg.get("api_floor", 0.0) or 0.0) / 100.0
    empty_action = str(jail_runtime_cfg.get("hf_empty_action", "target_argmax") or "target_argmax")
    if empty_action not in ("target_argmax", "target_sample"):
        raise RuntimeError(f"partial_tilt_output.hf.empty_action={empty_action!r} unknown "
                           f"(target_argmax | target_sample)")
    measure = bool(jail_runtime_cfg.get("hf_measure_oracle", True))
    greedy = bool(jail_runtime_cfg.get("hf_greedy", False))

    NO_THINK = hf.get("target_no_think", core._NO_THINK_PREFIX)
    NO_THINK_C = hf.get("corrupt_no_think", core._CORRUPT_NO_THINK_PREFIX)

    t_prefs: List[List[int]] = []
    j_prefs: List[List[int]] = []
    for tm in target_msgs_batch:
        ts = tok.apply_chat_template(tm, tokenize=False, add_generation_prompt=True)
        if no_think_target:
            ts += NO_THINK
        t_prefs.append(tok.encode(ts, add_special_tokens=False))
        conv = [m for m in tm if m.get("role") != "system"]
        j_msgs = ([{"role": "system", "content": sys_prompt}] + conv) if sys_prompt else conv
        js = tok_c.apply_chat_template(j_msgs, tokenize=False, add_generation_prompt=True) + NO_THINK_C
        if prefill:
            js += prefill
        j_prefs.append(tok_c.encode(js, add_special_tokens=False))

    B = len(t_prefs)
    gen: List[List[int]] = [[] for _ in range(B)]
    tlps: List[List[float]] = [[] for _ in range(B)]
    # Per-token diagnostics. n_support/n_mass are sums to be averaged at the end; the point of
    # them is that they are per-TOKEN, so a k-sweep is resolvable at one cell rather than needing
    # ~99 scenarios the way a presence difference does.
    n_pos = n_empty = n_agree = 0
    sum_support = 0
    sum_mass = 0.0

    with torch.no_grad():
        ti, ta = _hf_left_pad(t_prefs, pad_id, device)
        ci, ca = _hf_left_pad(j_prefs, pad_id, device)
        to = mt(input_ids=ti, attention_mask=ta, use_cache=True, logits_to_keep=1)
        co = mc(input_ids=ci, attention_mask=ca, use_cache=True, logits_to_keep=1)
        tp, cp = to.past_key_values, co.past_key_values
        tl = to.logits[:, -1, :].float().to(device)
        cl = co.logits[:, -1, :].float().to(device)
        taf, caf = ta, ca
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(int(max_tokens)):
            keep = _support_mask(tl, cl, top_k, rule)
            if floor > 0.0:
                keep = keep & (torch.softmax(tl, dim=-1) >= floor)
            empty = ~keep.any(dim=-1)
            # Size of the TRUE surviving set, taken before the fallback below replaces an
            # empty row with a one-hot. Measured after, an empty position contributes 1
            # instead of 0 and the mean reads as though something survived when nothing did
            # -- at k=1 that showed up as "support 1.00, empty 31%", which cannot both be true.
            support_now = keep.sum(-1)
            # A row with no survivors cannot be sampled from at all, so give it a
            # one-hot admissible support instead of a NaN: the target's own top-1 is
            # always plausible by construction.
            t_am = tl.argmax(dim=-1)
            if bool(empty.any()):
                onehot = torch.zeros_like(keep)
                onehot[torch.arange(B, device=device), t_am] = True
                keep = torch.where(empty.unsqueeze(-1), onehot, keep)

            probs = _combine(tl, cl, b1, b2, rule, keep, temperature)

            if measure:
                # What the SAME rule would have done with no truncation, from the same forward
                # pass: the cost of top-k, isolated from sampling noise by comparing argmaxes.
                full = _combine(tl, cl, b1, b2, rule,
                                torch.ones_like(keep), temperature)
                live = ~done
                n_live = int(live.sum())
                if n_live:
                    agree = (full.argmax(-1) == probs.argmax(-1)) & live
                    n_agree += int(agree.sum())
                    sum_mass += float((torch.where(keep, full, torch.zeros_like(full))
                                       .sum(-1) * live).sum())
                    sum_support += int((support_now * live).sum())
                    n_empty += int((empty & live).sum())
                    n_pos += n_live

            if greedy:
                nxt = probs.argmax(dim=-1)
            else:
                nxt = torch.multinomial(probs, 1).squeeze(-1)
            if bool(empty.any()) and empty_action == "target_sample":
                draw = torch.multinomial(torch.softmax(tl, dim=-1), 1).squeeze(-1)
                nxt = torch.where(empty, draw, nxt)

            # Plausibility is the UNMODIFIED target logprob, never the tilted one.
            tlp = tl.gather(-1, nxt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(tl, dim=-1)
            live = (~done) & (nxt != eos_id)
            for i in range(B):
                if bool(live[i]):
                    tlps[i].append(float(tlp[i]))

            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            for i in range(B):
                if not done[i]:
                    gen[i].append(int(nxt[i]))
            done = done | (nxt == eos_id)
            if bool(done.all()):
                break

            ones = torch.ones(B, 1, dtype=torch.long, device=device)
            taf = torch.cat([taf, ones], -1)
            caf = torch.cat([caf, ones], -1)
            to = mt(input_ids=nxt.unsqueeze(-1), attention_mask=taf,
                    past_key_values=tp, use_cache=True)
            co = mc(input_ids=nxt.unsqueeze(-1), attention_mask=caf,
                    past_key_values=cp, use_cache=True)
            tp, cp = to.past_key_values, co.past_key_values
            tl = to.logits[:, -1, :].float().to(device)
            cl = co.logits[:, -1, :].float().to(device)

    if n_pos:
        _record_stats("batch", {
            "rule": rule, "top_k": top_k, "b1": b1, "b2": b2,
            "positions": n_pos,
            "mean_support": round(sum_support / n_pos, 4),
            # Records written before this flag counted an empty position as support 1, so a
            # reader must subtract empty_rate from mean_support to compare them with these.
            "support_excludes_empty": True,
            "empty_rate": round(n_empty / n_pos, 6),
            "argmax_agree": round(n_agree / n_pos, 6),
            "mean_mass_kept": round(sum_mass / n_pos, 6),
        })
        print(f"  [hf_partial] rule={rule} k={top_k or 'full'} "
              f"support={sum_support / n_pos:.2f} empty={100.0 * n_empty / n_pos:.2f}% "
              f"agree={100.0 * n_agree / n_pos:.2f}% mass={100.0 * sum_mass / n_pos:.2f}%",
              flush=True)

    out: List[Dict] = []
    for i in range(B):
        ids = [x for x in gen[i] if x != eos_id and x != pad_id]
        out.append({
            "best_text": tok.decode(ids, skip_special_tokens=True).strip(),
            "best_ids": ids,
            "best_token_probs": [math.exp(l) * 100.0 for l in tlps[i]],
        })
    return out
