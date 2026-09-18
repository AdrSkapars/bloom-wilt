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
import re
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


def _side_masks(tl, cl, top_k: int):
    """Per-side top-k masks, (target, elicited).

    top_k=0 keeps everything, which is what makes k=0 the untruncated anchor rather than a
    branch. For "poe" the survivors are the INTERSECTION: a geometric mixture needs a finite
    logprob from both sides, and a token outside a context's top-k has none. For "mix" the
    survivors are the UNION, since an arithmetic mixture lets one side carry a token alone.
    """
    if top_k <= 0:
        ones = torch.ones_like(tl, dtype=torch.bool)
        return (ones, ones)
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
    return (t_keep, c_keep)


def _support_mask(tl, cl, top_k: int, rule: str):
    """Boolean [B, V] mask of candidates that survive truncation under `rule`."""
    t_keep, c_keep = _side_masks(tl, cl, top_k)
    # "poe" needs a finite logprob from BOTH sides, so only the intersection survives.
    # "poe_union" and "mix" let a candidate through on one side alone -- they differ in what
    # the absent side then contributes, not in which candidates are eligible.
    return (t_keep & c_keep) if rule == "poe" else (t_keep | c_keep)


_TOPM = re.compile(r"^top([0-9]+)_(disjoint|tv|outside)$")


def _q_of(tl, cl, t_keep, c_keep, metric: str):
    """Per-row disagreement q in [0, 1], from the TRUNCATED sides.

    Same three measures as the hosted engine, so a local number is comparable to an API one.
    Each side is renormalised over its own surviving mass, which is what makes these
    distributions rather than arbitrary sums.

    Note "elicited_outside" DEGENERATES at top_k=0: with no truncation nothing is outside the
    target's set, so q is identically 0 and alpha never moves off alpha0. tv and margin stay
    meaningful at full vocab.
    """
    pt = torch.softmax(tl, dim=-1)
    pe = torch.softmax(cl, dim=-1)
    pt = torch.where(t_keep, pt, torch.zeros_like(pt))
    pe = torch.where(c_keep, pe, torch.zeros_like(pe))
    pt = pt / pt.sum(-1, keepdim=True).clamp_min(1e-12)
    pe = pe / pe.sum(-1, keepdim=True).clamp_min(1e-12)
    if metric == "tv":
        q = 0.5 * (pt - pe).abs().sum(-1)
    elif metric == "jsd":
        # Jensen-Shannon, divided by ln2 so it lands in [0,1]: 0 when the two truncated
        # distributions coincide, 1 when their supports are disjoint. Symmetric, and unlike a
        # plain KL it stays finite when one side assigns zero to what the other wants -- which
        # is the common case here, so KL would be infinite at most positions.
        m = 0.5 * (pt + pe)

        def _kl(a, b):
            return (a * (torch.log(a.clamp_min(1e-12)) - torch.log(b.clamp_min(1e-12)))).sum(-1)

        q = (0.5 * _kl(pt, m) + 0.5 * _kl(pe, m)) / math.log(2.0)
    elif metric == "hellinger":
        # sqrt(1 - BC), BC = sum sqrt(p_t * p_e) the Bhattacharyya coefficient. Apt for a
        # PRODUCT rule specifically: BC is exactly the normalising mass of the geometric
        # mixture at equal weights, so this measures how far the poe product collapses. The
        # disagreement measure matched to the operator rather than borrowed from elsewhere.
        bc = (pt.clamp_min(0) * pe.clamp_min(0)).sqrt().sum(-1)
        q = (1.0 - bc).clamp_min(0.0).sqrt()
    elif metric == "top1_mismatch":
        # The crudest measure available: 1 when the two sides disagree about the single most
        # likely token, 0 otherwise, so mean_q reads directly as "fraction of positions where
        # the argmaxes differ". A control. If this drives the schedule as well as jsd or
        # hellinger, the fine structure of the disagreement is not what matters and the
        # continuous metrics are buying nothing.
        q = (pt.argmax(-1) != pe.argmax(-1)).float()
    elif _TOPM.match(metric):
        # The top-m FAMILY: everything here measures disagreement inside a window of m
        # candidates per side, and nothing here cares what the combine rule truncates to. That
        # separation is the point -- the window is the MEASUREMENT scale, so the beta schedule
        # can be driven by partial information while the logits are still combined in full.
        #
        #   top{m}_disjoint  1 when the two top-m sets share no candidate, else 0.
        #   top{m}_tv        total variation between the two top-m distributions, each
        #                    renormalised over its own window. Smooth, and EXACTLY 1 when the
        #                    sets are disjoint -- so _disjoint is this thresholded at 1, not a
        #                    different idea.
        #   top{m}_outside   the elicited window's mass on candidates the target's window did
        #                    not propose. The hosted engine's elicited_outside with the window
        #                    pinned at m, which is what stops it degenerating: the plain
        #                    metric measures mass outside a set that GROWS with top_k, so it
        #                    goes to 0 as truncation relaxes (0.17 -> 0.0000 over the sweep).
        _mo = _TOPM.match(metric)
        m, kind = int(_mo.group(1)), _mo.group(2)
        tv_, ti = pt.topk(m, dim=-1)
        ev_, ei = pe.topk(m, dim=-1)
        if kind == "disjoint":
            # Only count a shared index if BOTH sides actually put mass on it. Under truncation
            # a row can have fewer than m survivors and topk pads with zero-probability
            # entries, whose indices would otherwise manufacture an overlap out of padding.
            both = (tv_.unsqueeze(-1) > 0) & (ev_.unsqueeze(-2) > 0)
            shared = ((ti.unsqueeze(-1) == ei.unsqueeze(-2)) & both).any(-1).any(-1)
            q = (~shared).float()
        else:
            # Scatter each window back to vocab width so the two can be compared token by
            # token: the windows hold DIFFERENT tokens, so an elementwise op on the topk
            # outputs would be comparing the target's 3rd choice against the elicited side's
            # 3rd choice, which are unrelated.
            zt = torch.zeros_like(pt).scatter(-1, ti, tv_)
            ze = torch.zeros_like(pe).scatter(-1, ei, ev_)
            zt = zt / zt.sum(-1, keepdim=True).clamp_min(1e-12)
            ze = ze / ze.sum(-1, keepdim=True).clamp_min(1e-12)
            if kind == "tv":
                q = 0.5 * (zt - ze).abs().sum(-1)
            else:   # outside
                # Membership from the INDICES, not from zt > 0: a tail token inside the window
                # can underflow to exactly 0 in float32 and would then read as "outside".
                t_in = torch.zeros_like(pt, dtype=torch.bool).scatter(
                    -1, ti, torch.ones_like(ti, dtype=torch.bool))
                q = torch.where(t_in, torch.zeros_like(ze), ze).sum(-1)
    elif metric == "target_top_gap":
        # The target's OWN top-1, scored under both distributions: how much probability the
        # elicited context withholds from the token the target most wants.
        #     q = phat_t(t_top) - phat_e(t_top)
        # The only cross-distribution measure here -- every other one compares each side's
        # pick within its own distribution. Graded, so unlike top1_mismatch (a hard switch
        # where 1**kappa = 1 makes kappa inert) the exponent is a live dial again. Clamped at
        # 0 for the case where the elicited side likes the target's pick MORE than the target
        # does, which is agreement, not negative disagreement.
        t_top = pt.argmax(-1)
        idx = torch.arange(pt.shape[0], device=pt.device)
        q = (pt[idx, t_top] - pe[idx, t_top]).clamp_min(0.0)
    elif metric == "margin":
        # What the ELICITED side gains by getting its way here, rather than how far apart the
        # two distributions are overall.
        e_top = pe.argmax(-1)
        t_top = pt.argmax(-1)
        idx = torch.arange(pe.shape[0], device=pe.device)
        q = (pe[idx, e_top] - pe[idx, t_top]).clamp_min(0.0)
    elif not metric or metric == "elicited_outside":
        q = torch.where(t_keep, torch.zeros_like(pe), pe).sum(-1)
    else:
        # NOT a silent fallback. A typo in BLOOM_API_Q_METRIC used to land in the
        # elicited_outside branch and produce a run that looked perfectly healthy: at
        # top_k=0 that metric is identically 0, so alpha never leaves alpha0 and the output
        # is indistinguishable from a deliberate fixed-beta run.
        raise ValueError("unknown q metric %r" % (metric,))
    return q.clamp(0.0, 1.0)


def _alpha_of(q, alpha0: float, alpha_k: float, q_ref: float = 1.0):
    """Target weight per position: alpha = alpha0 * (1 - min(1, q/q_ref)**alpha_k).

    q_ref exists because the unscaled form is bounded below by alpha0*(1 - max q). A metric
    whose q never approaches 1 therefore never approaches alpha=0, so it nudges every position
    a little and hands over none of them. Measured on racial: top5_tv averages q ~ 0.20 and the
    whole alpha_k range 1..10 moved mean alpha only between 0.184 and 0.219, against alpha0 =
    0.222. The binary metrics beat it for exactly this reason -- their q is 1 where they fire,
    so alpha reaches 0 and the elicited context takes the position outright.

    Capping q/q_ref at 1 gives a graded metric that same reach: at or above q_ref it is full
    handover, below it stays proportional. q_ref = 1.0 is the identity, so the default path is
    bit-identical to before this existed.

    The result is clamped off exactly 0 so the target's own ordering still breaks ties among
    candidates the elicited side never proposed.
    """
    qe = (q / q_ref).clamp(max=1.0) if q_ref != 1.0 else q
    return (alpha0 * (1.0 - qe ** alpha_k)).clamp_min(1e-9)


def _combine(tl, cl, b1, b2, rule: str, keep, temperature: float,
             t_keep=None, c_keep=None):
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
    # b1/b2 arrive as scalars for a fixed weighting and as [B] tensors under adaptive alpha;
    # unsqueeze so both broadcast against [B, V] the same way.
    if torch.is_tensor(b1):
        b1 = b1.unsqueeze(-1)
    if torch.is_tensor(b2):
        b2 = b2.unsqueeze(-1)
    if rule == "poe_union" and t_keep is not None and c_keep is not None:
        # A candidate the other side did not propose keeps its own side's logit and has ZERO
        # ADDED for the missing one, rather than being dropped. Same arithmetic as "poe"
        # wherever both sides proposed, so it differs only on the union minus the intersection.
        #
        # CAVEAT, and it is not small: a logit has no absolute scale -- softmax is invariant to
        # adding a constant to every logit -- so "0" is an arbitrary reference point, and this
        # rule is the only one here that is NOT shift-invariant. Adding c to every target logit
        # leaves "poe" and "mix" identical and moves this one. Whether 0 sits high or low among
        # a model's tail logits is a property of the checkpoint, not of the method.
        z = b1 * torch.where(t_keep, tl, torch.zeros_like(tl))             + b2 * torch.where(c_keep, cl, torch.zeros_like(cl))
    else:
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
    if rule not in ("poe", "poe_union", "mix"):
        raise RuntimeError(
            f"partial_tilt_output.rule={rule!r} is not available on engine='hf_partial' "
            f"(poe | poe_union | mix). 'spec' is a call-cost optimisation with no local "
            f"analogue, and "
            f"'corner' is a hosted-API construct.")
    # partial_tilt_output.floor is a PERCENT (the hosted engine's unit); the local mask needs
    # a probability. Converting here rather than at the config keeps one unit in the config.
    floor = float(jail_runtime_cfg.get("api_floor", 0.0) or 0.0) / 100.0
    empty_action = str(jail_runtime_cfg.get("hf_empty_action", "target_argmax") or "target_argmax")
    if empty_action not in ("target_argmax", "target_sample"):
        raise RuntimeError(f"partial_tilt_output.hf.empty_action={empty_action!r} unknown "
                           f"(target_argmax | target_sample)")
    # ADAPTIVE alpha(q). Replaces the fixed pair with alpha = alpha0*(1 - q**kappa), used as
    # (b1, b2) = (alpha, 1-alpha). Only the RATIO matters to poe, so this is the same family
    # with its redundant degree of freedom removed: alpha0 = b1/(b1+b2), i.e. the fixed
    # b1=1.0/b2=1.5 used everywhere above is alpha0 = 0.4. At q=1 alpha=0 and the position is
    # resolved by the elicited context alone.
    adaptive = bool(jail_runtime_cfg.get("api_adaptive", False))
    gain = b1 + b2          # see the GAIN note in the decode loop
    alpha0 = float(jail_runtime_cfg.get("api_alpha0", 0.4))
    alpha_k = float(jail_runtime_cfg.get("api_alpha_k", 10.0) or 10.0)
    q_metric = str(jail_runtime_cfg.get("api_q_metric", "elicited_outside") or "elicited_outside")
    # NOT `or 1.0`: that idiom maps a deliberate 0.0 to the identity, so a q_ref the user
    # meant as "hand over everywhere" would silently become "never rescale". Absent is 1.0;
    # zero is an error and says so.
    _qr = jail_runtime_cfg.get("api_q_ref", 1.0)
    q_ref = 1.0 if _qr is None else float(_qr)
    _METRICS = ("elicited_outside", "tv", "margin", "jsd", "hellinger", "top1_mismatch",
                "target_top_gap")
    # topM_disjoint is a FAMILY, not a name -- top1_disjoint, top2_disjoint, ... -- so
    # membership has to be parsed. m must be a positive integer: "topfive_disjoint" would
    # otherwise slip through to the decode and die per-position instead of here, at startup.
    _m_ok = bool(_TOPM.match(q_metric)) and int(_TOPM.match(q_metric).group(1)) >= 1
    if q_metric not in _METRICS and not _m_ok:
        # The old message named three of the seven metrics it accepted, which is worse than
        # useless when the entire job of the error is to say what you may write instead.
        raise RuntimeError(f"partial_tilt_output.mix.q_metric={q_metric!r} unknown "
                           f"({' | '.join(_METRICS)} | topM_disjoint|tv|outside)")
    if adaptive and not (0.0 <= alpha0 <= 1.0):
        raise RuntimeError(f"partial_tilt_output.mix.alpha0={alpha0!r} must be in [0, 1]")
    if not (0.0 < q_ref <= 1.0):
        raise RuntimeError(f"partial_tilt_output.mix.q_ref={q_ref!r} must be in (0, 1]")
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
    sum_q = sum_alpha = 0.0

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
            t_keep, c_keep = _side_masks(tl, cl, top_k)
            keep = (t_keep & c_keep) if rule == "poe" else (t_keep | c_keep)
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

            if adaptive:
                q = _q_of(tl, cl, t_keep, c_keep, q_metric)
                # RESCALE q before the exponent sees it. alpha = alpha0*(1 - q**kappa) is
                # bounded below by alpha0*(1 - max q), so a metric whose q never approaches 1
                # can never approach alpha=0 -- it nudges everywhere and hands over control
                # nowhere. Measured on racial: top5_tv averages q~0.20, and the whole kappa
                # range 1..10 moved mean alpha only between 0.184 and 0.219 against an alpha0
                # of 0.222. That is why the binary metrics win: q hits 1 EXACTLY where they
                # fire, alpha goes to 0, and the elicited context takes the position outright.
                #
                # Dividing by q_ref and capping at 1 gives a smooth metric the same reach:
                # q >= q_ref is full handover, below it stays graded. q_ref=1.0 is the
                # identity, so every existing run and the whole default path are unchanged.
                a = _alpha_of(q, alpha0, alpha_k, q_ref)
                # GAIN. (alpha, 1-alpha) fixes the RATIO but also fixes the SUM at 1, and the
                # sum is an inverse temperature: softmax(2.5*z) is sharper than softmax(z).
                # Only an argmax is scale-free, and this decode samples. Rescaling by b1+b2
                # makes alpha0 = b1/(b1+b2) reproduce the fixed pair exactly at q=0, so an
                # adaptive-vs-fixed comparison isolates the schedule instead of confounding it
                # with a temperature change -- the same trap as the unnormalised beta in the
                # original LogitTilt.
                w1, w2 = a * gain, (1.0 - a) * gain
                sum_q += float((q * (~done)).sum())
                sum_alpha += float((a * (~done)).sum())
            else:
                w1, w2 = b1, b2
            probs = _combine(tl, cl, w1, w2, rule, keep, temperature, t_keep, c_keep)

            if measure:
                # What the SAME rule would have done with no truncation, from the same forward
                # pass: the cost of top-k, isolated from sampling noise by comparing argmaxes.
                _all = torch.ones_like(keep)
                full = _combine(tl, cl, w1, w2, rule, _all, temperature, _all, _all)
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
            "adaptive": adaptive,
            "alpha0": alpha0 if adaptive else None,
            "alpha_k": alpha_k if adaptive else None,
            "q_metric": q_metric if adaptive else None,
            "q_ref": q_ref if adaptive else None,
            # Both are sums over LIVE rows divided by live positions, so they are the mean q
            # and mean alpha actually applied -- the thing to look at when a schedule does
            # nothing because q never got near 1.
            "mean_q": round(sum_q / n_pos, 6) if adaptive else None,
            "mean_alpha": round(sum_alpha / n_pos, 6) if adaptive else None,
            "gain": gain if adaptive else None,
        })
        if adaptive:
            print(f"  [hf_partial] adaptive a0={alpha0} k={alpha_k} metric={q_metric}"
                  + (f" q_ref={q_ref}" if q_ref != 1.0 else "") + " "
                  f"mean_q={sum_q / n_pos:.4f} mean_alpha={sum_alpha / n_pos:.4f}", flush=True)
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
