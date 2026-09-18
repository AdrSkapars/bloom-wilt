#!/usr/bin/env python3
"""CPU tests for the hf_partial engine. No GPU, no weights, no API.

The engine cannot be exercised end to end from the Windows laptop (no CUDA, no local Qwen),
so these pin down the parts that are actually new -- the truncation mask, the two combine
rules, and the decode loop's control flow -- against a stub model. The KV-cache plumbing
around them is lifted from wilt._hf_poe_generate, which is exercised by every paper run.

The load-bearing test is `test_k0_is_logittilt`: at top_k=0 the engine must reproduce the
paper's operator EXACTLY, because the whole sweep is calibrated against that anchor. If it
drifts, every k>0 number is measured from a moving baseline.

  python -X utf8 experiments/postpaper/partial_tilt/test_hftilt.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))

from bloom.bloom.hftilt import _alpha_of, _combine, _q_of, _support_mask  # noqa: E402

B, V = 3, 50
torch.manual_seed(0)
TL = torch.randn(B, V)
CL = torch.randn(B, V)
FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  -- " + detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def test_k0_is_logittilt():
    """top_k=0 + rule=poe must equal softmax((b1*l_t + b2*l_e)/T) over the full vocab."""
    for b1, b2, T in ((1.0, 1.5, 1.0), (1.0, 4.0, 0.7), (0.0, 1.0, 1.0)):
        keep = _support_mask(TL, CL, 0, "poe")
        got = _combine(TL, CL, b1, b2, "poe", keep, T)
        want = torch.softmax((b1 * TL + b2 * CL) / T, dim=-1)
        check("k=0 poe == LogitTilt (b1=%g b2=%g T=%g)" % (b1, b2, T),
              torch.allclose(got, want, atol=1e-6),
              "max|diff|=%.2e" % (got - want).abs().max())


def test_poe_support_is_intersection():
    for k in (1, 3, 5, 10):
        keep = _support_mask(TL, CL, k, "poe")
        t_idx = set(TL[0].topk(k).indices.tolist())
        c_idx = set(CL[0].topk(k).indices.tolist())
        want = t_idx & c_idx
        got = set(keep[0].nonzero().flatten().tolist())
        check("poe support is the intersection (k=%d, |S|=%d)" % (k, len(want)), got == want)
        check("poe support <= k (k=%d)" % k, int(keep.sum(-1).max()) <= k)


def test_mix_support_is_union():
    k = 4
    keep = _support_mask(TL, CL, k, "mix")
    want = set(TL[0].topk(k).indices.tolist()) | set(CL[0].topk(k).indices.tolist())
    check("mix support is the union", set(keep[0].nonzero().flatten().tolist()) == want)


def test_renormalisation_is_a_noop_for_poe():
    """The comment in _combine claims per-side renormalisation cannot change a poe result.

    Worth testing rather than asserting: it is true only because a softmax is invariant to a
    per-row constant shift, and that is exactly the property that FAILS for the mix rule.
    """
    keep = _support_mask(TL, CL, 6, "poe")
    base = _combine(TL, CL, 1.0, 2.0, "poe", keep, 1.0)
    shifted = _combine(TL - TL.logsumexp(-1, keepdim=True),
                       CL - CL.logsumexp(-1, keepdim=True), 1.0, 2.0, "poe", keep, 1.0)
    check("poe is shift-invariant (renormalising each side is a no-op)",
          torch.allclose(base, shifted, atol=1e-6),
          "max|diff|=%.2e" % (base - shifted).abs().max())
    # ... and that the same shift DOES move the arithmetic rule, i.e. they really differ.
    kmix = _support_mask(TL, CL, 6, "mix")
    m1 = _combine(TL, CL, 1.0, 2.0, "mix", kmix, 1.0)
    m2 = _combine(TL, CL, 1.0, 9.0, "mix", kmix, 1.0)
    check("mix weights actually change the mix distribution", not torch.allclose(m1, m2, atol=1e-4))


def test_probabilities_live_only_on_the_support():
    # k=10 so every row has a non-empty intersection: the "zero mass off-support" contract
    # only applies where there IS a support. Rows with none are the backstop's job, tested
    # separately below.
    # Smallest k where every row has a survivor, found rather than guessed: on random logits
    # over V=50 the top-k sets stay disjoint for a surprisingly long way up.
    for k in range(2, V + 1):
        keep = _support_mask(TL, CL, k, "poe")
        if bool(keep.any(-1).all()):
            break
    assert bool(keep.any(-1).all()), "no k gives every row a survivor"
    p = _combine(TL, CL, 1.0, 1.0, "poe", keep, 1.0)
    check("poe puts zero mass off-support", float(p[~keep].abs().max()) < 1e-9)
    check("poe rows sum to 1", torch.allclose(p.sum(-1), torch.ones(B), atol=1e-5))

    # Backstop: a row with no survivors must come back as a real distribution, not NaN --
    # silent NaN here would poison the sampler and every plausibility number downstream.
    empty = torch.zeros_like(keep)
    empty[1] = keep[1]                       # row 0 and 2 have no support at all
    pe = _combine(TL, CL, 1.0, 1.0, "poe", empty, 1.0)
    check("empty support falls back to a valid distribution, not NaN",
          bool(torch.isfinite(pe).all()) and torch.allclose(pe.sum(-1), torch.ones(B), atol=1e-5))
    kmix = _support_mask(TL, CL, 3, "mix")
    q = _combine(TL, CL, 1.0, 1.0, "mix", kmix, 1.0)
    check("mix puts zero mass off-support", float(q[~kmix].abs().max()) < 1e-9)
    check("mix rows sum to 1", torch.allclose(q.sum(-1), torch.ones(B), atol=1e-5))


def test_empty_intersection_is_reachable():
    """Disjoint top-k sets must produce an empty poe support -- the case the engine has to
    handle with hf.empty_action, and the one the k-sweep is expected to hit at small k."""
    tl = torch.full((1, 10), -10.0)
    cl = torch.full((1, 10), -10.0)
    tl[0, :2] = torch.tensor([5.0, 4.0])      # target wants 0, 1
    cl[0, 8:] = torch.tensor([5.0, 4.0])      # elicited wants 8, 9
    keep = _support_mask(tl, cl, 2, "poe")
    check("disjoint top-k gives an empty poe support", int(keep.sum()) == 0)
    check("the same position is non-empty under mix", int(_support_mask(tl, cl, 2, "mix").sum()) == 4)


def test_poe_union_zero_fills_the_missing_side():
    """poe_union keeps the union and ADDS ZERO for whichever side did not propose a token."""
    from bloom.bloom.hftilt import _side_masks
    # A k whose intersection is non-empty in every row: the second assertion below compares
    # the two rules ON the intersection, and at small k over random logits there isn't one.
    for k in range(2, V + 1):
        t_keep, c_keep = _side_masks(TL, CL, k)
        if bool((t_keep & c_keep).any(-1).all()):
            break
    keep = t_keep | c_keep
    got = _combine(TL, CL, 1.0, 2.0, "poe_union", keep, 1.0, t_keep, c_keep)
    want_z = 1.0 * torch.where(t_keep, TL, torch.zeros_like(TL))         + 2.0 * torch.where(c_keep, CL, torch.zeros_like(CL))
    want = torch.softmax(torch.where(keep, want_z, torch.full_like(want_z, float("-inf"))), -1)
    check("poe_union = zero-filled logit sum over the union",
          torch.allclose(got, want, atol=1e-6))

    # On the INTERSECTION the two rules must agree exactly: both sides contributed a real
    # logit there, so zero-filling never applied. Only the union-minus-intersection differs.
    both = t_keep & c_keep
    p_poe = _combine(TL, CL, 1.0, 2.0, "poe", both, 1.0)
    # PER ROW. poe_union leaks mass to the union-only candidates, and each row leaks a
    # different amount, so a single renormalisation across the whole tensor compares rows
    # against each other rather than each row against itself.
    worst = 0.0
    for i in range(B):
        m = both[i]
        a = p_poe[i][m]; a = a / a.sum()
        c = got[i][m]; c = c / c.sum()
        worst = max(worst, float((a - c).abs().max()))
    check("poe and poe_union rank the intersection identically",
          worst < 1e-5, "max|diff|=%.2e" % worst)

    # The shift-invariance that holds for poe must FAIL here -- that is the cost of zero-fill,
    # and it should be visible in a test rather than only in a comment.
    shifted = _combine(TL + 3.0, CL, 1.0, 2.0, "poe_union", keep, 1.0, t_keep, c_keep)
    check("poe_union is NOT shift-invariant (0 is an arbitrary reference)",
          not torch.allclose(got, shifted, atol=1e-4))
    p1 = _combine(TL, CL, 1.0, 2.0, "poe", both, 1.0)
    p2 = _combine(TL + 3.0, CL, 1.0, 2.0, "poe", both, 1.0)
    check("  ...while poe still is", torch.allclose(p1, p2, atol=1e-6))


def test_alpha0_matches_the_fixed_weights_at_q0():
    """alpha0 = b1/(b1+b2), so alpha0=0.4 must reproduce b1=1.0,b2=1.5 wherever q=0.

    This is the claim the whole adaptive-vs-fixed comparison rests on: if the schedule does
    not start from the fixed run's operating point, a difference between them cannot be
    attributed to the schedule.
    """
    keep = _support_mask(TL, CL, 0, "poe")
    fixed = _combine(TL, CL, 1.0, 1.5, "poe", keep, 1.0)
    gain = 1.0 + 1.5
    adapt = _combine(TL, CL, 0.4 * gain, 0.6 * gain, "poe", keep, 1.0)
    check("alpha0=0.4 x gain == fixed b1=1.0/b2=1.5 for poe",
          torch.allclose(fixed, adapt, atol=1e-6),
          "max|diff|=%.2e" % (fixed - adapt).abs().max())

    # WITHOUT the gain the two differ, and not slightly: (alpha, 1-alpha) sums to 1 while the
    # fixed pair sums to 2.5, and that sum is an inverse temperature. Asserting the failure
    # keeps the reason for the gain from being quietly dropped later.
    nogain = _combine(TL, CL, 0.4, 0.6, "poe", keep, 1.0)
    check("  ...and WITHOUT the gain they differ (sum = inverse temperature)",
          not torch.allclose(fixed, nogain, atol=1e-3))

    # ... and that per-row tensor weights broadcast the same as scalars
    b1 = torch.full((B,), 0.4 * gain)
    b2 = torch.full((B,), 0.6 * gain)
    check("per-row tensor weights match scalar weights",
          torch.allclose(_combine(TL, CL, b1, b2, "poe", keep, 1.0), adapt, atol=1e-6))


def test_q_is_bounded_and_responds():
    """q must land in [0,1] under every metric, and reach 1 when the sides are disjoint."""
    from bloom.bloom.hftilt import _q_of, _side_masks
    tl = torch.full((1, 10), -10.0); cl = torch.full((1, 10), -10.0)
    tl[0, :2] = torch.tensor([5.0, 4.0])
    cl[0, 8:] = torch.tensor([5.0, 4.0])
    t_keep, c_keep = _side_masks(tl, cl, 2)
    for m in ("elicited_outside", "tv"):
        q = _q_of(tl, cl, t_keep, c_keep, m)
        check("q = 1 on disjoint sides (%s)" % m,
              0.0 <= float(q[0]) <= 1.0 and float(q[0]) > 0.99, "q=%.4f" % float(q[0]))
    # margin does NOT reach 1 on disjoint sides, and that is correct rather than a bug: it is
    # p_e(elicited top-1) - p_e(target top-1), so it is capped by how PEAKED the elicited
    # distribution is. Here the elicited side splits 0.73/0.27 over two tokens, so the most it
    # can report is 0.73. Consequence for the schedule: under margin, alpha never reaches 0
    # unless the elicited context is also very confident -- a different lever from tv, not a
    # rescaling of it.
    qm = float(_q_of(tl, cl, t_keep, c_keep, "margin")[0])
    check("margin is capped by elicited peakedness, not by disjointness",
          0.70 < qm < 0.76, "q=%.4f (expected ~0.731 = softmax([5,4])[0])" % qm)
    # identical contexts -> no disagreement
    t2, c2 = _side_masks(tl, tl, 2)
    for m in ("elicited_outside", "tv", "margin"):
        q = _q_of(tl, tl, t2, c2, m)
        check("q = 0 when the contexts agree (%s)" % m, float(q[0]) < 1e-6,
              "q=%.4f" % float(q[0]))


def test_decode_loop_with_a_stub_model():
    """Drive _driven_hf_partial with a fake HF model: fixed logits, no weights.

    Catches the control-flow bugs the pure-math tests cannot -- an empty support crashing the
    sampler, stats never being recorded, EOS not terminating, plausibility read off the tilted
    distribution instead of the target's.
    """
    from bloom.bloom import hftilt

    VOC, EOS = 12, 11

    class _Out:
        def __init__(self, logits):
            self.logits = logits
            self.past_key_values = None

    class _Stub:
        """Fixed distribution: its own favourite, plus token 5 which BOTH sides rank second.

        The shared second choice is deliberate -- it makes the top-2 intersection exactly {5},
        so the truncated decode has somewhere to go and the intersection path is what gets
        exercised. With no shared token the engine would emit EOS immediately and the test
        would pass vacuously on an empty transcript.
        """
        SHARED = 5

        def __init__(self, fav):
            self.fav = fav

        def __call__(self, input_ids=None, attention_mask=None, use_cache=True,
                     past_key_values=None, logits_to_keep=None):
            b = input_ids.shape[0]
            lg = torch.full((b, 1, VOC), -8.0)
            lg[:, 0, self.fav] = 6.0
            lg[:, 0, self.SHARED] = 2.0
            lg[:, 0, EOS] = -4.0         # reachable, but must not win the tilted argmax
            return _Out(lg)

    class _Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "X"

        def encode(self, s, add_special_tokens=False):
            return [1, 2]

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids)

    hf = {"mt": _Stub(3), "mc": _Stub(7), "tok": _Tok(), "tok_c": _Tok(),
          "device": torch.device("cpu"), "pad_id": 0, "eos_id": EOS,
          "target_no_think": "", "corrupt_no_think": ""}
    cfg = {"system_prompt": "s", "prefill": "", "b1": 1.0, "b2": 1.0,
           "api_rule": "poe", "api_top_k": 2, "api_floor": 0.0,
           "hf_empty_action": "target_argmax", "hf_measure_oracle": True, "hf_greedy": True}

    os.environ.pop("BLOOM_FOLDER", None)      # keep the stats sink from writing anywhere
    out = hftilt._driven_hf_partial(hf, cfg, [[{"role": "user", "content": "hi"}]] * 2,
                                    6, 1.0, False)
    check("stub decode returns one result per scenario", len(out) == 2)
    check("stub decode returns the expected keys",
          all({"best_text", "best_ids", "best_token_probs"} <= set(o) for o in out))
    # target top-2 is {3, 5}, elicited top-2 is {7, 5}: the intersection is {5}, so a truncated
    # decode must emit token 5 -- NEITHER side's own favourite. That is the whole point of the
    # rule, and it is the one assertion here that would catch a union/intersection mix-up.
    check("truncated poe emits the shared candidate, not either favourite",
          all(o["best_ids"] and set(o["best_ids"]) == {5} for o in out),
          "got %r" % [o["best_ids"] for o in out])
    check("token probs are percentages in (0, 100]",
          all(0.0 < p <= 100.0 for o in out for p in o["best_token_probs"]))
    # Built from the stub's own construction rather than retyped, so the expectation cannot
    # drift from the model it is checking. The emitted token is 5; under the TARGET alone that
    # is a low-probability choice, whereas under the tilted distribution it is the argmax --
    # so this distinguishes the two and would catch reporting the steered probability.
    _t = torch.full((VOC,), -8.0)
    _t[3] = 6.0          # target's favourite
    _t[_Stub.SHARED] = 2.0
    _t[EOS] = -4.0
    want = 100.0 * torch.softmax(_t, -1)[_Stub.SHARED].item()
    check("plausibility is the TARGET probability, not the tilted one",
          all(abs(p - want) < 1e-3 for o in out for p in o["best_token_probs"]),
          "want %.4f got %r" % (want, [o["best_token_probs"][:1] for o in out]))

    # top_k=0 on the same stub must instead emit the tilted argmax and keep going.
    cfg0 = dict(cfg, api_top_k=0)
    out0 = hftilt._driven_hf_partial(hf, cfg0, [[{"role": "user", "content": "hi"}]] * 2,
                                     6, 1.0, False)
    check("k=0 decode produces tokens", all(len(o["best_ids"]) >= 1 for o in out0))

    # The q metric is validated in TWO places -- _q_of, and a whitelist at the top of
    # _driven_hf_partial -- and adding topM_disjoint to only the first let four racial jobs
    # launch and die 7.6s in. A metric the maths accepts must also survive startup.
    for name in ("top1_disjoint", "top5_disjoint"):
        cfgq = dict(cfg, api_top_k=0, api_adaptive=True, api_alpha0=0.222, api_alpha_k=2.0,
                    api_q_metric=name)
        try:
            outq = hftilt._driven_hf_partial(hf, cfgq, [[{"role": "user", "content": "hi"}]] * 2,
                                             6, 1.0, False)
            check("%s survives startup validation" % name, all("best_ids" in o for o in outq))
        except RuntimeError as e:
            check("%s survives startup validation" % name, False, str(e))

    cfgbad = dict(cfg, api_top_k=0, api_adaptive=True, api_q_metric="topfive_disjoint")
    try:
        hftilt._driven_hf_partial(hf, cfgbad, [[{"role": "user", "content": "hi"}]] * 2,
                                  6, 1.0, False)
        check("malformed topM name is rejected at startup", False, "no exception")
    except RuntimeError:
        check("malformed topM name is rejected at startup", True)




def test_top1_disjoint_reproduces_top1_mismatch():
    """m=1 is the SAME metric under a new name -- the regression anchor for the family."""
    keep = torch.ones_like(TL, dtype=torch.bool)
    a = _q_of(TL, CL, keep, keep, "top1_mismatch")
    c = _q_of(TL, CL, keep, keep, "top1_disjoint")
    check("top1_disjoint == top1_mismatch", torch.equal(a, c), "%r vs %r" % (a, c))


def test_disjointness_is_rarer_as_m_grows():
    """q must be NON-INCREASING in m: disjoint top-m implies disjoint top-(m-1).

    This is the whole point of the family -- m is a dial on how often the switch fires. If
    it were not monotone, sweeping m would not be sweeping anything interpretable.
    """
    tl = torch.randn(64, V)
    cl = torch.randn(64, V)
    keep = torch.ones_like(tl, dtype=torch.bool)
    qs = [_q_of(tl, cl, keep, keep, "top%d_disjoint" % m) for m in (1, 2, 3, 4, 5)]
    for m in range(1, len(qs)):
        check("q(m=%d) <= q(m=%d) rowwise" % (m + 1, m), bool((qs[m] <= qs[m - 1]).all()))
    rates = [float(q.mean()) for q in qs]
    check("firing rate strictly falls somewhere in 1..5", rates[0] > rates[-1],
          "rates %r" % (rates,))


def test_top5_disjoint_differs_from_top1_mismatch():
    """The discriminating case, and the only test here that proves the two are not aliases.

    A previous round of "sanity checks" compared new metrics only on inputs that were either
    identical or fully disjoint -- where EVERY metric returns 0.0 or 1.0 -- and so passed
    while the code under test was wrong. Row 0 below is built specifically so the two metrics
    must DISAGREE: the argmaxes differ, but the top-5 sets are the same five tokens.
    """
    tl = torch.full((2, V), -10.0)
    cl = torch.full((2, V), -10.0)
    # row 0: same five candidates, different order -> top1 differs, top5 overlaps
    tl[0, [0, 1, 2, 3, 4]] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    cl[0, [0, 1, 2, 3, 4]] = torch.tensor([4.0, 5.0, 3.0, 2.0, 1.0])
    # row 1: genuinely unrelated candidates -> both metrics fire
    tl[1, [0, 1, 2, 3, 4]] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    cl[1, [10, 11, 12, 13, 14]] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    keep = torch.ones_like(tl, dtype=torch.bool)
    q1 = _q_of(tl, cl, keep, keep, "top1_mismatch")
    q5 = _q_of(tl, cl, keep, keep, "top5_disjoint")
    check("row0: top1 fires", q1[0] == 1.0, "got %r" % (q1[0],))
    check("row0: top5 does NOT fire", q5[0] == 0.0, "got %r" % (q5[0],))
    check("row1: both fire", q1[1] == 1.0 and q5[1] == 1.0, "got %r %r" % (q1[1], q5[1]))


def test_topm_tv_is_the_smooth_form_of_topm_disjoint():
    """TV over the two top-m windows is EXACTLY 1 iff the windows are disjoint.

    That identity is the reason the smooth metric belongs to the same family rather than
    being a second idea: top5_disjoint is top5_tv thresholded at 1.
    """
    tl = torch.randn(256, V)
    cl = torch.randn(256, V)
    keep = torch.ones_like(tl, dtype=torch.bool)
    d = _q_of(tl, cl, keep, keep, "top5_disjoint")
    t = _q_of(tl, cl, keep, keep, "top5_tv")
    check("tv == 1 exactly where disjoint fires",
          bool(((t > 1 - 1e-5) == (d > 0.5)).all()),
          "disjoint %d rows, tv==1 %d rows" % (int((d > 0.5).sum()), int((t > 1 - 1e-5).sum())))


def test_topm_tv_is_actually_graded():
    """The discriminating test: a smooth metric must take values strictly inside (0, 1).

    If it only ever returned 0 or 1 it would be the binary metric wearing a different name,
    and the whole point of the smooth family would be lost -- which is precisely the kind of
    thing an earlier round of checks failed to notice.
    """
    tl = torch.randn(256, V)
    cl = torch.randn(256, V)
    keep = torch.ones_like(tl, dtype=torch.bool)
    for name in ("top5_tv", "top5_outside"):
        q = _q_of(tl, cl, keep, keep, name)
        interior = ((q > 1e-4) & (q < 1 - 1e-4)).float().mean()
        check("%s is graded, not a switch" % name, float(interior) > 0.25,
              "only %.1f%% of rows strictly inside (0,1)" % (100 * float(interior)))
        check("%s stays in [0,1]" % name, bool((q >= 0).all() and (q <= 1).all()))


def test_topm_tv_is_not_the_full_vocab_tv():
    """The window is the point: pinning it at m must differ from measuring over everything."""
    tl = torch.randn(128, V)
    cl = torch.randn(128, V)
    keep = torch.ones_like(tl, dtype=torch.bool)
    windowed = _q_of(tl, cl, keep, keep, "top5_tv")
    full = _q_of(tl, cl, keep, keep, "tv")
    check("top5_tv differs from full-vocab tv",
          float((windowed - full).abs().mean()) > 0.05,
          "mean |diff| = %.4f" % float((windowed - full).abs().mean()))


def test_topm_metrics_agree_on_the_two_extremes():
    """Identical windows -> 0; disjoint windows -> 1. Both smooth metrics, both ends."""
    tl = torch.full((2, V), -10.0)
    cl = torch.full((2, V), -10.0)
    vals = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    tl[0, [0, 1, 2, 3, 4]] = vals
    cl[0, [0, 1, 2, 3, 4]] = vals            # identical
    tl[1, [0, 1, 2, 3, 4]] = vals
    cl[1, [10, 11, 12, 13, 14]] = vals       # disjoint
    keep = torch.ones_like(tl, dtype=torch.bool)
    for name in ("top5_tv", "top5_outside"):
        q = _q_of(tl, cl, keep, keep, name)
        check("%s: identical windows -> 0" % name, abs(float(q[0])) < 1e-5, "got %r" % float(q[0]))
        check("%s: disjoint windows -> 1" % name, abs(float(q[1]) - 1.0) < 1e-5,
              "got %r" % float(q[1]))


def test_padding_cannot_manufacture_an_overlap():
    """A row truncated below m must not count topk's zero-probability padding as agreement."""
    tl = torch.full((1, V), -10.0)
    cl = torch.full((1, V), -10.0)
    tl[0, 0] = 5.0
    cl[0, 7] = 5.0
    t_keep = torch.zeros_like(tl, dtype=torch.bool); t_keep[0, 0] = True
    c_keep = torch.zeros_like(cl, dtype=torch.bool); c_keep[0, 7] = True
    q = _q_of(tl, cl, t_keep, c_keep, "top5_disjoint")
    check("one survivor each, different token -> disjoint", q[0] == 1.0, "got %r" % (q[0],))
    c_keep2 = torch.zeros_like(cl, dtype=torch.bool); c_keep2[0, 0] = True
    q2 = _q_of(tl, cl, t_keep, c_keep2, "top5_disjoint")
    check("one survivor each, same token -> not disjoint", q2[0] == 0.0, "got %r" % (q2[0],))


def test_q_ref_identity_leaves_alpha_untouched():
    """q_ref=1.0 must be EXACTLY the old formula -- every prior run depends on it."""
    q = torch.rand(64)
    for a0, kk in ((0.222, 2.0), (0.6, 10.0), (0.4, 1.0)):
        want = (a0 * (1.0 - q ** kk)).clamp_min(1e-9)
        got = _alpha_of(q, a0, kk, 1.0)
        check("q_ref=1 identity at a0=%s k=%s" % (a0, kk), torch.allclose(want, got, atol=0))


def test_q_ref_restores_full_handover():
    """The whole point: a graded q that never nears 1 must still be able to reach alpha ~ 0.

    Without q_ref, q=0.2 under alpha0=0.222 leaves alpha at 0.178 -- a 20% nudge. With
    q_ref=0.2 the same position hands the token over outright.
    """
    q = torch.tensor([0.2])
    unscaled = float(_alpha_of(q, 0.222, 1.0, 1.0))
    scaled = float(_alpha_of(q, 0.222, 1.0, 0.2))
    check("unscaled q=0.2 barely moves alpha", unscaled > 0.15, "got %.4f" % unscaled)
    check("q_ref=0.2 sends the same q to full handover", scaled < 1e-6, "got %.6f" % scaled)


def test_q_ref_stays_graded_below_the_reference():
    """Above q_ref: saturated. Below: proportional, and still monotone decreasing in q."""
    qs = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8])
    a = _alpha_of(qs, 0.222, 1.0, 0.4)
    check("alpha is non-increasing in q", bool((a[1:] <= a[:-1] + 1e-9).all()), "%r" % a)
    check("alpha=alpha0 at q=0", abs(float(a[0]) - 0.222) < 1e-6, "got %r" % float(a[0]))
    check("alpha saturates at q=q_ref", float(a[4]) < 1e-6, "got %r" % float(a[4]))
    check("alpha stays saturated above q_ref", float(a[6]) < 1e-6, "got %r" % float(a[6]))
    interior = [float(x) for x in a[1:4]]
    check("alpha is graded strictly between", all(1e-6 < v < 0.222 for v in interior),
          "%r" % interior)


def test_q_ref_is_validated():
    """0 would divide by zero; >1 would weaken a graded metric instead of strengthening it."""
    from bloom.bloom import hftilt
    VOC, EOS = 12, 11

    class _O:
        def __init__(s_, l):
            s_.logits, s_.past_key_values = l, None

    class _S:
        def __call__(s_, input_ids=None, attention_mask=None, use_cache=True,
                     past_key_values=None, logits_to_keep=None):
            lg = torch.full((input_ids.shape[0], 1, VOC), -8.0)
            lg[:, 0, 3] = 6.0
            lg[:, 0, 5] = 2.0
            return _O(lg)

    class _T:
        def apply_chat_template(s_, m, tokenize=False, add_generation_prompt=True):
            return "X"

        def encode(s_, x, add_special_tokens=False):
            return [1, 2]

        def decode(s_, x, skip_special_tokens=True):
            return "y"

    hf = {"mt": _S(), "mc": _S(), "tok": _T(), "tok_c": _T(),
          "device": torch.device("cpu"), "pad_id": 0, "eos_id": EOS,
          "target_no_think": "", "corrupt_no_think": ""}
    base = {"system_prompt": "s", "prefill": "", "b1": 1.0, "b2": 1.0,
            "api_rule": "poe", "api_top_k": 0, "api_floor": 0.0,
            "hf_empty_action": "target_argmax", "hf_measure_oracle": True, "hf_greedy": True,
            "api_adaptive": True, "api_alpha0": 0.222, "api_alpha_k": 1.0,
            "api_q_metric": "top5_tv"}
    os.environ.pop("BLOOM_FOLDER", None)
    for bad in (0.0, 1.5):
        try:
            hftilt._driven_hf_partial(hf, dict(base, api_q_ref=bad),
                                      [[{"role": "user", "content": "hi"}]], 3, 1.0, False)
            check("q_ref=%r rejected at startup" % bad, False, "no exception")
        except RuntimeError:
            check("q_ref=%r rejected at startup" % bad, True)
        except Exception as e:
            check("q_ref=%r rejected at startup" % bad, False,
                  "wrong error: %s: %s" % (type(e).__name__, e))
    # and a legal one must still run end to end
    out = hftilt._driven_hf_partial(hf, dict(base, api_q_ref=0.25),
                                    [[{"role": "user", "content": "hi"}]], 3, 1.0, False)
    check("q_ref=0.25 decodes", len(out) == 1 and "best_ids" in out[0])


def test_unknown_metric_raises():
    """A typo must not quietly become elicited_outside, which is identically 0 at k=0."""
    keep = torch.ones_like(TL, dtype=torch.bool)
    try:
        _q_of(TL, CL, keep, keep, "top1_mismtach")
        check("unknown metric raises", False, "no exception")
    except ValueError:
        check("unknown metric raises", True)

for fn in (test_k0_is_logittilt, test_poe_support_is_intersection, test_mix_support_is_union,
           test_poe_union_zero_fills_the_missing_side,
           test_alpha0_matches_the_fixed_weights_at_q0, test_q_is_bounded_and_responds,
           test_renormalisation_is_a_noop_for_poe, test_probabilities_live_only_on_the_support,
           test_empty_intersection_is_reachable,
           test_top1_disjoint_reproduces_top1_mismatch,
           test_disjointness_is_rarer_as_m_grows,
           test_top5_disjoint_differs_from_top1_mismatch,
           test_topm_tv_is_the_smooth_form_of_topm_disjoint,
           test_topm_tv_is_actually_graded,
           test_topm_tv_is_not_the_full_vocab_tv,
           test_topm_metrics_agree_on_the_two_extremes,
           test_padding_cannot_manufacture_an_overlap,
           test_q_ref_identity_leaves_alpha_untouched,
           test_q_ref_restores_full_handover,
           test_q_ref_stays_graded_below_the_reference,
           test_q_ref_is_validated,
           test_unknown_metric_raises,
           test_decode_loop_with_a_stub_model):
    print(fn.__name__)
    fn()

print("\n%d failed" % len(FAILS) if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
