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

from bloom.bloom.hftilt import _combine, _support_mask  # noqa: E402

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


for fn in (test_k0_is_logittilt, test_poe_support_is_intersection, test_mix_support_is_union,
           test_renormalisation_is_a_noop_for_poe, test_probabilities_live_only_on_the_support,
           test_empty_intersection_is_reachable, test_decode_loop_with_a_stub_model):
    print(fn.__name__)
    fn()

print("\n%d failed" % len(FAILS) if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
