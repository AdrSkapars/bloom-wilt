# LogitTilt over a hosted API

Running the WILT target through a hosted `/completions` endpoint instead of local weights,
so experiments no longer need a rented GPU box.

```bash
bash experiments/postpaper/api_tilt/run_api.sh elicited 1   # b1=0, b2=1
bash experiments/postpaper/api_tilt/run_api.sh vanilla  1   # b1=1, b2=0
```

Code: `src/bloom/bloom/apitilt.py`, selected as `jailbroken_output.engine=api_tilt`, which
a `BLOOM_TARGET_MODEL=api/<model>` target forces automatically.

## What an API can and cannot do

LogitTilt samples from `z = b1·ℓ_target + b2·ℓ_jail` — two contexts of the same weights,
stepped in lockstep, mixed in **full-vocab logit space before sampling**. No hosted API
exposes that. But two corners of the (b1, b2) plane need no mixing at all, because in each
the sampling distribution is a single context's own softmax:

| | b1 | b2 | z | status |
|---|---|---|---|---|
| target only   | 1 | 0 | `ℓ_target` | had it (`vanilla_15s`, local) |
| tilted        | 1 | β | `ℓ_target + β·ℓ_jail` | had it (`jail_b0.5`, `jail_b1`, local) — **needs logits** |
| elicited only | 0 | 1 | `ℓ_jail` | **the missing corner — this is what `api_tilt` adds** |

Everything strictly between the corners is refused by `apitilt.py` rather than silently
approximated. Approximating the true tilt from the top-k alternatives an API *will* return
is a separate, explicit experiment (see "next" below).

## Mechanics

The two corners differ in how the reported plausibility is obtained. The metric, matching
the local engine's `best_token_probs`, is always the **unmodified-target probability of the
tokens that were actually sampled**.

* **target only** — the tokens came from the target itself, so the generation call's own
  logprobs *are* the on-policy target probs. One call.
* **elicited only** — the tokens came from the **jail** context (jail system prompt, target
  system prompt dropped, behaviour-file prefill appended and never sampled). Their target
  probability needs a separate teacher-forced pass over the target context:
  `echo=true` + `max_tokens=0` scores a supplied sequence.

Prompts are rendered here by Jinja (`BLOOM_TARGET_CHAT_TEMPLATE`) rather than by a
tokenizer, since there is no tokenizer in the process. That template is therefore the only
thing pinning the API's tokenization to the local runs'.

### Two gotchas worth keeping

**Score token ids, not text.** Sampling can produce a *non-canonical* tokenization, so
re-tokenizing the decoded string gives a different token sequence — observed live: 73
sampled tokens re-tokenizing to 72. Scoring the string would then report per-token
probabilities for a sequence the model never emitted, and (because the split is taken from
the end) misalign every token. Fireworks accepts an **integer-array prompt**, so
`score_ids()` sends `prefix_ids + sampled_ids` and gets position-for-position logprobs.
Costs one extra `max_tokens=0` call to tokenize the prefix. The engine verifies the ids
come back unchanged and raises if they do not.

**`top_k` must be pinned.** A provider-side default `top_k` would truncate the tail and
quietly stop matching the local `multinomial(softmax(z/T))` step. The client sends
`top_p=1.0, top_k=0` explicitly.

## Validation

Re-scoring the **local** vanilla run's stored `gen_token_ids` through this template and the
Fireworks echo endpoint:

* every token id reproduced **exactly**, on all 9 turns tested (3 transcripts × 3 turns) —
  the Jinja render matches what the local tokenizer produced;
* mean token probability within **1.31 pp** on average, API running consistently a little
  *lower* (7 of 9 turns).

## Caveats when comparing to the local arms

* **No naturalness floor.** The local `jail_b0.5` / `jail_b1` arms ran `target_floor=1e-4`
  (mask tokens the *target* gives < 1e-4, then sample the tilt among survivors). The floor
  thresholds on the target distribution at *every* step, which needs target logits while
  decoding — impossible here. The elicited-only corner is defined without a target term
  anyway, so it runs floor-off, and unfloored pure-jail sampling does reach tokens at
  ~0% target probability.
* **Serving numerics differ** from the local FP4 load: expect API plausibility ~1 pp low.
  Low-probability tokens diverge most in relative terms, so **min**-token statistics are
  less comparable across engines than means. Part of that gap is the decode-vs-prefill
  difference below rather than quantisation as such.
* Everything else is matched deliberately: same kickoff bank (so identical scenarios *and*
  identical turn-1 kickoffs), 15 scenarios, seed 1, 3 turns, temperature 1.0,
  `target_max_tokens=250`.

## Providers

`_PROVIDERS` in `apitilt.py`. Fireworks is verified end-to-end. Any provider needs:
`/completions` serving the model, `echo=true` + `max_tokens=0` scoring, per-token logprobs,
and an integer-array prompt. Together is registered but unverified on the last point.

## Generation and echo are not the same forward pass

Measured directly (same prompt, same model, 60 tokens):

| | mean \|diff\| | max |
|---|---|---|
| echo vs echo | **0.0000 pp** (bit-identical, deterministic) | 0.000 pp |
| generation vs echo | **2.10 pp** | 20.8 pp |

The incremental-decode path and the single-pass prefill/echo path produce numerically
different logits, in both directions, uniformly across positions. Consequences:

* Everything in the top-5 datasets comes from the **echo** path, so it is internally
  consistent, deterministic and re-runnable.
* `best_token_probs` (the reported plausibility) is echo-derived — `score_ids` — on the
  elicited path, and so is comparable to the top-5 data.
* `gen_token_probs_jail` stored in the elicited transcripts is **generation**-derived, so
  it is *not* directly comparable to echo numbers (mean 2.3 pp apart, max 26 pp). Re-score
  it through `score_ids` if you need it on the same footing.
* Target-only plausibility comes from the generation call by construction (the tokens were
  drawn from that very distribution), so vanilla's numbers are on the generation path.

This is why the elicited-direction dataset validates so cleanly on one column and not the
other: `target.lp` reproduces the run's stored target probs to 5e-5 pp (echo vs echo),
while `elicited.lp` differs from the stored jail probs (echo vs generation).

## Paired top-5 datasets

`collect_top5.py` teacher-forces a run's generated tokens through **both** contexts and
records the top-5 alternatives at every position. `--arm` chooses whose tokens:

```bash
python -X utf8 experiments/postpaper/api_tilt/collect_top5.py --arm vanilla_15s
python -X utf8 experiments/postpaper/api_tilt/collect_top5.py --arm api_elicited_15s
python -X utf8 experiments/postpaper/api_tilt/top5_overlap.py  --arm vanilla_15s
```

Output: `<run>/round_1/top5/transcript_vNrM.top5.json` + `_index.json`. Per turn, parallel
arrays indexed by position:

```
token_ids[i], tokens[i]           the token that run emitted
target.lp[i],   target.top[i]     top[i] = [[token_string, logprob] x 5], most likely first
elicited.lp[i], elicited.top[i]
```

`top[i]` is the distribution that **predicted** token `i`, so the emitted token is often
absent from it. Alternatives carry no token ids, only strings, so set overlap is computed
on strings. Both directions verified: every one of the 45 turns per arm echoes the source
run's `gen_token_ids` back unchanged.

### Headline

| | target-only tokens | elicited-only tokens |
|---|---|---|
| positions | 5164 | 3603 |
| mean \|top5 ∩ top5\| | 3.72 / 5 | 3.61 / 5 |
| identical top-5 sets | 22.5% | 18.7% |
| disjoint top-5 sets | 1.2% | 1.0% |
| same top-1 token | 74.1% | 71.7% |
| emitted token in target's top-5 | 97.2% | 84.6% |
| emitted token in elicited's top-5 | 91.0% | 92.3% |
| p(emitted) under target | 69.58% | 51.00% |
| p(emitted) under elicited | 55.65% | 58.96% |

Each run's own context is the more generous one along its own trajectory, as it must be —
but only mildly: the contexts largely agree on the *candidate set* (3.6–3.7 of 5 shared)
and differ mainly in how they *rank* it. The ~1% of positions with disjoint top-5s are
where a top-5 reconstruction of `b1·ℓ_target + b2·ℓ_jail` has no shared support at all.

## Next

The top-5 approximation itself: reconstruct the tilt from these 5-alternative lists and
measure how far it lands from the true full-vocab tilt (`jail_b1` locally).
