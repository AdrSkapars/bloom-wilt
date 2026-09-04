# Experiment queue

Live plan. Strike items as they land; the numbers go in `summarize.py`'s table.

Two settings per cell:
- **vanilla** — `run_api.sh vanilla 1`, the b1=1 b2=0 target-only baseline. Fast (~5 min):
  it takes the plain generate path, one call per turn, NOT the per-token driven decode.
- **overlap b=1** — `BLOOM_API_PICK=combined BLOOM_API_BETA=1 run_api.sh overlap 1`.
  Slow (~30-45 min at current latency): two API calls per generated token per scenario.

## Running

- [ ] `combined_sample b=0.5`, then `combined_sample b=2`  (self_harm, DeepSeek)
- [ ] `combined b=1.5`  (self_harm, DeepSeek)

## Queued — DeepSeek-V4-Flash

- [ ] prefill ablation: `combined b=1`, `BLOOM_JAIL_PREFILL=0`, self_harm
      (vs the existing b=1 at 26.0 @ 72.34% — does the behaviour-file prefill do the work?)
- [ ] goblin: vanilla, then `combined b=1`
- [ ] selfpres: vanilla, then `combined b=1`

## Queued — GLM-5.3-flash  (`accounts/fireworks/models/glm-5p3-flash`)

Setup first: chat template, BOS token, `_USES_THINK_BLOCK` entry, `tokenizer.json`.
Then, in this order, because the first run is the canary:

- [ ] self_harm vanilla  <- CANARY. Exercises template + tokenizer + BOS + think-block
      end to end in ~5 min. Do not start the rest until this looks right.
- [ ] self_harm `combined b=1`
- [ ] goblin vanilla, then `combined b=1`
- [ ] selfpres vanilla, then `combined b=1`

## Queued — gpt-oss-120b  (`accounts/fireworks/models/gpt-oss-120b`)

Same shape, same canary-first order. Higher risk: it uses OpenAI's **harmony** format
(`<|start|>assistant<|channel|>final<|message|>`) rather than a conventional chat
template, which also interacts with `parse_message`'s channel handling. If it misbehaves,
substitute **`qwen3p7-plus`** — second-fastest of the six measured (1161ms median, p90
1296ms) and a conventional template.

## Why the canary matters

Three of the four setup files fail LOUDLY (a wrong template or tokenizer breaks the id
round-trip). `_USES_THINK_BLOCK` fails SILENTLY: register a model as not auto-opening a
`<think>` block when it does, and the elicited context's next-token distribution is
dominated by `</think>`. The run completes, the numbers look plausible, and they mean
nothing. The canary is the cheapest way to catch that.

## Watch during runs, not just after

- `[api_tilt] retry` lines — a few are normal, hundreds mean the endpoint is degraded
- `CUT SHORT` — a scenario whose reply was truncated by API failure; contaminates that
  transcript's presence and plausibility
- empty-overlap fallback rate — was 1-6% on DeepSeek; a very different rate on a new
  model suggests the two contexts are not being built as intended
- tok_avg wildly off the ~70% range, or presence pinned at 0 or 100, on the canary
