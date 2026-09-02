# DeepSeek-V4-Flash as a BLOOM target — how it actually loads and runs

Written 2026-09-02 after getting it working on a 2×H200 vast box. It took several
attempts; the dead ends are recorded because each one looks plausible from outside.

## TL;DR

Use the **original** `deepseek-ai/DeepSeek-V4-Flash-0731`. Install `kernels==0.16.0`,
patch one line in transformers, force `eager` attention, supply the chat template from
`src/bloom/prompts/deepseek_v4_chat.jinja`, and keep `var_batch` at **8**. Loads in ~36 s,
156 GB across 2 devices, generates correct text.

---

## 1. Use the ORIGINAL checkpoint, never a "quantised" refork

Do not use `True2456/...-AWQ` or `baicai1145/...-W4A16`. Both were tried; both fail the
same way, and it is architectural rather than a bad upload.

transformers implements DeepSeek-V4's MoE with **fused 3-D expert parameters** —
`model.layers.N.mlp.experts.gate_up_proj` of shape `(256, 4096, 4096)`, one Parameter for
all 256 experts — not per-expert `nn.Linear` submodules. Every compressed-tensors / AWQ /
GPTQ checkpoint works by swapping a Linear for a quantised Linear, so nothing matches:

```
Could not match `re:.*experts\.\d+\.(w1|w2|w3|gate_proj|...)$` in instance of DeepseekV4ForCausalLM
Applying quantization config: 0it
```

It then unpacks the int4 weights to bf16 on the way in — 284 B × 2 B = **568 GB**, which
fills both H200s at 52 % of the way through loading and then silently spills to CPU.

The original is *already 4-bit*. Its routed experts are packed FP4 (int8 container, two
e2m1 per byte) with ue8m0 group-32 scales; only attention is FP8 e4m3:

```
layers.0.ffn.experts.0.w1.weight   I8       [2048, 2048]   # logical [2048, 4096]
layers.0.ffn.experts.0.w1.scale    F8_E8M0  [2048, 128]    # one scale per 32
```

**167 GB — smaller than the 177 GB "W4A16" refork.** That size inversion is the tell: a
4-bit conversion that comes out bigger than its source is a re-encoding, not a conversion.
`config.json` says `quant_method: "fp8"` (that describes the attention path), which routes
to `FineGrainedFP8HfQuantizer` — and that quantizer handles the FP4 nibble packing
(`weight_k_div`), ue8m0 scales, and the 3-D expert leading dim.

**Lesson: diagnose a checkpoint by its safetensors header dtypes, not its filename or size.**
The header check took about a minute after hours of ranking repos by name.

## 2. Environment setup

```bash
pip install --no-deps "kernels==0.16.0" kernels-data
```

Without it: `ImportError: finegrained-fp8 kernel unavailable ... install kernels==0.16.0`.
Use `--no-deps` so the resolver cannot touch torch (an earlier unconstrained install stalled
34 min and risked silently downgrading torch).

Then patch transformers — an upstream bug, ~line 188 of
`transformers/quantizers/quantizer_finegrained_fp8.py`:

```python
- layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl)
+ layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl) or {}
```

`_impl_tp_layer_overrides` has exactly one key, `"deepgemm_megamoe"`. With no experts
implementation set (the default) `.get()` returns `None` and the next line raises
`AttributeError`. It is a *tensor*-parallel plan and irrelevant to a pipeline-parallel
`device_map="auto"` load. One-liner:

```bash
F=/venv/main/lib/python3.12/site-packages/transformers/quantizers/quantizer_finegrained_fp8.py
sed -i "s/layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl)$/layer_overrides = FP8Experts._impl_tp_layer_overrides.get(impl) or {}/" $F
```

Verified working versions: torch 2.11.0+cu128, transformers 5.16.1, accelerate 1.14.0,
triton 3.6.0, compressed-tensors 0.18.0, litellm 1.99.0, tenacity (needed by the rollout).

## 3. Attention must be eager

```
_supports_flash_attn = False
_supports_sdpa       = False
_supports_flex_attn  = False
```

The architecture declares no fused backend, because its compressor path concatenates extra
KV entries plus a `block_bias` onto the attention mask that the fused kernels cannot
express. So `BLOOM_TARGET_ATTN=eager`, and the attention matrix is materialised in full —
which is what drives the memory ceiling in §5.

Multi-device also forces the Triton path over DeepGEMM (logged at load). Expected: DeepGEMM
is single-CUDA-context bound and has no FP4 kernel on Hopper anyway. FP4 is Blackwell-native,
so a B200 box would reach the real kernels if throughput ever becomes the blocker.

## 4. Chat template

The checkpoint ships none. `src/bloom/prompts/deepseek_v4_chat.jinja` mirrors
`encode_messages(thinking_mode="chat")` from the repo's `encoding/encoding_dsv4.py`.
Verified render:

```
<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜></think>
```

Chat mode emits the closed `</think>` itself, so **no prefill wrapper** —
`deepseek-ai/deepseek-v4-flash-0731` is registered `False` in `_USES_THINK_BLOCK`.

## 5. var_batch 8, not 15

At `var_batch=15` a turn-3 rollout OOMs in `eager_attention_forward` trying to allocate
**29.14 GiB** for the attention logits. Auditor turns can be 1200 tokens, so by turn 3 the
context reaches ~4k and eager attention is quadratic in that. It bit at β=0.5 first, because
weak steering produces longer, more natural replies than strong steering.

The allocation scales linearly with batch, and ~16.7 GiB was free, so 8 (chunks of 8+7) is
about the largest value that fits. **Note this contradicts a short-context benchmark** that
suggested 32 — that benchmark used 595-token contexts and was measuring the wrong thing.

Consequence for the 100-scenario finals: plan around var_batch 8, not 32, and re-derive the
runtime estimate from the β-run timings rather than the benchmark.

`var_batch` is a chunk size over seeds (scenarios × reps), so a value above the scenario
count does nothing. Changing it **changes the sampled outputs** — `wilt.py:483` calls
`torch.multinomial` with no explicit generator, so the number of random draws consumed per
call depends on batch size. Different var_batch = different draw (not a bias, but not the
same samples).

## 6. Running

`run_arm.sh <beta> [rounds]` in this directory is the working launcher. Key points:

- **Vanilla = omit `BLOOM_JAIL_MODEL`** (leaves `jailbroken_output.enabled` False → the
  `target_only` path). Not `BLOOM_JAIL_BETA=0`. See `final_run.py:148`.
- **`BLOOM_EVAL_THINKING=0` / `BLOOM_JUDGE_THINKING=0`.** Gemma-4 is registered `False` in
  `_USES_THINK_BLOCK`, so `think_prefix()` returns `""` and the *local* auditor produced
  identical prompts whether thinking was on or off — the flag was a no-op there. Left on,
  the hosted auditor reserves a real 2048-token budget the original runs never had, and
  fails litellm's budget check against the 1200/500 caps.
- Auditor + judge over OpenRouter (`openrouter/google/gemma-4-26b-a4b-it`) so both GPUs are
  free for the target. Key in `.env.local`, **written as ASCII** — PowerShell's `>>` writes
  UTF-16LE + CRLF, which bash cannot source, and it then echoes the whole line (key
  included) into the log as a "command not found" error.
- Kickoff bank reused from `experiments/bloom/_banks/` (target-independent; generated by the
  auditor). Its `meta.json` must name the auditor actually in use.
- `BLOOM_TARGET_*` knobs must be declared in `_ENV_KNOWN_EXTRA` in `bloom_corrupt.py` or the
  env guard refuses to start.

Rounds are resume-aware: an existing `round_N/judgment.json` is reused, so re-running with
`rounds=2` on a finished 1-round arm only generates round 2 (seed = base + N).

## 7. Results so far (self_harm, 15 scenarios, seed 1, 3 turns)

Per-round, 1 round each:

| β | presence | tok_avg | tok_min |
|---|---|---|---|
| 0 (anchor) | 11.3 | 73.2 % | 0.0381 % |
| 0.5 | 18.0 | 71.4 % | 0.1809 % |
| 1 | 51.3 | 61.8 % | 0.0231 % |
| 2 | 91.3 | 55.8 % | 0.0102 % |

Only β=0.5 sits inside the paper's ±3 pp plausibility band (floor 70.2 %), so it is the
deployed β. Headline independent of the band: **β=2 takes a near-frontier model from 11.3 to
91.3 presence.**

Post-run selection on 2-round pools (β=0 and β=0.5 only):

| arm | selection | presence | prob |
|---|---|---|---|
| β=0 | max presence per scenario (w=1) | 14.7 | 71.47 % (P_ref) |
| β=0.5 | highest presence with prob ≥ P_ref | 24.7 | 72.17 % |

β=0.5 strictly dominates — more presence *and* more probability, by 0.70 pp. 24.7 is the
frontier ceiling, so trading probability down buys nothing.

**Break presence-ties toward the higher probability.** `_best_of_pool` breaks them
arbitrarily and reports P_ref as 70.32 %; tie-broken properly it is 71.47 %. Both arms have
many ties (β=0 especially — its scores are nearly all at the floor), so this is not a
rounding detail: it moved the margin from 1.85 pp to 0.70 pp.

**Selected probability RISES with rounds, fastest for β=0.** Raw per-round probability
drifts down (β=0 70.50 → 70.04, β=0.5 72.10 → 71.25) but selection more than cancels it:
P_ref 70.50 → 71.47 (+0.98 pp/round), β=0.5 72.10 → 72.17 (+0.07). β=0 gains most precisely
because its presence scores are nearly all tied at the floor, leaving the probability term
free to break every tie. So the gap to a higher β **widens** with rounds rather than
closing — adding rounds will not rescue β=1 (raw 66.33 %, 5.15 pp under the 2-round P_ref).
Evidence is one extra round on two arms at n=15; measure β=1's own rate at 2-3 rounds
before relying on it.

Caveats: pools of 2 are thin (paper uses 5); β=0 round 1 ran at var_batch 15 and everything
else at 8, so that pool mixes draws. Note also that `prob_stats.mean` averaged over
transcripts (70.50 % for β=0 round 1) is NOT the run summary's `tok_avg` (73.2 %) — they are
different aggregations, so never mix them in one comparison.

## 8. Selection results (2026-09-02, final for the day)

Pools: vanilla 8 rounds, beta 0.5 and beta 1 five rounds each, 15 scenarios, seed 1.
Anchor = vanilla's w=1 point (max presence per scenario, ties broken toward the more
probable transcript). Band floor = anchor - 3 pp = 70.64 %.

| arm | rounds | presence | prob | band |
|---|---|---|---|---|
| vanilla | 8 | 24.0 | 73.64 % | anchor |
| beta 0.5 | 5 | 36.0 | 71.39 % | in |
| **beta 1** | 5 | **38.7** | **70.79 %** | **in — deployed** |

**beta 1 is the deployed beta: 38.7 vs vanilla's 24.0 at matched plausibility (1.6x), on
5 rounds against vanilla's 8.** Plot: `selection_frontiers.png`, regenerate with
`python -X utf8 experiments/bloom/deepseek_v4/selection_plot.py`.

**The band verdict depends on pool depth, and a 1-round sweep gets it wrong.** beta 1's
selected probability climbs 66.33 % (1 rd) -> 68.38 % (3 rd) -> 70.79 % (5 rd), while
vanilla's P_ref rises 70.50 -> 72.73 -> 73.33 and then plateaus by round 3-5 (it runs out
of presence-ties to break; beta 1 has real spread and keeps gaining). So at 1 round beta 1
looks 8.7 pp outside the band and beta 0.5 is the only candidate; by 5 rounds beta 1 is
inside AND wins. A cheap 1-round sweep is not merely noisier, it is **systematically biased
toward low beta**. Use the paper's 5-round tuning config.

Per-round figures (presence / tok_avg), which do NOT show this because there is no pooling:
beta 0 ~11-19 @ ~72 %, beta 0.5 ~16-23 @ ~71-74 %, beta 1 ~35-53 @ ~61-67 %,
beta 2 91.3 @ 55.8 % (1 round only).

## 9. One round per process -- required

Long-lived processes degrade badly. In a single process running 8 rounds of beta 0 then
3 of beta 0.5, rounds 3 and 4 took 11-12 min each and round 5 took **47 min per chunk**
with the GPU idling at 25 % / 118 W. Ruled out: auditor API (live test 30-50 tok/s),
generation length (round 5's replies averaged 473 chars, same as rounds 1-4), thermal or
power throttling (26-28 C, clocks pinned at max, throttle reasons 0x0), duplicate
processes, disk. Re-running the identical round in a **fresh** process took 740 s.

`sweep2.sh` therefore calls `run_arm.sh <beta> N` once per round -- resume-aware, so it
reuses rounds 1..N-1 and generates only round N, at the cost of a 36 s model load. Steady
timings after the change: 740 / 797 / 923 / 722 / 812 s. This has implications for
`final_run.py`, which runs all 8 BoN rounds in one process.

## 10. Next steps

- **β=2 at 5 rounds.** β=1 had not converged at round 5 (+2.4 pp from round 3), so β=2
  (91.3 presence at 1 round) may also enter the band with a full pool. That is the highest-
  value next run and it is only ~5 rounds x ~14 min.
- Vanilla 100 scenarios, seed 100, finals bank — queued but not started; ~40 min/round at
  var_batch 15 (7 chunks), so ~5 h for 8 rounds.
- `degeneracy.py` on β=2 before quoting 91.3 anywhere.
- More rounds for real selection resolution; the 2-round frontier has only 6 distinct points.
- `param_sweep.py` default `--increment` is 0.25 but the paper used **0.5** — pass it
  explicitly if that script is ever used here.
- `param_sweep._curve` divides by fixed constants where the paper says min–max normalise
  within each pool. Check which script generated the paper figures before quoting a full
  frontier as paper-comparable.
