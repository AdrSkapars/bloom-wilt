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

## Queued — two more grid columns (16 runs)

Same 3x3 (self_harm/goblin/selfpres x DeepSeek/GLM/gpt-oss), two more pick rules. Each
becomes another column of the cells.py table.

**A. `BLOOM_API_PICK=elicited`** -- argmax by the ELICITED context alone, still restricted
to the overlap. Expected to land between `combined` and elicited-only, because it keeps the
target's candidate set but chooses by the jail ranking instead of the product. The one cell
already measured supports that: DeepSeek self_harm scored 81.3 @ 56.46%, against combined's
26.0 and elicited-only's 100.0 -- ~77% captured against combined's 10.5%. If that holds
across the grid it is a far better rule than combined and the beta sweep was run on the
weaker one.

  - [x] DeepSeek self_harm (done: 81.3 @ 56.46%)
  - [ ] DeepSeek goblin, selfpres
  - [ ] GLM x3, gpt-oss x3

**B. `BLOOM_API_PICK=combined BLOOM_API_BETA=2`** -- the existing rule at the beta that
worked best on DeepSeek self_harm (48.0 @ 66.96%, against 26.0 at beta=1).

  - [x] DeepSeek self_harm (done: 48.0 @ 66.96%)
  - [ ] DeepSeek goblin, selfpres
  - [ ] GLM x3, gpt-oss x3

16 new runs. Overlap arms have run 9-46 min depending on model (gpt-oss fastest, GLM
slowest), so ~3.5 h at two at a time. Folder names do not collide:
api_overlap_elicited_15s and api_overlap_combined_b2_15s.

## Queued — 5 rounds on two hard cells (LAST, likely tomorrow)

Precedent: DeepSeek self_harm `combined` went 26.0 single-round -> 62.0 after 5-round
selection, enough to beat the real full-vocab tilt. So selection can move these a long way,
and the question is whether it rescues the cells where the single-round method fails.

Two cells, chosen so they disagree:

  - [ ] gpt-oss self_harm  (-1.6% captured, ceiling 100.0). The clearest failure with
        everything still to gain. If selection rescues THIS, selection is doing the real
        work and the pick rule matters less than it appears.
  - [ ] GLM goblin  (0.0% captured, ceiling 90.7). Tests whether pool depth can beat the
        reachability constraint. PREDICTION: it cannot -- no amount of resampling puts
        goblin tokens into the target's top-5. If it moves anyway, the reachability story
        is wrong and needs revisiting.

Four extra rounds per cell (~100 min each at current speeds), so this is a night's work on
its own, not an add-on. Run AFTER the 16 column-A/B runs.

Convention, pinned before running rather than chosen after: use selection_compare.py, all
arms cut to the SAME number of rounds, anchor = vanilla's w=1 point, one-sided band at
anchor-3pp. Report presence 0-100 and mean token prob from cells.py / summarize.py, never
the run log's tok_avg (different aggregation).

Both behaviour banks hold rounds 1-5, so 5 rounds is the maximum available for goblin and
selfpres; self_harm has 1-8.

## Queued — elicited-only, every cell

`elicited` is the UPPER BOUND: the jail context sampling unconstrained, no overlap rule.
With vanilla (lower bound) and overlap b=1 (the method) it gives the full range per cell
rather than two points. Cheap -- ~5 min each, plain generate path, no driven decode.

DeepSeek already has self_harm elicited (100.0 @ 50.86%); the other two are missing, so
its rows are not comparable to the new models' until they exist.

- [ ] DeepSeek: goblin, selfpres
- [ ] GLM: self_harm, goblin, selfpres
- [ ] gpt-oss (or qwen3p7-plus): self_harm, goblin, selfpres

Eight runs, ~40 min total. Run AFTER the current overlap arms, and prefer pairing a cheap
elicited arm with a slow overlap arm rather than two slow ones.

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

## When a run fails: FIX IT, do not move on

A failed arm is a hole in the results, and the next arm will usually fail the same way.
So on failure: read the log, find the cause, fix it, RE-RUN THAT ARM. Only then advance.

  - `RuntimeError: api_tilt request failed after N attempts` -> the retry budget or the
    timeout is wrong for current conditions, or the endpoint is degraded. Check the retry
    reasons in the log before changing anything.
  - a traceback out of `_driven_overlap` -> a bug, not weather. Fix the code.
  - `Round 1 FAILED` with no api_tilt errors -> look upstream at the auditor or judge.
  - repeated identical failures on the SAME arm, external cause (endpoint down): stop the
    queue and say so. Do not burn the remaining arms against a dead endpoint -- that is
    how a whole night gets wasted producing nothing.

Never let a failure quietly become "0 of N arms produced results". Report it when it
happens, with the cause.

## Stuck or too slow

Known-good timings at normal endpoint latency:

  - vanilla / elicited-only arm: ~3-5 min total
  - overlap arm: ~20-45 min total, roughly 3 turns, one `rule=overlap` line per turn

Treat as stuck and investigate:

  - log file untouched for >15 min while a python process is alive (this exact signature
    cost an hour: a hung socket held in CLOSE_WAIT while the endpoint answered fine)
  - an overlap arm past ~75 min, or a vanilla arm past ~15 min
  - retries climbing fast with no new `rule=overlap` line appearing

Diagnose before killing: check whether the endpoint itself is healthy with a single
direct call, and read `Fireworks-Server-Time-To-First-Token` on a response -- if TTFT is
tens of seconds, it is a cold replica and ours to route around, not a bug in the code.

## Process hazards -- all three of these bit us on 2026-09-04

1. **Never edit a shell script while a run is executing it.** Bash reads scripts
   incrementally by byte offset, so an edit shifts the file underneath the running shell
   and it resumes mid-token. Cost two spurious `FAILED` markers on runs whose data was
   actually fine. Write a NEW file instead (this is why `run_cell.sh` exists alongside
   `run_api.sh`), or wait for the run to finish.

2. **TaskStop kills the bash wrapper, NOT the python child.** A killed run left an orphan
   that ran for an hour, consumed API capacity and inflated the retry counts of the runs
   that were supposed to have the endpoint to themselves. After stopping a task, always
   check `Get-Process python` and kill survivors explicitly.

3. **Never `rm -rf` a run directory that might still be written.** Deleting the aborted
   run's folder while its orphan was still alive produced a directory with transcripts but
   no `cfg.json` -- complete-looking data whose settings cannot be verified. Quarantine
   with `mv`, never delete, and only once nothing is running.

## Launcher

`run_cell.sh` is the general one: `BEH=<beh> MODEL=<model> run_cell.sh <arm> [rounds]`.
`run_api.sh` is the older self_harm + DeepSeek-only version, kept because runs referenced
it; prefer run_cell.sh for anything new.

## Watch during runs, not just after

- `[api_tilt] retry` lines — a few are normal, hundreds mean the endpoint is degraded
- `CUT SHORT` — a scenario whose reply was truncated by API failure; contaminates that
  transcript's presence and plausibility
- empty-overlap fallback rate — was 1-6% on DeepSeek; a very different rate on a new
  model suggests the two contexts are not being built as intended
- tok_avg wildly off the ~70% range, or presence pinned at 0 or 100, on the canary
