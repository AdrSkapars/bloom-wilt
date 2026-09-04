# Experiment queue

Live plan. Strike items as they land; the numbers go in `summarize.py`'s table.

Two settings per cell:
- **vanilla** — `run_api.sh vanilla 1`, the b1=1 b2=0 target-only baseline. Fast (~5 min):
  it takes the plain generate path, one call per turn, NOT the per-token driven decode.
- **overlap b=1** — `BLOOM_API_PICK=combined BLOOM_API_BETA=1 run_api.sh overlap 1`.
  Slow (~30-45 min at current latency): two API calls per generated token per scenario.

## Running

- [ ] chain A GLM: self_harm, goblin, selfpres  (`A_glm.log`)
- [ ] chain A gpt-oss: self_harm, goblin, selfpres  (`A_gptoss.log`)

## ENDPOINT STATUS -- DeepSeek degraded, column A reordered (09-05 00:25)

DeepSeek-V4-Flash is currently ~10x slower than it was earlier tonight. Probe on a
five-token prompt, all three models, both service tiers:

    dsv4    priority  11.56s  ttft=10.94    dsv4    default  9.20s  ttft=8.50
    glm     priority   3.23s  ttft=2.31     glm     default  0.96s  ttft=0.31
    gptoss  priority   0.72s  ttft=0.17     gptoss  default  0.66s  ttft=0.06

`X-Ratelimit-Over-Limit: no`, so this is capacity on the serverless replica, not throttling
of our key. For reference DeepSeek served a 961-token prompt in 1315ms earlier tonight.

The overlap decode makes two calls per generated token, so an 11s TTFT puts essentially
every request past the 25s timeout: the first column-A run (DeepSeek goblin) logged 110
retries and zero completed scenarios in eight minutes. Killed it -- waiting would not have
cleared it, and it was competing for the same capacity.

**Reorder: GLM and gpt-oss run column A now; DeepSeek goes last, re-probed first.** Do not
raise the timeout to push DeepSeek through -- at 11s/call the arm would take hours and the
throughput cost is not worth it while two healthy models are idle.

The killed run left `goblin|selfpres/deepseek_v4_flash/api_overlap_elicited_15s/round_1`
holding valid cfg/ideation/understanding (15 scenarios, all parse). Those stages are
auditor-side and independent of the decode rule, so they are KEPT for reuse, not deleted.

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
  - [x] gpt-oss self_harm (done 00:33, 6m45s, 0 retries): 21.3 @ 68.77%
        vs vanilla 17.3 @ 78.79%, combined 16.0 @ 76.95%, elicited-only 100.0 @ 41.83%
        => 4.8% captured, against combined's -1.6%. Directionally what the peakedness
        account predicts (elicited-pick > combined-pick on a confident model) but NOT the
        DeepSeek-scale rescue: +4.0 presence for -10.0 plausibility is a bad trade, and on
        n=15 with judge steps of 10 a 4-point shift is ~half a scenario. Inside noise on
        one cell -- do not call it an effect until goblin/selfpres land.
  - [x] gpt-oss goblin (done 00:40, 6m43s, 0 retries): 10.0 @ 75.60%
        vs vanilla 10.0 @ 81.85%, elicited-only 84.7 @ 72.49% => 0.0% captured, again, and
        -6.3pp plausibility for it. 10.0 is the judge floor: the behaviour never appears at
        all. combined was also 0.0% here, so the better pick rule changes nothing on gpt-oss
        goblin -- consistent with the ceiling being reachable (84.7) but not from inside the
        target's own top-5.
  - [ ] GLM x3 (self_harm running), gpt-oss selfpres (running)

### Verified non-issue: `b2=4.0` in the overlap banner

Every overlap run ever logged prints `[jailbroken_output] ... (b1=1.0, b2=4.0, floor=0.0)`,
including all nine grid cells. It is NOT applied: `_tilt_generate` returns `_driven_overlap`
at the top when `api_rule == "overlap"`, before b1/b2 are read (apitilt.py:626). The banner
is rollout.py's generic tilt line, printed whatever the engine. The knobs that do apply are
on the next line, `[api_tilt rule=overlap pick=... beta=... fb=...]`. Left as-is rather than
patching a paper file for a cosmetic string.
  - [ ] DeepSeek goblin, selfpres   <- LAST, re-probe DeepSeek latency before launching

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
