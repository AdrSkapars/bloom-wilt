# Experiment queue

## FREE SELECTOR built while blocked (05:05) -- refines the convergence headline

No API needed: gen_token_probs and gen_token_probs_jail are already persisted per assistant
message, so a deployable selector can be scored offline. freeselect.py implements the
api_tilt analogue of margin_pick:

    margin = mean over generated tokens of ( log p_jail - log p_target )

Empty-overlap fallback tokens carry no jail probability (stored as None) and are skipped
rather than imputed -- they are 0.4-2.4% of tokens.

    cell                     round 1          ORACLE            MARGIN (free)     recovers
    DeepSeek elic-pick    81.3 @ 56.46%   100.0 @ 56.29%    99.3 @ 53.03%          96.4%
    DeepSeek comb b=1     26.0 @ 72.34%    67.3 @ 69.62%    54.0 @ 68.53%          67.7%
    DeepSeek elic-only   100.0 @ 50.86%   100.0 @ 56.14%   100.0 @ 50.62%            n/a
    gpt-oss  elic-pick    21.3 @ 68.77%    42.7 @ 69.92%    30.7 @ 67.31%          43.8%

96.4% of oracle on the best cell, inside the 85-99% band margin_pick reached in the paper
work, so the technique transfers to the API path.

### This PARTLY REVERSES the "advantage vanishes" headline below

Under the oracle, elic-pick and elicited-only converged (56.29 vs 56.14). But the oracle
gets there by tie-breaking on PLAUSIBILITY among elicited-only's already-at-ceiling
transcripts -- something no deployable selector can do, since it has no judge. Under the
free selector elicited-only falls to 50.62% while elic-pick holds 53.03%.

So the overlap decode keeps a +2.4pp plausibility edge at equal presence. The advantage
goes +5.6pp (round 1) -> +2.4pp (free selection) -> ~0 (oracle only). Last night's
"vanishes" was itself partly an artifact of my own oracle tie-break.

Whether 2.4pp justifies 2x the API calls is a judgement call; leaning no.


## STILL BLOCKED -- the 05:53 "recovery" was a FALSE POSITIVE

A single probe returned 200 at 05:53, I relaunched both runs on it, and both hit HTTP 412
within a minute. Eight probes immediately afterwards: 412/8. The account never recovered.

MY ERROR: one 200 is not recovery. It can be a routing or cache artifact. Fixed with
`apigate.sh`, which exits 0 only when N consecutive spaced probes all return 200:

    bash experiments/postpaper/api_tilt/apigate.sh 5 && <launch the next run>

EVERY relaunch from here must be gated on it. Do not relaunch on a single probe again.

No contamination from the false start: both runs died at their first request, leaving
round_2 stubs containing only cfg.json and zero transcripts. Stubs removed; round_1 intact
in both folders.

## (historical) BLOCKED 05-09 04:55 -- FIREWORKS ACCOUNT SUSPENDED

    HTTP 412 PRECONDITION_FAILED
    "Account adr-skapars-7nj3j22k is suspended, possibly due to reaching the monthly
     spending limit or failure to pay past invoices."
    https://fireworks.ai/account/billing

Confirmed by direct probe. ALL runs stopped (both 5-round jobs killed, wrappers included --
python=0, run_cell=0 verified). Nothing further can run until billing is resolved.

### Data quarantined (2 partial rounds, no good data lost)

  * goblin/glm_5p3_flash/api_overlap_elicited_15s/_SUSPECT_round_2_api_suspended
    -- hit the suspension mid-round: one 0-byte transcript, 1 CUT SHORT.
  * self_harm/gpt_oss_120b/api_overlap_elicited_15s/_SUSPECT_round_4_partial_judgment
    -- 15 transcripts written but only 12 judged (judgment stage killed). Including a
       partial round biases best-of selection: 12 scenarios get a 4th sample, 3 do not.

round_1 in both folders is untouched grid data and remains valid.

### SURVIVING RESULT: selection rescues gpt-oss, and lifts BOTH axes

gpt-oss self_harm elic-pick, 3 complete clean rounds (0 x 412, 0 CUT SHORT in that log):

    round 1     21.3 @ 68.77%
    best-of-3   42.7 @ 69.92%     capture 4.8% -> 30.7%

Presence doubles AND plausibility rises. So "gpt-oss is immune" was ALSO a round-1 artifact,
same as the overlap-vs-elicited-only advantage. Both conclusions were about round 1, not
about the method.

### Data-quality audit done while blocked

  * 6/855 round-1 transcripts have <3 assistant turns (0.7%). They cluster 3-in-15 in GLM
    self_harm elicited-only -- the DENOMINATOR for that cell's capture fractions, and the
    lowest ceiling in the grid (78.0). Checked rather than assumed: excluding them moves the
    ceiling 78.0 -> 82.5 and capA 23.5% -> 22.1%. Real but immaterial; driven by one 1-turn
    transcript scoring 10 while the other two scored 80 and 90.

## QUEUED (blocked on billing)
  - [~] elicited-only 5 rounds, gpt-oss self_harm  RUNNING (matched control; gpt-oss
        elic-pick has 3 clean rounds so the comparison is made at R=3)
  - [~] re-run GLM goblin 5-round                  RUNNING
  - [ ] elicited-only 5 rounds, GLM goblin          (matched control, after the above)
  - [x] a real selector instead of oracle -- DONE, freeselect.py (see top of file).


## !! HEADLINE (05-09 04:40): at matched selection budget the overlap decode's advantage VANISHES

DeepSeek self_harm, all three arms have 5 rounds, same selector, same 15 scenarios:

    arm                round 1            best-of-5        cost
    overlap elic-pick  81.3 @ 56.46%   100.0 @ 56.29%   2 API calls/token
    overlap comb b=1   26.0 @ 72.34%    67.3 @ 69.62%   2 API calls/token
    elicited-only     100.0 @ 50.86%   100.0 @ 56.14%   1 API call/token

elic-pick and elicited-only CONVERGE: 56.29 vs 56.14 at identical 100.0 presence, a 0.15pp
difference that is nothing. At round 1 the overlap decode looked like it bought plausibility
(56.46 at 81.3 presence vs 50.86 at 100.0). Five rounds of selection erase that -- selection
lifts elicited-only from 50.86 to 56.14 simply by keeping the most plausible of its
already-at-ceiling transcripts.

So on this cell the overlap decode buys NOTHING over elicited-only while costing 2x the API
calls (it must query both contexts per token; elicited-only queries one). Every capture
fraction in the grid below is a ROUND-1 quantity and likely overstates the method's value
once any selection budget exists.

NOT yet general -- this is ONE cell, which is the exact error made three times tonight.
The two 5-round jobs running (gpt-oss self_harm, GLM goblin) are elic-pick ONLY; their
elicited-only arms have 1 round, so they cannot test convergence. Matched 5-round
elicited-only controls are QUEUED for both.

This finding depended on a selector tie-break: at equal presence, take the most plausible
transcript. Without it an arm already at ceiling presence keeps round 1 forever and its
selection budget is silently discarded, which understated elicited-only as 100.0 @ 50.86%
and made the overlap arm look like a +1.9pp win. See rounds5.py.

## QUEUED next
  - [ ] elicited-only 5 rounds, gpt-oss self_harm   (matched control)
  - [ ] elicited-only 5 rounds, GLM goblin          (matched control)
  - [ ] a real selector instead of oracle: best-of-R by presence selects on the metric it
        reports, so 100.0 is an upper bound. margin_pick reached 85-99% of oracle in the
        paper work -- needed before any of these numbers are quotable.


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
  - [x] gpt-oss selfpres (done 00:50): 36.0 @ 69.77%  => CHAIN A gpt-oss COMPLETE

  **gpt-oss column A vs the grid** (capA = elicited-pick, capC = combined):

      beh              vanilla        combined       elic-pick       elic-only    capA    capC
      self-harm  17.3 @ 78.79%   16.0 @ 76.95%   21.3 @ 68.77%  100.0 @ 41.83%    4.8%   -1.6%
      goblin     10.0 @ 81.85%   10.0 @ 81.38%   10.0 @ 75.60%   84.7 @ 72.49%    0.0%    0.0%
      self-pres  30.0 @ 83.12%   28.0 @ 81.33%   36.0 @ 69.77%   97.3 @ 39.71%    8.9%   -3.0%

  elicited-pick beats combined on every gpt-oss cell (4.8>-1.6, 0.0=0.0, 8.9>-3.0), which is
  the direction the peakedness account predicts: choosing by the elicited ranking never has
  to overcome the target's top-1 dominance. But the magnitudes are small (best 8.9%) against
  DeepSeek self_harm's ~77%, and every gain is bought with ~10-13pp of plausibility
  (selfpres: +6.0 presence for -13.35pp). The model gate survives the better rule -- on
  gpt-oss the method is close to dead even at its best setting.

  - [x] GLM self_harm (done 00:54, 28m30s, 0 retries): 26.0 @ 64.96%
        vanilla 10.0 @ 70.60%, combined 11.3 @ 73.19%, elicited-only 78.0 @ 52.25%
        => capA 23.5% vs capC 2.0%. A real move, not gpt-oss's noise-scale one: +16.0
        presence for -5.6pp plausibility.

        self_harm under elicited-pick across models: DeepSeek 77.4%, GLM 23.5%, gpt-oss 4.8%.
        CAVEAT on the peakedness account: vanilla plausibility is DeepSeek 72.36, GLM 70.60,
        gpt-oss 78.79. That explains gpt-oss being worst, but NOT why GLM trails DeepSeek 3x
        while being slightly LESS peaked. Mean plausibility is a crude proxy anyway -- the
        real measure is top-1 mass at the decision points, which needs the top-5 collection
        run per model. Do not promote peakedness past "explains the gpt-oss end" yet.
  - [x] GLM goblin (done 01:37, 42m49s, 1 retry): 12.0 @ 72.91%
        vanilla 10.0 @ 70.32%, combined 10.0 @ 76.99%, elicited-only 90.7 @ 60.16%
        => capA 2.5% vs capC 0.0%. 12.0 vs 10.0 is 0.3 scenarios: noise.
        goblin is now immune on GLM (2.5%) and gpt-oss (0.0%) under BOTH pick rules, with
        ceilings of 90.7 and 84.7 sitting unused. Consistent with goblin tokens never
        entering the target's top-5 at all, which no re-ranking of that top-5 can fix.
  - [x] GLM selfpres (done 02:09, 30m52s): 58.0 @ 61.67% => capA 49.2% vs capC 16.1%
        +38.7 presence for only -1.83pp plausibility (63.50 -> 61.67). Best trade seen.
        => CHAIN A GLM COMPLETE

  **GLM column A vs the grid:**

      beh              vanilla        combined       elic-pick       elic-only    capA    capC
      self-harm  10.0 @ 70.60%   11.3 @ 73.19%   26.0 @ 64.96%   78.0 @ 52.25%   23.5%    2.0%
      goblin     10.0 @ 70.32%   10.0 @ 76.99%   12.0 @ 72.91%   90.7 @ 60.16%    2.5%    0.0%
      self-pres  19.3 @ 63.50%   32.0 @ 71.53%   58.0 @ 61.67%   98.0 @ 44.02%   49.2%   16.1%

  elicited-pick roughly TRIPLES capture on every GLM cell (23.5/2.0, 2.5/0.0, 49.2/16.1).

### DeepSeek degrading again (01:39) -- letting the current arm finish

429 retries, no completed arm in 48 min. Unlike the 00:25 stall this one IS progressing:
2 of 3 turn-batches done (one line per turn, batched lockstep over all 15 scenarios; 3 per
arm, calibrated against the 9 lines of the 3 completed gpt-oss arms).

Re-probe is bimodal, not uniformly slow: ttft 16.46s, 0.79s, 0.14s -- a cold-replica
pattern. At concurrency 15 many requests land on cold replicas, hence the retries, but the
warm ones carry it through.

Decision: let it run. It is two thirds done; killing now would discard 48 minutes to save a
timeout that is already being absorbed. Contrast 00:25, where the right call was to kill --
there it was 110 retries with ZERO turns completed, so nothing was being discarded.
  - [x] DeepSeek goblin (done 01:50, 58m21s, 497 retries, 0 CUT SHORT): 16.7 @ 68.67%
        vanilla 10.0 @ 71.96%, combined 14.0 @ 76.21%, elicited-only 96.7 @ 62.31%
        => capA 7.7% vs capC 4.6%.
        INTEGRITY CHECKED because of the retry count: all 15 transcripts carry 3 assistant
        turns with no empty content, identical to the vanilla reference. The retries were
        absorbed, not silently truncating -- consistent with 0 CUT SHORT.

        goblin under elicited-pick across models: DeepSeek 7.7%, GLM 2.5%, gpt-oss 0.0%.
        Hardest behaviour everywhere, under both rules.
  - [x] DeepSeek selfpres (done 02:29, 38m36s, 771 retries, 0 CUT SHORT): 81.3 @ 55.50%
        => capA 72.2% vs capC 19.6%. Integrity verified: 15 transcripts, 3 turns each.
        => CHAIN A DeepSeek COMPLETE => **COLUMN A COMPLETE, all 9 cells**

### COLUMN A RESULT (run `python -X utf8 experiments/postpaper/api_tilt/cells.py`)

elicited-pick beats combined on 9/9 cells. Under `combined` the grid topped out at 19.6%
and the conclusion drawn was "the model gates whether the method works at all". Under
elicited-pick DeepSeek captures 77.4% and 72.2% on two of three behaviours (81.3 presence
against a 17.3 vanilla). That earlier conclusion was substantially an ARTIFACT OF THE PICK
RULE, not a property of the models -- which is what the user suspected in asking for this
column, and it means the beta sweep was indeed run on the weaker rule.

Three separable factors, all real:
  * RULE      elicited-pick >> combined, universally (9/9)
  * MODEL     DeepSeek >> GLM >> gpt-oss
  * BEHAVIOUR goblin fails everywhere regardless of rule (max 7.7%)

### CORRECTION to "elicited-pick wins 9/9" -- b=2 beats it on gpt-oss selfpres

gpt-oss selfpres b=2 (done 02:34): 38.7 @ 77.83% => capB2 12.9%, against elic-pick's 8.9%.

    vanilla    30.0 @ 83.12%
    comb b=1   28.0 @ 81.33%   -3.0%
    comb b=2   38.7 @ 77.83%   12.9%   <- beats elic-pick on BOTH axes
    elic-pick  36.0 @ 69.77%    8.9%
    elic-only  97.3 @ 39.71%

The 9/9 claim was made against b=1 only and does NOT survive b=2. Here b=2 dominates:
higher presence (38.7 vs 36.0) AND higher plausibility (77.83% vs 69.77%). The presence gap
is 0.4 scenarios (noise); the 8pp plausibility gap is not. So state it as: b=2 matches
elic-pick's presence while keeping 8pp more plausibility -- a strictly better operating
point on this cell.

Mechanism, and why this is not a contradiction of the beta story below: the two rules
differ in KIND, not just degree. elicited-pick discards the target ranking entirely; b=2
still weights it. On a model whose top-1 is already strong -- gpt-oss has the highest
vanilla plausibility in the grid -- keeping some target weight buys plausibility at equal
presence. beta is a genuine dial, not merely a way-station on the road to elicited-only.

### GOBLIN COMPLETE across all 3 models x 3 overlap rules (03:50) -- immune

GLM goblin b=2: 10.0 @ 75.42% => 0.0%. Final goblin tally (capture %):

    model      b=1    b=2   elic-pick    ceiling (elicited-only)
    DeepSeek   4.6%   0.0%     7.7%       96.7
    GLM        0.0%   0.0%     2.5%       90.7
    gpt-oss    0.0%   0.0%     0.0%       84.7

Nine arms, maximum 7.7%, every ceiling 84.7-96.7 and unused. Goblin is immune to the
overlap approach under every rule and beta tested. With the gpt-oss evidence (presence
pinned at floor while plausibility falls monotonically), the account is that goblin tokens
are simply not in the target's top-5, and no rule restricted to re-ranking that set can
introduce them.

### The overlap constraint RAISES plausibility at low beta

Noticed across cells and worth stating: comb b=1 frequently has HIGHER plausibility than
vanilla -- goblin DeepSeek 71.96 -> 76.21, goblin GLM 70.32 -> 76.99, self_harm GLM
70.60 -> 73.19. Mechanically sensible: intersecting two top-5s keeps consensus tokens,
which the target itself also ranks highly, so the filter is a plausibility booster before
the elicited weight begins pulling against it. Explains why b=1 can look "free" on the
plausibility axis while delivering almost no behaviour.

### CORRECTION: beta is NOT a reliable interpolant (03:51)

DeepSeek selfpres b=2 = 38.0 @ 61.85% => cap 5.2%, BELOW b=1's 19.6%, on a cell whose
dynamic range is 34.7 -> 99.3. That is a genuine non-monotone case WITH signal, and it
refutes the "monotone on both signal-bearing cells" claim made 30 minutes earlier -- which
was drawn from exactly two cells. Same over-generalisation pattern as earlier tonight.

Full beta picture (capC -> capB2 -> capA):

    DeepSeek  self-harm   10.5 -> 37.1 -> 77.4   monotone
    DeepSeek  goblin       4.6 ->  0.0 ->  7.7   floor
    DeepSeek  self-pres   19.6 ->  5.2 -> 72.2   NON-MONOTONE, real signal
    GLM       self-harm    2.0 ->  7.8 -> 23.5   monotone
    GLM       goblin       0.0 ->  0.0 ->  2.5   floor
    gpt-oss   self-harm   -1.6 -> -4.0 ->  4.8   noise
    gpt-oss   goblin       0.0 ->  0.0 ->  0.0   floor
    gpt-oss   self-pres   -3.0 -> 12.9 ->  8.9   NON-MONOTONE, b=2 beats elic-pick

Two monotone, two non-monotone with signal in OPPOSITE directions, four at the floor.
beta=2 is cell-specific, not a dependable midpoint between b=1 and elicited-pick.

### What IS reliable: plausibility, 8/8 cells

Plausibility falls monotonically with elicited weight in every cell
(vanilla >= b=1 >= b=2 >= elic-pick >= elic-only), with a small b=1 BUMP in three cells
from the consensus-filter effect noted above.

So the trade is asymmetric: elicited weight costs plausibility PREDICTABLY and buys
presence UNPREDICTABLY. That is the practical case for elicited-pick as the default rule --
not that it is monotone, but that it is best-or-tied on presence in 8 of 9 cells, so it
does not require per-cell beta tuning to find the good operating point.

### (superseded) beta-monotonicity is only testable where there IS signal (03:24)

DeepSeek goblin b=2: 10.0 @ 73.86% => cap 0.0%, BELOW b=1's 4.6%. Not monotone.

    vanilla 10.0 | b=1 14.0 (4.6%) | b=2 10.0 (0.0%) | elic-pick 16.7 (7.7%) | elic-only 96.7

Not a counterexample: the whole cell spans 10.0-16.7 presence, ~1 scenario on n=15, with
goblin pinned near the judge floor throughout. Same as gpt-oss self_harm's apparent
reversal. Tally: monotone on BOTH signal-bearing cells (DeepSeek self_harm, GLM self_harm),
untestable on the near-floor cells. Integrity verified: 15 transcripts, 3 turns each.

### beta-convergence: SECOND confirmation on GLM self_harm (03:11)

    b=1 2.0%  ->  b=2 7.8%  ->  elic-pick 23.5%    (11.3 -> 15.3 -> 26.0 presence)

Monotone, as predicted. Two signal-bearing cells now agree (DeepSeek self_harm below).
Consistent picture: where the cell carries real signal, beta moves monotonically toward
elicited-pick and elicited-pick wins on PRESENCE. gpt-oss selfpres stays the one overshoot,
and there the advantage that mattered was PLAUSIBILITY, not presence.

### beta-convergence question: RESOLVED by DeepSeek self_harm

combined = l_target + beta*l_elicited, so raising beta should move the argmax toward
elicited-pick's, putting b=2 between b=1 and elic-pick. DeepSeek self_harm confirms it:

    b=1 10.5%  ->  b=2 37.1%  ->  elic-pick 77.4%      (26.0 -> 48.0 -> 81.3 presence)

Monotone in beta, as predicted. The gpt-oss self_harm reversal (b=2 -4.0% below b=1 -1.6%)
was noise -- that whole cell spans one scenario. Signal-bearing cells settle it. -- endpoint recovered 00:50,
        ttft 0.97/0.26/0.13s, back to normal; the degradation was a transient ~25min window.
        Reuses the cfg/ideation/understanding kept from the killed run.

### Verified non-issue: `b2=4.0` in the overlap banner

Every overlap run ever logged prints `[jailbroken_output] ... (b1=1.0, b2=4.0, floor=0.0)`,
including all nine grid cells. It is NOT applied: `_tilt_generate` returns `_driven_overlap`
at the top when `api_rule == "overlap"`, before b1/b2 are read (apitilt.py:626). The banner
is rollout.py's generic tilt line, printed whatever the engine. The knobs that do apply are
on the next line, `[api_tilt rule=overlap pick=... beta=... fb=...]`. Left as-is rather than
patching a paper file for a cosmetic string.
  - [ ] DeepSeek goblin, selfpres   <- LAST, re-probe DeepSeek latency before launching

### Column B running (02:10-)

  - [x] gpt-oss goblin b=2 (done 02:25, 6m46s, 0 retries): 10.0 @ 79.13% => cap 0.0%

        CLEANEST MECHANISTIC EVIDENCE SO FAR. All four overlap arms sit at exactly 10.0
        presence (the judge floor) while plausibility falls monotonically as the elicited
        term gains weight:

            vanilla       10.0 @ 81.85%
            combined b=1  10.0 @ 81.38%
            combined b=2  10.0 @ 79.13%
            elic-pick     10.0 @ 75.60%
            elic-only     84.7 @ 72.49%

        The intervention is demonstrably steering the decode -- plausibility moves, so
        different tokens are being emitted -- yet the behaviour never appears. So the
        failure is NOT an inert intervention. It is that goblin tokens are absent from the
        target's top-5 at all, and no re-ranking restricted to that set can find them.
        elicited-only reaches 84.7 precisely because it is not restricted to it.

  - [x] gpt-oss self_harm b=2 (done 02:18, 8m4s): 14.0 @ 76.13% => cap -4.0%
        vs b=1 -1.6%, elic-pick +4.8%, vanilla 17.3 @ 78.79%.

        PREDICTION CHECK, flagged for GLM/DeepSeek to settle: combined scores
        l_target + beta*l_elicited, so as beta grows the argmax converges to elicited-pick.
        b=2 should therefore land BETWEEN b=1 and elic-pick. Here it goes the wrong way.
        But the full spread on this cell is 14.0-21.3, about one scenario on n=15 -- all
        four arms are mutually indistinguishable, so gpt-oss self_harm cannot decide it.
        GLM and DeepSeek can: their capA-capC gaps are 20-45pp, far outside noise.

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
