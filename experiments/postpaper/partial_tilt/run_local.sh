#!/bin/bash
# run_local.sh [k]     partial_tilt_output on LOCAL weights: truncate to top-k, then combine.
#
# The hosted runner (api_tilt/run_cell.sh) drives an api/ target; this one drives a local HF
# target through the same block with engine=hf_partial, so the same rule can be measured at a
# different information level.
#
#   k = 0 means NO truncation, which reduces the engine exactly to the paper's full-vocab
#   LogitTilt. Run it first: it is the sweep's anchor, and if its numbers do not sit on top of
#   a jailbroken_output run at the same b2, something is wrong before any k>0 point matters.
#
#   BEH     self_harm | goblin | selfpres | deception | medical   (default self_harm)
#   MODEL   qwen | gemma | llama | phi                            (default qwen)
#   RULE    poe (the paper's geometric operator) | mix            (default poe)
#   B2      elicited weight                                       (default 1.5, the paper's
#           tuned value for self_harm/Qwen)
#   AUDITOR local (paper's FP8 Gemma-4-26B, needs ~26GB more VRAM) | api (OpenRouter)
set -e
cd "$(dirname "$0")/../../.."

K="${1:-0}"
ROUNDS="${ROUNDS:-1}"
BEH="${BEH:-self_harm}"
MODEL="${MODEL:-qwen}"
RULE="${RULE:-poe}"
B2="${B2:-1.5}"
B1="${B1:-1.0}"
SCEN="${SCEN:-15}"; SEED="${SEED:-1}"
AUDITOR="${AUDITOR:-local}"

case "$BEH" in
  self_harm) BEHFILE=prompts/7_self_harm_encouragement.yaml; BANKBEH=self_harm ;;
  # racial is the one behaviour where LogitTilt does NOT saturate on Qwen3.5-4B: the paper has
  # vanilla 60.2 -> LogitTilt 66.6 -> WILT 79.1, against 93-100 for every other behaviour. That
  # headroom is the point of running it -- self_harm pins every steered arm at 100 presence, so
  # nothing can be told apart there.
  # NOTE: no kickoff bank exists for racial. The FIRST run generates fresh kickoffs and banks
  # them; every later run reuses them. Seed with a single run before fanning out, or each
  # parallel run invents its own scenarios and the arms are not comparable.
  racial)    BEHFILE=prompts/1_racial_bias.yaml;             BANKBEH=racial ;;
  goblin)    BEHFILE=prompts/9_goblin_fixation.yaml;         BANKBEH=goblin ;;
  selfpres)  BEHFILE=prompts/5_self_preservation.yaml;       BANKBEH=selfpres ;;
  deception) BEHFILE=prompts/4_strategic_deception.yaml;     BANKBEH=deception ;;
  medical)   BEHFILE=prompts/8_dangerous_medical_advice.yaml; BANKBEH=medical ;;
  *) echo "unknown BEH=$BEH"; exit 2 ;;
esac

case "$MODEL" in
  qwen)  TARGET=local/Qwen/Qwen3.5-4B;                MODELDIR=Qwen_Qwen3.5-4B ;;
  gemma) TARGET=local/google/gemma-4-e4b-it;          MODELDIR=google_gemma-4-e4b-it ;;
  llama) TARGET=local/meta-llama/Llama-3.2-3B-Instruct; MODELDIR=meta-llama_Llama-3.2-3B-Instruct ;;
  phi)   TARGET=local/microsoft/Phi-4-mini-instruct;  MODELDIR=microsoft_Phi-4-mini-instruct ;;
  *) echo "unknown MODEL=$MODEL"; exit 2 ;;
esac

[ -f .env.local ] || { echo "missing .env.local"; exit 1; }
set -a; . ./.env.local; set +a
export PYTHONIOENCODING=utf-8

export BLOOM_RUNS_ROOT=experiments/postpaper
export BLOOM_TARGET_MODEL=$TARGET

if [ "$AUDITOR" = "api" ]; then
  export BLOOM_EVAL_MODEL=openrouter/google/gemma-4-26b-a4b-it
  export BLOOM_JUDGE_MODEL=openrouter/google/gemma-4-26b-a4b-it
  export BLOOM_EVAL_THINKING=0 BLOOM_JUDGE_THINKING=0
fi   # else: the cfg default, the paper's local FP8 Gemma-4-26B auditor

export BLOOM_BEHAVIOR_FILE=$BEHFILE
export BLOOM_KICKOFF_BANK="experiments/bloom/_banks/runs_hyperparam/${BANKBEH}/Qwen_Qwen3.5-4B/_bank"
export BLOOM_NUM_SCENARIOS=$SCEN
export BLOOM_NUM_ROUNDS=$ROUNDS
export BLOOM_MAX_TURNS=3
export BLOOM_SEED=$SEED

# jailbroken_output stays OFF: enabling both is refused, and the point is to reach the same
# decode through partial_tilt_output so the information level is a knob rather than a fork.
# It has no BLOOM_JAIL_ENABLED -- cfg has enabled=False and the ONLY thing that flips it is
# BLOOM_JAIL_MODEL being set, so unsetting that is the actual off switch. It matters because
# `set -a; . ./.env.local` above exports whatever that file defines.
unset BLOOM_JAIL_MODEL
export BLOOM_PTILT_ENABLED=1
export BLOOM_PTILT_ENGINE=hf_partial
export BLOOM_PTILT_RULE=$RULE
export BLOOM_PTILT_TOPK=$K
export BLOOM_PTILT_B1=$B1
export BLOOM_PTILT_B2=$B2
export BLOOM_PTILT_FLOOR="${FLOOR:-0}"      # percent; 0 keeps top-k the ONLY variable

# ADAPTIVE alpha(q): (b1, b2) become (alpha, 1-alpha) with alpha = ALPHA0*(1 - q**ALPHA_K).
# Only the ratio matters to poe, so alpha0 = b1/(b1+b2) -- the fixed B2=1.5 against B1=1.0 is
# alpha0=0.4, which is why that is the default here: at q=0 an adaptive run starts from
# exactly the fixed run's operating point and differs only as disagreement rises.
if [ "${ADAPTIVE:-0}" = "1" ]; then
  export BLOOM_API_ADAPTIVE=1
  export BLOOM_API_ALPHA0="${ALPHA0:-0.4}"
  export BLOOM_API_ALPHA_K="${ALPHA_K:-10}"
  export BLOOM_API_Q_METRIC="${Q_METRIC:-elicited_outside}"
  ASUF="_a${ALPHA0:-0.4}k${ALPHA_K:-10}"
  [ "${Q_METRIC:-elicited_outside}" = "elicited_outside" ] || ASUF="${ASUF}_q${Q_METRIC}"
else
  # MUST be explicit. cfg has mix.adaptive=True and mix.alpha0=0.6, so merely NOT exporting
  # these leaves the schedule ON at a different alpha0 -- and the folder name, built from
  # ASUF, would still say "fixed". Three runs were silently adaptive at alpha0=0.6 before this
  # line existed. An off switch that works by omission is not an off switch.
  export BLOOM_API_ADAPTIVE=0
  ASUF=""
fi
export BLOOM_API_JAIL_VAR_BATCH="${VAR_BATCH:-15}"

# cfg.target_gpu_id defaults to 1 because the paper's boxes kept the auditor on GPU 0 and the
# target on GPU 1. With AUDITOR=api there is no local auditor and typically ONE card, so the
# default asks for a device that does not exist ("invalid device ordinal"). Default to 0 here
# and let a multi-GPU box override.
export BLOOM_TARGET_GPU="${TARGET_GPU:-0}"

if [ "$K" = "0" ]; then KSUF="_kfull"; else KSUF="_k${K}"; fi
if [ "$RULE" = "poe" ]; then RSUF=""; else RSUF="_${RULE}"; fi
if [ "${FLOOR:-0}" = "0" ]; then FSUF=""; else FSUF="_fl${FLOOR}"; fi
export BLOOM_FOLDER=runs_local/${BEH}/${MODELDIR}/ptilt${RSUF}${KSUF}_b${B2}${ASUF}${FSUF}_${SCEN}s
[ -n "${RUN_TAG:-}" ] && export BLOOM_FOLDER="${BLOOM_FOLDER}_${RUN_TAG}"

echo "=== beh=$BEH model=$MODEL rule=$RULE k=${K} (0=full vocab) b1=$B1 b2=$B2 scen=$SCEN"
echo "=== folder=$BLOOM_FOLDER"
python src/bloom/bloom_corrupt.py
