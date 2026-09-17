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
export BLOOM_API_JAIL_VAR_BATCH="${VAR_BATCH:-15}"

if [ "$K" = "0" ]; then KSUF="_kfull"; else KSUF="_k${K}"; fi
if [ "$RULE" = "poe" ]; then RSUF=""; else RSUF="_${RULE}"; fi
if [ "${FLOOR:-0}" = "0" ]; then FSUF=""; else FSUF="_fl${FLOOR}"; fi
export BLOOM_FOLDER=runs_local/${BEH}/${MODELDIR}/ptilt${RSUF}${KSUF}_b${B2}${FSUF}_${SCEN}s
[ -n "${RUN_TAG:-}" ] && export BLOOM_FOLDER="${BLOOM_FOLDER}_${RUN_TAG}"

echo "=== beh=$BEH model=$MODEL rule=$RULE k=${K} (0=full vocab) b1=$B1 b2=$B2 scen=$SCEN"
echo "=== folder=$BLOOM_FOLDER"
python src/bloom/bloom_corrupt.py
