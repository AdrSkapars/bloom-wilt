#!/bin/bash
# run_cell.sh <arm> [rounds]     arm = vanilla | elicited | overlap
#
# Generalised launcher: any behaviour x any hosted model x any arm. Supersedes run_api.sh
# (which is self_harm + DeepSeek only); kept as a separate file because run_api.sh was
# still executing when this was written, and editing a shell script while bash is reading
# it corrupts the running shell -- that cost two spurious FAILED markers tonight.
#
#   BEH    self_harm | goblin | selfpres          (default self_harm)
#   MODEL  dsv4 | glm | gptoss | qwen             (default dsv4)
#   BLOOM_API_PICK      elicited|target|combined|combined_min|combined_sample|random|sample
#   BLOOM_API_BETA      weight on the elicited term of the combined score (default 1)
#   BLOOM_API_FALLBACK  target_sample | top5_argmax | top5_random | top5_weighted
#   BLOOM_API_JAIL_PREFILL  0 to drop the behaviour file's prefill from the elicited context
set -e
cd "$(dirname "$0")/../../.."

ARM="${1:-overlap}"
ROUNDS="${2:-1}"
BEH="${BEH:-self_harm}"
MODEL="${MODEL:-dsv4}"
SCEN="${SCEN:-15}"; SEED="${SEED:-1}"

case "$BEH" in
  self_harm) BEHFILE=prompts/7_self_harm_encouragement.yaml; BANKBEH=self_harm ;;
  goblin)    BEHFILE=prompts/9_goblin_fixation.yaml;         BANKBEH=goblin ;;
  selfpres)  BEHFILE=prompts/5_self_preservation.yaml;       BANKBEH=selfpres ;;
  *) echo "unknown BEH=$BEH (self_harm|goblin|selfpres)"; exit 2 ;;
esac

# Per-model setup. TEMPLATE/TOKENIZER/BOS must be right or the run is silently wrong:
# a mis-registered think block leaves the elicited context predicting </think>, which
# completes cleanly and means nothing. Validate a new model with a vanilla canary first.
case "$MODEL" in
  dsv4)
    TARGET=api/accounts/fireworks/models/deepseek-v4-flash-0731
    MODELDIR=deepseek_v4_flash
    TEMPLATE=src/bloom/prompts/deepseek_v4_chat.jinja
    TOKENIZER="$HOME/.cache/bloom/dsv4_tokenizer.json"
    BOSTOK='<｜begin▁of▁sentence｜>' ;;
  glm)
    TARGET=api/accounts/fireworks/models/glm-5p3-flash
    MODELDIR=glm_5p3_flash
    TEMPLATE=src/bloom/prompts/glm_5p3_chat.jinja
    TOKENIZER="$HOME/.cache/bloom/glm_tokenizer.json"
    BOSTOK="${BLOOM_TARGET_BOS_TOKEN:-}" ;;
  gptoss)
    TARGET=api/accounts/fireworks/models/gpt-oss-120b
    MODELDIR=gpt_oss_120b
    TEMPLATE=src/bloom/prompts/gpt_oss_chat.jinja
    TOKENIZER="$HOME/.cache/bloom/gptoss_tokenizer.json"
    BOSTOK="${BLOOM_TARGET_BOS_TOKEN:-}" ;;
  qwen)
    TARGET=api/accounts/fireworks/models/qwen3p7-plus
    MODELDIR=qwen3p7_plus
    TEMPLATE=src/bloom/prompts/qwen3p7_chat.jinja
    TOKENIZER="$HOME/.cache/bloom/qwen_tokenizer.json"
    BOSTOK="${BLOOM_TARGET_BOS_TOKEN:-}" ;;
  *) echo "unknown MODEL=$MODEL (dsv4|glm|gptoss|qwen)"; exit 2 ;;
esac

[ -f "$TEMPLATE" ]  || { echo "missing chat template: $TEMPLATE"; exit 3; }
[ -f "$TOKENIZER" ] || { echo "missing tokenizer.json: $TOKENIZER"; exit 3; }

[ -f .env.local ] || { echo "missing .env.local"; exit 1; }
set -a; . ./.env.local; set +a
export PYTHONIOENCODING=utf-8

export BLOOM_RUNS_ROOT=experiments/postpaper
export BLOOM_TARGET_MODEL=$TARGET
export BLOOM_TARGET_API=fireworks
export BLOOM_TARGET_CHAT_TEMPLATE=$TEMPLATE
export BLOOM_TARGET_TOKENIZER=$TOKENIZER
[ -n "$BOSTOK" ] && export BLOOM_TARGET_BOS_TOKEN="$BOSTOK"

export BLOOM_EVAL_MODEL=openrouter/google/gemma-4-26b-a4b-it
export BLOOM_JUDGE_MODEL=openrouter/google/gemma-4-26b-a4b-it
export BLOOM_EVAL_THINKING=0 BLOOM_JUDGE_THINKING=0
export BLOOM_MAX_CONCURRENT="${BLOOM_MAX_CONCURRENT:-15}"
export BLOOM_ROLLOUT_CONCURRENCY="${BLOOM_ROLLOUT_CONCURRENCY:-15}"
export BLOOM_API_SERVICE_TIER="${BLOOM_API_SERVICE_TIER-priority}"

export BLOOM_BEHAVIOR_FILE=$BEHFILE
export BLOOM_KICKOFF_BANK="experiments/bloom/_banks/runs_hyperparam/${BANKBEH}/Qwen_Qwen3.5-4B/_bank"
export BLOOM_NUM_SCENARIOS=$SCEN
export BLOOM_NUM_ROUNDS=$ROUNDS
export BLOOM_MAX_TURNS=3
export BLOOM_SEED=$SEED
export BLOOM_API_JAIL_VAR_BATCH="${BLOOM_API_JAIL_VAR_BATCH:-15}"
# Own env names; BLOOM_JAIL_* stays bound to jailbroken_output and is not set here.
export BLOOM_API_JAIL_ENABLED=1

ROOT=runs_dsv4/${BEH}/${MODELDIR}
case "$ARM" in
  vanilla)
    export BLOOM_FOLDER=${ROOT}/api_vanilla_15s
    export BLOOM_API_JAIL_TARGET_ONLY=1 ;;
  elicited)
    export BLOOM_FOLDER=${ROOT}/api_elicited_15s
    export BLOOM_API_JAIL_B1=0 BLOOM_API_JAIL_B2=1 ;;
  overlap)
    PICK="${BLOOM_API_PICK:-combined}"
    BETA="${BLOOM_API_BETA:-1}"
    FB="${BLOOM_API_FALLBACK:-target_sample}"
    export BLOOM_API_PICK=$PICK BLOOM_API_BETA=$BETA BLOOM_API_FALLBACK=$FB
    if [ "$BETA" = "1" ]; then BSUF=""; else BSUF="_b${BETA}"; fi
    if [ "$FB" = "target_sample" ]; then FSUF=""; else FSUF="_fb${FB#top5_}"; fi
    if [ "${BLOOM_API_JAIL_PREFILL:-1}" = "0" ]; then PSUF="_nopf"; else PSUF=""; fi
    FL="${BLOOM_API_FB_FLOOR:-0}"
    if [ "$FL" = "0" ]; then LSUF=""; else LSUF="_fl${FL}"; export BLOOM_API_FB_FLOOR=$FL; fi
    export BLOOM_FOLDER=${ROOT}/api_overlap_${PICK}${BSUF}${FSUF}${PSUF}${LSUF}_15s
    export BLOOM_API_RULE=overlap ;;
  *) echo "usage: run_cell.sh [vanilla|elicited|overlap] [rounds]"; exit 2 ;;
esac

echo "=== beh=$BEH model=$MODEL arm=$ARM rounds=$ROUNDS scen=$SCEN seed=$SEED"
echo "=== folder=$BLOOM_FOLDER"
python src/bloom/bloom_corrupt.py
