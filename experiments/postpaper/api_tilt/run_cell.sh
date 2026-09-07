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
#   BLOOM_API_JAIL_B2   elicited-term weight in the union mixture (default 1)
#   BLOOM_API_FLOOR     min target prob (percent) for an emitted token, BOTH stages
#   BLOOM_API_STAGE2    threshold | never      BLOOM_API_STAGE2_THETA  escalate iff q >= theta
#   BLOOM_API_FLOOR_ACTION  repick | stage2    BLOOM_API_FALLBACK  jail_resample | target_sample
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

ROOT=runs_dsv4/${BEH}/${MODELDIR}
case "$ARM" in
  vanilla)
    # BLOOM_TARGET_TEMP=0 gives greedy/argmax target decoding; the folder records it.
    TT="${BLOOM_TARGET_TEMP:-}"
    if [ -n "$TT" ] && [ "$TT" != "1" ]; then TSUF="_t${TT}"; else TSUF=""; fi
    export BLOOM_FOLDER=${ROOT}/api_vanilla${TSUF}_15s
    export BLOOM_API_JAIL_ENABLED=0 ;;          # disabled = the un-steered corner
  elicited)
    export BLOOM_FOLDER=${ROOT}/api_elicited_15s
    export BLOOM_API_JAIL_ENABLED=1 BLOOM_API_JAIL_B1=0 BLOOM_API_JAIL_B2=1 ;;
  overlap)
    B2="${BLOOM_API_JAIL_B2:-1}"
    FL="${BLOOM_API_FLOOR:-1e-05}"
    TH="${BLOOM_API_STAGE2_THETA:-0.95}"
    S2="${BLOOM_API_STAGE2:-threshold}"
    FA="${BLOOM_API_FLOOR_ACTION:-stage2}"
    FB="${BLOOM_API_FALLBACK:-jail_descend}"
    export BLOOM_API_JAIL_ENABLED=1 BLOOM_API_RULE=overlap
    export BLOOM_API_JAIL_B2=$B2 BLOOM_API_FLOOR=$FL BLOOM_API_STAGE2=$S2            BLOOM_API_STAGE2_THETA=$TH BLOOM_API_FLOOR_ACTION=$FA BLOOM_API_FALLBACK=$FB
    if [ "$B2" = "1" ];              then BSUF="";  else BSUF="_b${B2}";   fi
    if [ "$FL" = "1e-05" ];          then LSUF="";  else LSUF="_fl${FL}";  fi
    if [ "$S2" = "threshold" ];      then S2SUF=""; else S2SUF="_s2${S2}"; fi
    if [ "$TH" = "0.95" ];           then THSUF=""; else THSUF="_th${TH}"; fi
    if [ "$FA" = "stage2" ];         then ASUF="";  else ASUF="_${FA}";    fi
    if [ "$FB" = "jail_descend" ];   then FSUF="";  else FSUF="_fb${FB}";  fi
    AD="${BLOOM_API_ADAPTIVE:-1}"
    if [ "$AD" = "1" ]; then
      A0="${BLOOM_API_ALPHA0:-0.5}"; AK="${BLOOM_API_ALPHA_K:-10}"
      export BLOOM_API_ADAPTIVE=1 BLOOM_API_ALPHA0=$A0 BLOOM_API_ALPHA_K=$AK
      ADSUF="_a${A0}k${AK}"
    else ADSUF=""; fi
    TE="${BLOOM_API_TARGET_EVERY:-0}"
    if [ "$TE" = "0" ]; then TESUF=""; else TESUF="_te${TE}"; export BLOOM_API_TARGET_EVERY=$TE; fi
    DET="${BLOOM_API_DET_FALLBACK:-1}"
    export BLOOM_API_DET_FALLBACK=$DET
    if [ "$DET" = "1" ]; then DSUF=""; else DSUF="_nodet"; fi
    if [ "${BLOOM_API_JAIL_PREFILL:-1}" = "0" ]; then PSUF="_nopf"; else PSUF=""; fi
    export BLOOM_FOLDER=${ROOT}/api_mix${BSUF}${LSUF}${S2SUF}${THSUF}${ASUF}${FSUF}${TESUF}${ADSUF}${DSUF}${PSUF}_15s ;;
  *) echo "usage: run_cell.sh [vanilla|elicited|overlap] [rounds]"; exit 2 ;;
esac

# RUN_TAG appends a suffix to the folder so an identical config can be repeated
# without the round-level resume treating it as already done. Same seed, so the
# scenario bank and opening turns are shared and the repeat isolates run-to-run
# non-determinism in the hosted model.
if [ -n "${RUN_TAG:-}" ]; then export BLOOM_FOLDER="${BLOOM_FOLDER%_15s}_${RUN_TAG}_15s"; fi
echo "=== beh=$BEH model=$MODEL arm=$ARM rounds=$ROUNDS scen=$SCEN seed=$SEED"
echo "=== folder=$BLOOM_FOLDER"
python src/bloom/bloom_corrupt.py
