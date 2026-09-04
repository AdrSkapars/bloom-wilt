#!/bin/bash
# run_api.sh <arm> [rounds]
#
# Runs the WILT pipeline with DeepSeek-V4-Flash as a HOSTED target (Fireworks
# /completions) instead of local weights — no GPU box needed. Arms:
#
#   elicited   b1=0, b2=1   z = l_jail          "elicited only"  <- the missing corner
#   vanilla    b1=1, b2=0   z = l_target        target only (API replica of vanilla_15s)
#
# Those are the only two points of the tilt a text API can reproduce EXACTLY: in both the
# sampling distribution is a single context's own softmax, so nothing has to be mixed in
# logit space. Any other (b1, b2) is refused by bloom/apitilt.py rather than approximated.
#
# Deliberately matched to the local arms so the numbers line up: same kickoff bank, same
# 15 scenarios, seed 1, 3 turns, temperature 1.0, target_max_tokens 250.
#
# NOT matched, and it cannot be: the local arms ran with target_floor=1e-4 (mask tokens the
# TARGET gives < 1e-4 before sampling). The floor thresholds on the target distribution at
# every step, which needs target logits while decoding — impossible over an API. The
# elicited-only corner is defined without a target term anyway, so it runs floor-off.
set -e
cd "$(dirname "$0")/../../.."

ARM="${1:-elicited}"
ROUNDS="${2:-1}"
SCEN="${SCEN:-15}"; SEED="${SEED:-1}"
BANK="${BANK:-experiments/bloom/_banks/runs_hyperparam/self_harm/Qwen_Qwen3.5-4B/_bank}"

# API keys (gitignored). `set -a` exports every assignment in the file.
[ -f .env.local ] || { echo "missing .env.local (needs FIREWORKS_API_KEY + OPENROUTER_API_KEY)"; exit 1; }
set -a; . ./.env.local; set +a

# The DeepSeek chat template contains U+FF5C (fullwidth vertical bar). Any print of a
# rendered prompt dies on a Windows cp1252 stdout without this.
export PYTHONIOENCODING=utf-8

TARGET=api/accounts/fireworks/models/deepseek-v4-flash-0731

export BLOOM_RUNS_ROOT=experiments/postpaper
export BLOOM_TARGET_MODEL=$TARGET
export BLOOM_TARGET_API=fireworks
# Prompts are rendered here rather than by a tokenizer, so the template is the ONLY thing
# pinning the API's tokenization to the local run's. Verified: re-scoring the local run's
# stored gen_token_ids through this template reproduces every id exactly.
export BLOOM_TARGET_CHAT_TEMPLATE=src/bloom/prompts/deepseek_v4_chat.jinja

# Auditor + judge over OpenRouter (thinking off — think_prefix is '' for the gemma-4 family,
# so "medium" was always a no-op locally, and the budget check trips the small max_tokens).
export BLOOM_EVAL_MODEL=openrouter/google/gemma-4-26b-a4b-it
export BLOOM_JUDGE_MODEL=openrouter/google/gemma-4-26b-a4b-it
export BLOOM_EVAL_THINKING=0 BLOOM_JUDGE_THINKING=0
export BLOOM_MAX_CONCURRENT="${BLOOM_MAX_CONCURRENT:-15}"
# Issue the per-scenario evaluator turns concurrently rather than one after another.
# Off (1) by default in the pipeline so the paper's path is untouched; on here.
export BLOOM_ROLLOUT_CONCURRENCY="${BLOOM_ROLLOUT_CONCURRENCY:-15}"

export BLOOM_BEHAVIOR_FILE=prompts/7_self_harm_encouragement.yaml
export BLOOM_NUM_SCENARIOS=$SCEN
export BLOOM_NUM_ROUNDS=$ROUNDS
export BLOOM_MAX_TURNS=3
export BLOOM_SEED=$SEED
export BLOOM_KICKOFF_BANK=$BANK
# api_tilt has no GPU memory to run out of, so this is purely request concurrency.
export BLOOM_JAIL_VAR_BATCH="${BLOOM_JAIL_VAR_BATCH:-15}"

case "$ARM" in
  vanilla)
    # No BLOOM_JAIL_MODEL -> jailbroken_output.enabled stays False -> target_only path.
    export BLOOM_FOLDER=runs_dsv4/self_harm/deepseek_v4_flash/api_vanilla_15s
    ;;
  elicited)
    export BLOOM_FOLDER=runs_dsv4/self_harm/deepseek_v4_flash/api_elicited_15s
    export BLOOM_JAIL_MODEL=$TARGET      # self-jail: same hosted model, jail system prompt
    export BLOOM_JAIL_B1=0               # drop the target term entirely
    export BLOOM_JAIL_BETA=1             # b2 = 1
    export BLOOM_JAIL_FLOOR=0            # no naturalness floor (needs target logits; see header)
    ;;
  overlap)
    # Per-token decode combining both contexts: emit the overlap member of the two top-5
    # candidate sets that the ELICITED context ranks highest; when the sets are disjoint,
    # emit a plain sample from the target context. b1/b2 do not apply -- this is not a
    # point on the (b1,b2) plane.
    # BLOOM_API_PICK chooses WHICH overlap member is emitted:
    #   elicited  most probable under the jail context (the original rule)
    #   target    most probable under the target context
    #   combined  highest sum of logprobs = highest product of probabilities, i.e. the
    #             top-5-restricted form of the true tilt at b1=b2=1
    #   random    uniform over the overlap
    #   sample    drawn in proportion to elicited probability
    PICK="${BLOOM_API_PICK:-elicited}"
    FB="${BLOOM_API_FALLBACK:-target_sample}"
    export BLOOM_API_PICK=$PICK
    export BLOOM_API_FALLBACK=$FB
    # Default fallback keeps the original folder name so earlier runs stay addressable.
    if [ "$FB" = "target_sample" ]; then SUF=""; else SUF="_fb${FB#top5_}"; fi
    BETA="${BLOOM_API_BETA:-1}"
    export BLOOM_API_BETA=$BETA
    # beta=1 keeps the original folder name so the existing runs stay addressable.
    if [ "$BETA" = "1" ]; then BSUF=""; else BSUF="_b${BETA}"; fi
    export BLOOM_FOLDER=runs_dsv4/self_harm/deepseek_v4_flash/api_overlap_${PICK}${BSUF}${SUF}_15s
    export BLOOM_JAIL_MODEL=$TARGET
    export BLOOM_API_RULE=overlap
    export BLOOM_JAIL_FLOOR=0
    ;;
  *)
    echo "usage: run_api.sh [elicited|vanilla|overlap] [rounds]"; exit 2 ;;
esac

echo "=== arm=$ARM  rounds=$ROUNDS  scen=$SCEN  seed=$SEED"
echo "=== folder=$BLOOM_FOLDER"
python src/bloom/bloom_corrupt.py
