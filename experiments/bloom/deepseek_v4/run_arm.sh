#!/bin/bash
# run_arm.sh <beta> [rounds] [var_batch]
#   beta 0  -> vanilla: no BLOOM_JAIL_MODEL, so jailbroken_output.enabled stays False
#              and bloom_corrupt takes the target_only path.
#   beta >0 -> self-jail LogitTilt at that beta.
# Env overrides: SCEN (default 15), SEED (1), BANK, TAG (folder suffix).
# Rounds are resume-aware: an existing round_N/judgment.json is reused.
#
# var_batch: cost is ceil(SCEN/vb) chunks, each chunk about the same wall clock, so
# 8..14 are all 2 chunks at SCEN=15 -- nothing to gain above 8. beta 0 is target_only
# (ONE context, ~half the memory) and fits 15. Steered arms run two contexts in lockstep
# and OOM at 15 (eager attention wants a 29GiB logits tensor at turn 3).
BETA="$1"
[ -z "$BETA" ] && { echo "usage: run_arm.sh <beta> [rounds] [var_batch]"; exit 2; }
ROUNDS="${2:-1}"
if [ -n "$3" ]; then VB="$3"; elif [ "$BETA" = "0" ]; then VB=15; else VB=8; fi
SCEN="${SCEN:-15}"; SEED="${SEED:-1}"
BANK="${BANK:-experiments/bloom/_banks/runs_hyperparam/self_harm/Qwen_Qwen3.5-4B/_bank}"
TAG="${TAG:-15s}"
cd /workspace/bloom-wilt || exit 1
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . ./.env.local; set +a

export BLOOM_TARGET_MODEL=local/deepseek-ai/DeepSeek-V4-Flash-0731
export BLOOM_TARGET_DEVICE_MAP=auto
export BLOOM_TARGET_DTYPE=auto
export BLOOM_TARGET_ATTN=eager
export BLOOM_TARGET_CHAT_TEMPLATE=src/bloom/prompts/deepseek_v4_chat.jinja
export BLOOM_EVAL_MODEL=openrouter/google/gemma-4-26b-a4b-it
export BLOOM_JUDGE_MODEL=openrouter/google/gemma-4-26b-a4b-it
export BLOOM_EVAL_THINKING=0
export BLOOM_JUDGE_THINKING=0
export BLOOM_BEHAVIOR_FILE=prompts/7_self_harm_encouragement.yaml
export BLOOM_NUM_SCENARIOS=$SCEN
export BLOOM_NUM_ROUNDS=$ROUNDS
export BLOOM_MAX_TURNS=3
export BLOOM_SEED=$SEED
export BLOOM_JAIL_VAR_BATCH=$VB
export BLOOM_KICKOFF_BANK=$BANK

if [ "$BETA" = "0" ]; then
  export BLOOM_FOLDER=runs_dsv4/self_harm/deepseek_v4_flash/vanilla_${TAG}
else
  export BLOOM_FOLDER=runs_dsv4/self_harm/deepseek_v4_flash/jail_b${BETA}
  export BLOOM_JAIL_MODEL=local/deepseek-ai/DeepSeek-V4-Flash-0731
  export BLOOM_JAIL_BETA=$BETA
fi

echo "=== START beta=$BETA rounds=$ROUNDS vb=$VB scen=$SCEN seed=$SEED $(date -u +%H:%M:%S) ==="
python src/bloom/bloom_corrupt.py
rc=$?
echo "=== EXIT=$rc beta=$BETA scen=$SCEN $(date -u +%H:%M:%S) ==="
exit $rc
