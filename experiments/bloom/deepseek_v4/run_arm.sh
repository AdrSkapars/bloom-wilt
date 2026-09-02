#!/bin/bash
# run_arm.sh <beta> [rounds]
#   beta 0  -> vanilla: no BLOOM_JAIL_MODEL, so jailbroken_output.enabled stays False
#              and bloom_corrupt takes the target_only path (folder vanilla_15s).
#   beta >0 -> self-jail LogitTilt at that beta (folder jail_b<beta>).
# Rounds are resume-aware: an existing round_N/judgment.json is reused, so calling
# with rounds=2 on a completed 1-round arm only generates round_2 (seed = base+N).
# var_batch 8: at 15 the turn-3 eager attention matrix needs ~29GiB and OOMs.
BETA="$1"
ROUNDS="${2:-1}"
[ -z "$BETA" ] && { echo "usage: run_arm.sh <beta> [rounds]"; exit 2; }
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
export BLOOM_NUM_SCENARIOS=15
export BLOOM_NUM_ROUNDS=$ROUNDS
export BLOOM_MAX_TURNS=3
export BLOOM_SEED=1
export BLOOM_JAIL_VAR_BATCH=8
export BLOOM_KICKOFF_BANK=experiments/bloom/_banks/runs_hyperparam/self_harm/Qwen_Qwen3.5-4B/_bank

if [ "$BETA" = "0" ]; then
  export BLOOM_FOLDER=runs_dsv4/self_harm/deepseek_v4_flash/vanilla_15s
else
  export BLOOM_FOLDER=runs_dsv4/self_harm/deepseek_v4_flash/jail_b${BETA}
  export BLOOM_JAIL_MODEL=local/deepseek-ai/DeepSeek-V4-Flash-0731
  export BLOOM_JAIL_BETA=$BETA
fi

echo "START beta=$BETA rounds=$ROUNDS $(date -u +%H:%M:%S)"
python src/bloom/bloom_corrupt.py
echo "EXIT=$? beta=$BETA $(date -u +%H:%M:%S)"
