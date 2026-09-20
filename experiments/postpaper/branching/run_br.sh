#!/bin/bash
# run_br.sh <stage> [args]    dispatch entry point for branching.py.
set -e
cd "$(dirname "$0")/../../.."
export PATH="$PWD/.venv/bin:$PATH"
# the decode allocates and frees several [B, V] tensors per step at a 248k vocab;
# expandable segments stop that fragmenting the allocator
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
[ -f .env.local ] && { set -a; . ./.env.local; set +a; }
echo "=== host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec python experiments/postpaper/branching/branching.py "$@"
