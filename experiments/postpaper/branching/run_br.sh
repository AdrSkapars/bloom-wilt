#!/bin/bash
# run_br.sh <stage> [args]    dispatch entry point for branching.py.
set -e
cd "$(dirname "$0")/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
[ -f .env.local ] && { set -a; . ./.env.local; set +a; }
echo "=== host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec python experiments/postpaper/branching/branching.py "$@"
