#!/bin/bash
# run_traj.sh <stage> [args]    dispatch entry point for trajectory.py. See run_hftx.sh for
# why this is a file rather than an inline `bash -c`.
set -e
cd "$(dirname "$0")/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
[ -f .env.local ] && { set -a; . ./.env.local; set +a; }
echo "=== host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec python experiments/postpaper/oracle_check/trajectory.py "$@"
