#!/bin/bash
# run_oracle.sh <stage> [args]    dispatch entry point for oracle_check.py.
#
# Same two reasons run_hpc.sh exists: dispatch runs jobs in a CLEAN environment, so the
# venv is not on PATH and nothing from .env.local is exported. The judge stage needs
# OPENROUTER_API_KEY, and sourcing the file here keeps the key out of the dispatch command
# line (and therefore out of `dispatch list` and any shell history).
#
#   dispatch new --cpu 4 --gpu-cores 1 --max-mem 28 --numa-local --max-time 2h \
#       -- bash ~/bloom-wilt/experiments/postpaper/oracle_check/run_oracle.sh \
#          gen --model qwen --beh racial
set -e
cd "$(dirname "$0")/../../.."

export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
# HF_HOME also tells huggingface_hub where the stored token lives, which is what gets the
# gated Llama-3.2 repo. dispatch does not pass it through, so set it unless already given.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

[ -f .env.local ] || { echo "missing .env.local"; exit 1; }
set -a; . ./.env.local; set +a

echo "=== host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec python experiments/postpaper/oracle_check/oracle_check.py "$@"
