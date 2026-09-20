#!/bin/bash
# run_enum.sh [args]    dispatch entry point for enumerate_tree.py.
#
# A wrapper, not an inline `bash -c`: the dispatch command is built on a Windows client and
# $PWD / $PATH inside a quoted -c string expand LOCALLY before reaching the server. That has
# now cost two separate runs.
set -e
cd "$(dirname "$0")/../../.."
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "=== host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec python experiments/postpaper/branching/enumerate_tree.py "$@"
