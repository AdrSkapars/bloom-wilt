#!/bin/bash
# run_hftx.sh <stage> [args]    dispatch entry point for hf_transcripts.py.
#
# A wrapper rather than an inline `bash -c`: the dispatch command is assembled on a Windows
# client, and $PWD / $PATH inside a quoted -c string get expanded locally before the command
# ever reaches the server, producing a PATH full of C:/ paths and a job that dies instantly.
# Keeping the expansions inside a file on the server avoids the whole class of problem.
set -e
cd "$(dirname "$0")/../../.."

export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

[ -f .env.local ] && { set -a; . ./.env.local; set +a; }

echo "=== host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec python experiments/postpaper/oracle_check/hf_transcripts.py "$@"
