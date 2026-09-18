#!/bin/bash
# run_hpc.sh [k]    dispatch entry point for run_local.sh on the Manchester CS box (s3).
#
# dispatch runs jobs in a CLEAN environment, so nothing exported into the submitting shell
# survives. Everything the run needs is either set here or passed on the dispatch command
# line with `env VAR=...`:
#
#   dispatch new --cpu 4 --gpu-cores 1 --max-mem 32 --numa-local --max-time 1h \
#       -- env BEH=self_harm AUDITOR=api bash ~/bloom-wilt/experiments/postpaper/partial_tilt/run_hpc.sh 0
#
# The clean environment is also why this cannot just be run_local.sh: that script calls
# `python`, which does not exist outside the venv -- the system only ships python3.
set -e
cd "$(dirname "$0")/../../.."

export PATH="$PWD/.venv/bin:$PATH"

# Same fragmentation fix the Modal image carried. The decode allocates and frees several
# [B, V] tensors per token and the topk workspace grows with k; on a 22GB A10 that ended in
# two OOMs with ~2GB "reserved but unallocated". A6000s have 48GB so it is no longer load
# bearing, but it costs nothing and keeps the two boxes comparable.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

echo "=== host=$(hostname) job=${HPC_BATCH_JOB_ID:-?}"
echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch;print('=== torch',torch.__version__,'devices',torch.cuda.device_count())"

exec bash experiments/postpaper/partial_tilt/run_local.sh "$@"
