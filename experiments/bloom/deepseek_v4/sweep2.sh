#!/bin/bash
# ONE ROUND PER PROCESS. Rounds are resume-aware, so run_arm.sh <beta> N reuses
# rounds 1..N-1 and generates only round N. Costs a 36s model load per round, and
# avoids whatever accumulates in a long-lived process: rounds 3 and 4 took 11-12
# min each, round 5 in the same process took 47 min per chunk with the GPU idling
# at 25% / 118W (no throttling, healthy API, normal reply lengths).
LOG=/workspace/sweep2.log
one() {  # one <label> <beta> <round> <vb> <fallback_vb>
  local label=$1 beta=$2 rd=$3 vb=$4 fb=$5
  local t0=$SECONDS
  echo "### $label round=$rd vb=$vb START $(date -u +%H:%M:%S)" >> $LOG
  if ! /workspace/run_arm.sh "$beta" "$rd" "$vb" >> $LOG 2>&1; then
    echo "### $label round=$rd RETRY at vb=$fb $(date -u +%H:%M:%S)" >> $LOG
    /workspace/run_arm.sh "$beta" "$rd" "$fb" >> $LOG 2>&1 \
      || { echo "### $label round=$rd FAILED $(date -u +%H:%M:%S)" >> $LOG; return 1; }
  fi
  echo "### $label round=$rd DONE in $((SECONDS-t0))s $(date -u +%H:%M:%S)" >> $LOG
}
echo "##### SWEEP2 START $(date -u +%H:%M:%S)" >> $LOG
for r in 3 4 5; do one "b0.5" 0.5 "$r" 8 5; done
for r in 2 3 4 5; do one "b1"  1   "$r" 8 5; done
echo "##### 15-SCEN SWEEP DONE $(date -u +%H:%M:%S)" >> $LOG
export SCEN=100 SEED=100 TAG=100s
export BANK=experiments/bloom/_banks/runs_final/self_harm/_bank
for r in 1 2 3 4 5 6 7 8; do one "b0-100s" 0 "$r" 15 8 || break; done
echo "##### SWEEP2 DONE $(date -u +%H:%M:%S)" >> $LOG
