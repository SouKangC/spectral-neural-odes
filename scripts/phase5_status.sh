#!/usr/bin/env bash
# Phase 5 progress dashboard. Counts how many of the 12 × 3 × 3 = 108
# jobs are complete by checking for final_metrics.json. Also shows
# queue state.

set -euo pipefail
PROJECT=/projects/bfaz/LSA-NODE

CELLS=(periodic_3d_a__regular periodic_3d_b__regular
       lotka_volterra__regular glycolytic__regular
       periodic_3d_a__irregular periodic_3d_b__irregular
       lotka_volterra__irregular glycolytic__irregular
       forced_vibration__regular unstable_oscillator__regular
       lorenz63__irregular lorenz96__irregular)
MODELS=(lsa_node node ncde)
SEEDS=(0 1 2)

done_count=0
total=$((${#CELLS[@]} * ${#MODELS[@]} * ${#SEEDS[@]}))

printf "%-32s | %-9s | %-9s | %-9s\n" "cell" "lsa_node" "node" "ncde"
echo   "---------------------------------+-----------+-----------+-----------"
for cell in "${CELLS[@]}"; do
  row="$cell"
  cells_done="$cell"
  printf "%-32s" "$cell"
  for model in "${MODELS[@]}"; do
    cell_str=""
    for seed in "${SEEDS[@]}"; do
      p="$PROJECT/results/${cell}/${model}/seed${seed}/final_metrics.json"
      if [ -f "$p" ]; then
        cell_str+="✓"
        done_count=$((done_count + 1))
      else
        cell_str+="·"
      fi
    done
    printf " | %-9s" "$cell_str"
  done
  echo
done
echo "---------------------------------+-----------+-----------+-----------"
echo "done: $done_count / $total"
echo
echo "queued/running:"
squeue -h -u jhe12 -o "%i %j %T %M" 2>/dev/null | grep -E "^[0-9]+ (P5|E[0-9])" || echo "  (none)"
