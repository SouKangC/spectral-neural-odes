#!/usr/bin/env bash
# Submit the full Phase 5.1 synthetic 2×2 grid to the cluster.
#
# Cells × models × seeds = 12 × 3 × 3 = 108 jobs. Each job is one
# A100 for up to 1 h. Total compute: ~25–35 GPU-hours.
#
# Usage:
#   bash code/scripts/launch_phase5.sh                 # submit all
#   bash code/scripts/launch_phase5.sh --dry-run       # print only
#   bash code/scripts/launch_phase5.sh --seeds 0       # one seed
#   bash code/scripts/launch_phase5.sh --cells lotka_volterra__regular
#   bash code/scripts/launch_phase5.sh --models lsa_node ncde

set -euo pipefail

DRY=0
SEEDS=(0 1 2)
MODELS=(lsa_node node ncde)

# The 2×2 grid (synthetic). See progress.md §5.1.
PERIODIC_REGULAR=(periodic_3d_a__regular periodic_3d_b__regular
                  lotka_volterra__regular glycolytic__regular)
PERIODIC_IRREGULAR=(periodic_3d_a__irregular periodic_3d_b__irregular
                    lotka_volterra__irregular glycolytic__irregular)
APERIODIC_REGULAR=(forced_vibration__regular unstable_oscillator__regular)
APERIODIC_IRREGULAR=(lorenz63__irregular lorenz96__irregular)

CELLS=("${PERIODIC_REGULAR[@]}" "${PERIODIC_IRREGULAR[@]}"
       "${APERIODIC_REGULAR[@]}" "${APERIODIC_IRREGULAR[@]}")

# Allow filtering via flags.
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)        DRY=1; shift;;
    --seeds)          shift; SEEDS=(); while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do SEEDS+=("$1"); shift; done;;
    --models)         shift; MODELS=(); while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do MODELS+=("$1"); shift; done;;
    --cells)          shift; CELLS=();  while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do CELLS+=("$1");  shift; done;;
    *)                echo "unknown arg: $1" >&2; exit 2;;
  esac
done

PROJECT=/projects/bfaz/LSA-NODE
total=0
submitted=0
skipped=0

for cell in "${CELLS[@]}"; do
  cfg="$PROJECT/code/configs/${cell}.yaml"
  if [ ! -f "$cfg" ]; then
    echo "skip (no config): $cell" >&2
    continue
  fi
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      total=$((total + 1))
      job_name="P5-${cell}-${model}-s${seed}"
      out_path="$PROJECT/results/${cell}/${model}/seed${seed}/final_metrics.json"
      if [ -f "$out_path" ]; then
        echo "skip (already done): $job_name" >&2
        skipped=$((skipped + 1))
        continue
      fi
      cmd=(sbatch --job-name="$job_name"
           "$PROJECT/code/scripts/sbatch_train.sh"
           "$PROJECT/code/configs/${cell}.yaml"
           --seed "$seed" --model "$model")
      if [ "$DRY" = "1" ]; then
        echo "DRY: ${cmd[*]}"
      else
        ( cd "$PROJECT" && "${cmd[@]}" )
        submitted=$((submitted + 1))
      fi
    done
  done
done

echo "----"
echo "total candidates: $total"
echo "submitted:        $submitted"
echo "skipped (cached): $skipped"
