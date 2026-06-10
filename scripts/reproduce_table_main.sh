#!/usr/bin/env bash
# Reproduce the main table from paper §5.
# Each row is one (model, dataset, sampling, seed) cell; results land in
#   results/<dataset>__<sampling>/<model>/seed<S>/metrics.json
set -euo pipefail

MODELS=(lsa_node node ncde)
SEEDS=(0 1 2)

# ---------------------------------------------------------------------------
# §5.1 Synthetic 2x2 grid (periodicity × sampling regularity)
# ---------------------------------------------------------------------------
# Periodic generators — run under both regular and irregular sampling.
PERIODIC_GENERATORS=(periodic_3d_a periodic_3d_b lotka_volterra glycolytic)
# Aperiodic generators — split because the natural pairing differs:
#   regular: transient deterministic systems
#   irregular: chaotic systems (DIFFODE Lorenz protocol)
APERIODIC_REGULAR=(forced_vibration unstable_oscillator)
APERIODIC_IRREGULAR=(lorenz63 lorenz96)

# ---------------------------------------------------------------------------
# §5.2 Real-world datasets
# ---------------------------------------------------------------------------
REAL_REGULAR=(spanish_load building_load building_temp spanish_temp)
REAL_IRREGULAR=(ushcn physionet2012)

run() {
  local model="$1" dataset="$2" sampling="$3" seed="$4"
  local cfg="configs/${dataset}__${sampling}.yaml"
  [ -f "$cfg" ] || { echo "skip: $cfg"; return; }
  echo ">> $model | $dataset | $sampling | seed=$seed"
  python -m lsa_node.train --config "$cfg" --model "$model" --seed "$seed"
}

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # periodic × {regular, irregular}
    for d in "${PERIODIC_GENERATORS[@]}"; do
      run "$model" "$d" regular "$seed"
      run "$model" "$d" irregular "$seed"
    done
    # aperiodic
    for d in "${APERIODIC_REGULAR[@]}";   do run "$model" "$d" regular   "$seed"; done
    for d in "${APERIODIC_IRREGULAR[@]}"; do run "$model" "$d" irregular "$seed"; done
    # real-world
    for d in "${REAL_REGULAR[@]}";   do run "$model" "$d" regular   "$seed"; done
    for d in "${REAL_IRREGULAR[@]}"; do run "$model" "$d" irregular "$seed"; done
  done
done
