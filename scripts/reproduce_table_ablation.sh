#!/usr/bin/env bash
# Reproduce ablations from paper §6 on a small subset of datasets.
set -euo pipefail

DATASETS=(periodic_3d_a forced_vibration ecg200 physionet2012)
ABLATIONS=(no_stft no_attention no_rope no_hoyer no_filter_k freq_state)

for dataset in "${DATASETS[@]}"; do
  for abl in "${ABLATIONS[@]}"; do
    cfg="configs/abl_${abl}_${dataset}.yaml"
    [ -f "$cfg" ] || { echo "skip: $cfg"; continue; }
    echo ">> $abl | $dataset"
    python -m lsa_node.train --config "$cfg" --seed 0
  done
done
