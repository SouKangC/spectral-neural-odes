#!/bin/bash
#SBATCH --job-name=lsa-fast
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfaz-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o /projects/bfaz/LSA-NODE/results/slurm-%x-%j.out
#SBATCH -e /projects/bfaz/LSA-NODE/results/slurm-%x-%j.err

# Fast-iteration launcher: 30-minute cap, otherwise identical to
# sbatch_train.sh. Use with configs/fast_*.yaml.

set -euo pipefail
CONFIG="${1:?Usage: sbatch sbatch_train_fast.sh <config.yaml> [extra args]}"
shift || true

PROJECT=/projects/bfaz/LSA-NODE
RESULTS=$PROJECT/results
mkdir -p "$RESULTS"
# shellcheck disable=SC1091
source "$PROJECT/code/scripts/activate_env.sh"
cd "$PROJECT/code"
echo "[$(date -Is)] host=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[$(date -Is)] config=$CONFIG extra='$*'"
python -m lsa_node.train --config "$CONFIG" --out "$RESULTS" "$@"
