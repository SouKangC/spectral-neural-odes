#!/bin/bash
#SBATCH --job-name=lsa-node
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfaz-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -o /projects/bfaz/LSA-NODE/results/slurm-%x-%j.out
#SBATCH -e /projects/bfaz/LSA-NODE/results/slurm-%x-%j.err

# Generic LSA-NODE training launcher.
#
# Usage:
#   sbatch code/scripts/sbatch_train.sh configs/smoke.yaml [extra args]
#
# Override SBATCH directives with command-line flags:
#   sbatch --time=12:00:00 --job-name=lorenz63 code/scripts/sbatch_train.sh \
#       configs/lorenz63__irregular.yaml --seed 0

set -euo pipefail

CONFIG="${1:?Usage: sbatch sbatch_train.sh <config.yaml> [extra args]}"
shift || true

PROJECT=/projects/bfaz/LSA-NODE
RESULTS=$PROJECT/results
mkdir -p "$RESULTS"

# shellcheck disable=SC1091
source "$PROJECT/code/scripts/activate_env.sh"

cd "$PROJECT/code"
echo "[$(date -Is)] host=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[$(date -Is)] config=$CONFIG extra='$*'"

# Force results to live under <project>/results/<exp_id>/... — without
# an explicit --out, train.py would write under code/results/ which we
# don't want to commit.
python -m lsa_node.train --config "$CONFIG" --out "$RESULTS" "$@"
