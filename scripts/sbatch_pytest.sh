#!/bin/bash
#SBATCH --job-name=lsa-pytest
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfaz-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH -o /projects/bfaz/LSA-NODE/results/slurm-%x-%j.out
#SBATCH -e /projects/bfaz/LSA-NODE/results/slurm-%x-%j.err

# Run the test suite on a real GPU.
#
# Usage:
#   sbatch code/scripts/sbatch_pytest.sh                # all (fast + slow) tests
#   sbatch code/scripts/sbatch_pytest.sh -m slow        # only slow
#   sbatch code/scripts/sbatch_pytest.sh tests/test_lsa_node.py -v
#
# Anything passed after the script name is forwarded to pytest. The default
# (no args) runs the *full* suite including @pytest.mark.slow.

set -euo pipefail

PROJECT=/projects/bfaz/LSA-NODE
mkdir -p "$PROJECT/results"

# shellcheck disable=SC1091
source "$PROJECT/code/scripts/activate_env.sh"

cd "$PROJECT/code"
echo "[$(date -Is)] host=$(hostname) gpu=$(nvidia-smi -L | head -1)"

# By default include slow tests (they only make sense on a GPU).
if [ "$#" -eq 0 ]; then
  set -- -m ""
fi

pytest tests/ "$@"
