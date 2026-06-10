# shellcheck shell=bash
# Source this to activate the LSA-NODE conda env on the Delta cluster.
#   source code/scripts/activate_env.sh
# Idempotent: safe to source multiple times.

LSA_NODE_ENV="${LSA_NODE_ENV:-/work/nvme/bfaz/envs/lsa-node}"

if [ ! -d "$LSA_NODE_ENV" ]; then
  echo "lsa-node env not found at $LSA_NODE_ENV" >&2
  echo "Run: pip install -e . (after activating a base conda)" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source /u/jhe12/miniconda3/bin/activate "$LSA_NODE_ENV"

# Make the project importable without `pip install -e .`.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Be polite on shared login nodes — let SLURM tell us how many cores we
# have when running under srun/sbatch; default to a small cap otherwise.
_n_cpus="${SLURM_CPUS_PER_TASK:-${LSA_NODE_THREADS:-4}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_n_cpus}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$_n_cpus}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$_n_cpus}"
unset _n_cpus
