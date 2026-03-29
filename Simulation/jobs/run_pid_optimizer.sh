#!/bin/bash
#SBATCH --job-name=rocket-pid-opt
#SBATCH --output=../results/pid_opt_%j.out
#SBATCH --error=../results/pid_opt_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "PID Roll-Gain Optimizer (Improved Design)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job:  ${SLURM_JOB_ID}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "=========================================="

module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2024a

VENV_DIR="${SLURM_TMPDIR}/venv"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

cd "${SLURM_SUBMIT_DIR}/../src"

RESULTS_DIR="${SLURM_SUBMIT_DIR}/../results"
mkdir -p "${RESULTS_DIR}"

echo ""
echo "--- Grid Search (coarse) ---"
python -u pid_optimizer.py \
    --method grid \
    --wind-speeds 0 2 5 \
    --num-runs 2 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "--- Differential Evolution (roll-only; use run_pid_sixaxis.sh for pitch) ---"
python -u pid_optimizer.py \
    --method de \
    --wind-speeds 0 2 5 \
    --num-runs 3 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --population 30 \
    --generations 40 \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
