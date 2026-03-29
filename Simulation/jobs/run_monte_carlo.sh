#!/bin/bash
#SBATCH --job-name=rocket-mc
#SBATCH --output=../results/mc_%j.out
#SBATCH --error=../results/mc_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Monte Carlo Trajectory Analysis"
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

GAINS_FILE="${RESULTS_DIR}/best_gains.json"
GAINS_ARG=""
if [ -f "${GAINS_FILE}" ]; then
    echo "Using optimized gains from: ${GAINS_FILE}"
    GAINS_ARG="--gains-file ${GAINS_FILE}"
fi

echo ""
python -u monte_carlo.py \
    --runs 2000 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --seed 42 \
    ${GAINS_ARG} \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
