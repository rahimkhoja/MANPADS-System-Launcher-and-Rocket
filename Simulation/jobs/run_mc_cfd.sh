#!/bin/bash
#SBATCH --job-name=rocket-mc
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=../results/mc_cfd_%j.out
#SBATCH --error=../results/mc_cfd_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2024a

cd "${SLURM_SUBMIT_DIR}/../src"

RESULTS_DIR="${SLURM_SUBMIT_DIR}/../results"
GAINS_FILE="${1:-${RESULTS_DIR}/best_gains.json}"
AERO_TABLE="${2:-${RESULTS_DIR}/aero_table.json}"
NUM_RUNS="${3:-2000}"

echo "=========================================="
echo "Monte Carlo with CFD Aero Tables"
echo "Date:       $(date)"
echo "Gains:      ${GAINS_FILE}"
echo "Aero table: ${AERO_TABLE}"
echo "Runs:       ${NUM_RUNS}"
echo "Workers:    ${SLURM_CPUS_PER_TASK}"
echo "=========================================="

AERO_ARG=""
if [ -f "$AERO_TABLE" ]; then
    AERO_ARG="--aero-table ${AERO_TABLE}"
    echo "CFD aero table found, using it"
else
    echo "WARNING: No aero table at ${AERO_TABLE}, using Barrowman fallback"
fi

python -u monte_carlo.py \
    --runs "$NUM_RUNS" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --gains-file "$GAINS_FILE" \
    $AERO_ARG \
    --output-dir "$RESULTS_DIR"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
