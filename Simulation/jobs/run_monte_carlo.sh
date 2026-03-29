#!/bin/bash
#SBATCH --job-name=rocket-monte-carlo
#SBATCH --output=../results/mc_%j.out
#SBATCH --error=../results/mc_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# Monte Carlo Trajectory Analysis
# Run with: sbatch run_monte_carlo.sh [--sensitivity]

echo "=========================================="
echo "Monte Carlo Analysis Job Started"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "=========================================="

# Load required modules
module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2024a

# Create virtual environment
VENV_DIR="${SLURM_TMPDIR}/venv"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

cd "${SLURM_SUBMIT_DIR}/../src"

RESULTS_DIR="${SLURM_SUBMIT_DIR}/../results/mc_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Check for optimized gains
GAINS_FILE="${SLURM_SUBMIT_DIR}/../results/best_gains.json"
GAINS_ARG=""
if [ -f "${GAINS_FILE}" ]; then
    echo "Using optimized gains from: ${GAINS_FILE}"
    GAINS_ARG="--gains-file ${GAINS_FILE}"
fi

echo ""
echo "Running Monte Carlo Analysis (10,000 runs)..."
echo ""

python monte_carlo.py \
    --runs 10000 \
    --workers ${SLURM_CPUS_PER_TASK} \
    --seed 42 \
    ${GAINS_ARG} \
    --output-dir "${RESULTS_DIR}"

# Run sensitivity analysis if requested
if [[ "$1" == "--sensitivity" ]]; then
    echo ""
    echo "Running Sensitivity Analysis..."
    echo ""
    
    python monte_carlo.py \
        --runs 2000 \
        --workers ${SLURM_CPUS_PER_TASK} \
        ${GAINS_ARG} \
        --sensitivity \
        --output-dir "${RESULTS_DIR}"
fi

echo ""
echo "=========================================="
echo "Job Completed: $(date)"
echo "Results in: ${RESULTS_DIR}"
echo "=========================================="
