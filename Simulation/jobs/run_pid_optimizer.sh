#!/bin/bash
#SBATCH --job-name=rocket-pid-opt
#SBATCH --output=../results/pid_opt_%j.out
#SBATCH --error=../results/pid_opt_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# PID Gain Optimizer for Canard-Stabilized Rocket
# Run with: sbatch run_pid_optimizer.sh

echo "=========================================="
echo "PID Optimizer Job Started"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="

# Load required modules (ComputeCanada CVMFS)
module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2024a

# Create and activate virtual environment
VENV_DIR="${SLURM_TMPDIR}/venv"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Install additional dependencies
pip install --no-index --upgrade pip
pip install cupy-cuda12x 2>/dev/null || echo "CuPy not available, using NumPy"

# Ensure unbuffered Python output
export PYTHONUNBUFFERED=1

# Navigate to source directory
cd "${SLURM_SUBMIT_DIR}/../src"

# Create results directory
RESULTS_DIR="${SLURM_SUBMIT_DIR}/../results/job_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

echo ""
echo "Running Differential Evolution Optimization..."
echo ""

python -u pid_optimizer.py \
    --method de \
    --wind-speeds 0 2 5 8 \
    --num-runs 5 \
    --workers ${SLURM_CPUS_PER_TASK} \
    --population 40 \
    --generations 100 \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "Running Grid Search for Verification..."
echo ""

python pid_optimizer.py \
    --method grid \
    --wind-speeds 0 3 6 \
    --num-runs 3 \
    --workers ${SLURM_CPUS_PER_TASK} \
    --grid-points 6 \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "=========================================="
echo "Job Completed: $(date)"
echo "Results in: ${RESULTS_DIR}"
echo "=========================================="
