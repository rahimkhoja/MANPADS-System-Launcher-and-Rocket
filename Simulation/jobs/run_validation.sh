#!/bin/bash
#SBATCH --job-name=rocket-validate
#SBATCH --output=../results/validate_%j.out
#SBATCH --error=../results/validate_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Quick Validation (< 2 min)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2024a

VENV_DIR="${SLURM_TMPDIR}/venv"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

cd "${SLURM_SUBMIT_DIR}/../src"

echo ""
echo "--- 1. Dynamics smoke test ---"
python -u rocket_dynamics.py

echo ""
echo "--- 2. Tiny grid search (9 combos) ---"
python -u pid_optimizer.py \
    --method grid \
    --grid-points 3 \
    --num-runs 1 \
    --wind-speeds 0 3 \
    --output-dir /tmp/rocket_val

echo ""
echo "--- 3. Mini Monte Carlo (20 runs) ---"
python -u monte_carlo.py \
    --runs 20 \
    --seed 42 \
    --output-dir /tmp/rocket_val

echo ""
echo "--- 4. Kalman filter generation ---"
python -u kalman_filter.py

echo ""
echo "=========================================="
echo "VALIDATION PASSED: $(date)"
echo "=========================================="
