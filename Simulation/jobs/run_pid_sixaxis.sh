#!/bin/bash
#SBATCH --job-name=rocket-pid-6ax
#SBATCH --output=../results/pid_sixaxis_%j.out
#SBATCH --error=../results/pid_sixaxis_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "Six-axis PID (roll+pitch composite objective) — merges pitch into sim"
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

python -u pid_optimizer.py \
    --method de \
    --six-axis \
    --wind-speeds 0 2 5 \
    --num-runs 3 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --population 30 \
    --generations 40 \
    --output-dir "${RESULTS_DIR}"

echo "Saved best_gains.json — merge roll gains from roll-only job if needed."
