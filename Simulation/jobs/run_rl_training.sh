#!/bin/bash
#SBATCH --job-name=rocket-rl
#SBATCH --output=../results/rl_%j.out
#SBATCH --error=../results/rl_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "RL Controller Training"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job:  ${SLURM_JOB_ID}"
echo "GPU:  ${CUDA_VISIBLE_DEVICES:-none}"
echo "=========================================="

module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2024a
module load cuda/12.2

VENV_DIR="${SLURM_TMPDIR}/venv"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --no-index torch 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cu121

cd "${SLURM_SUBMIT_DIR}/../src"

RESULTS_DIR="${SLURM_SUBMIT_DIR}/../results"
mkdir -p "${RESULTS_DIR}"

python -u -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
python -u rl_controller.py \
    --timesteps 500000 \
    --eval-episodes 100 \
    --export-cpp \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
