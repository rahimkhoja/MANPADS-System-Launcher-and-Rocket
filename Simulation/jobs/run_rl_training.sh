#!/bin/bash
#SBATCH --job-name=rocket-rl
#SBATCH --output=../results/rl_%j.out
#SBATCH --error=../results/rl_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Reinforcement Learning Controller Training
# Run with: sbatch run_rl_training.sh

echo "=========================================="
echo "RL Training Job Started"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="

# Load modules
module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2024a
module load cuda/12.2

# Setup environment
VENV_DIR="${SLURM_TMPDIR}/venv"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Install PyTorch with CUDA support
pip install --no-index torch torchvision 2>/dev/null || \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

cd "${SLURM_SUBMIT_DIR}/../src"

RESULTS_DIR="${SLURM_SUBMIT_DIR}/../results/rl_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "Starting RL Training..."
echo ""

python rl_controller.py \
    --timesteps 1000000 \
    --eval-episodes 200 \
    --export-cpp \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "=========================================="
echo "Job Completed: $(date)"
echo "Results in: ${RESULTS_DIR}"
echo "=========================================="
