#!/bin/bash
#SBATCH --job-name=rocket-full-analysis
#SBATCH --output=../results/full_%j.out
#SBATCH --error=../results/full_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Complete Rocket Analysis Pipeline
# Runs: PID Optimization → Monte Carlo → RL Training
# Submit with: sbatch run_full_analysis.sh

echo "=========================================="
echo "COMPLETE ROCKET ANALYSIS PIPELINE"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "=========================================="

module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2024a
module load cuda/12.2

VENV_DIR="${SLURM_TMPDIR}/venv"
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --no-index torch 2>/dev/null || pip install torch

cd "${SLURM_SUBMIT_DIR}/../src"

RESULTS_DIR="${SLURM_SUBMIT_DIR}/../results/full_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}"

# ==================================================
# PHASE 1: Test Baseline Simulation
# ==================================================
echo ""
echo "PHASE 1: Testing Baseline Simulation"
echo "--------------------------------------"

python -c "
from rocket_dynamics import RocketSimulator
sim = RocketSimulator()
results = sim.run(duration=8.0, dt=0.005)
print(f'Baseline Results:')
print(f'  Max Altitude: {results[\"max_altitude\"]:.1f} m')
print(f'  Roll RMS: {results[\"roll_rms\"]:.2f}°')
print(f'  Stability Score: {results[\"stability_score\"]:.1f}')
"

# ==================================================
# PHASE 2: Kalman Filter Validation
# ==================================================
echo ""
echo "PHASE 2: Testing Kalman Filter"
echo "--------------------------------------"

python kalman_filter.py
mv attitude_ekf.h "${RESULTS_DIR}/" 2>/dev/null || true

# ==================================================
# PHASE 3: PID Optimization
# ==================================================
echo ""
echo "PHASE 3: PID Gain Optimization (Differential Evolution)"
echo "--------------------------------------"

python pid_optimizer.py \
    --method de \
    --wind-speeds 0 2 5 8 \
    --num-runs 3 \
    --workers ${SLURM_CPUS_PER_TASK} \
    --population 30 \
    --generations 75 \
    --output-dir "${RESULTS_DIR}"

# ==================================================
# PHASE 4: Monte Carlo Analysis
# ==================================================
echo ""
echo "PHASE 4: Monte Carlo Trajectory Analysis"
echo "--------------------------------------"

GAINS_FILE="${RESULTS_DIR}/best_gains.json"
GAINS_ARG=""
if [ -f "${GAINS_FILE}" ]; then
    GAINS_ARG="--gains-file ${GAINS_FILE}"
    echo "Using optimized gains"
fi

python monte_carlo.py \
    --runs 5000 \
    --workers ${SLURM_CPUS_PER_TASK} \
    --seed 42 \
    ${GAINS_ARG} \
    --output-dir "${RESULTS_DIR}"

# ==================================================
# PHASE 5: Reinforcement Learning (if GPU available)
# ==================================================
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo ""
    echo "PHASE 5: Reinforcement Learning Training"
    echo "--------------------------------------"
    
    python rl_controller.py \
        --timesteps 500000 \
        --eval-episodes 100 \
        --export-cpp \
        --output-dir "${RESULTS_DIR}"
else
    echo ""
    echo "PHASE 5: Skipped (No GPU)"
fi

# ==================================================
# PHASE 6: Generate Summary Report
# ==================================================
echo ""
echo "PHASE 6: Generating Summary Report"
echo "--------------------------------------"

python << 'EOF'
import json
from pathlib import Path
import sys

results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../results")

report = []
report.append("=" * 60)
report.append("ROCKET SIMULATION ANALYSIS REPORT")
report.append("=" * 60)

# Load best gains
gains_file = results_dir / "best_gains.json"
if gains_file.exists():
    with open(gains_file) as f:
        data = json.load(f)
    report.append("\nOPTIMIZED PID GAINS:")
    for k, v in data.get('gains', {}).items():
        report.append(f"  {k}: {v:.4f}")
    report.append(f"  Optimization Score: {data.get('score', 'N/A')}")

# Load Monte Carlo results
mc_file = results_dir / "monte_carlo_results.json"
if mc_file.exists():
    with open(mc_file) as f:
        data = json.load(f)
    summary = data.get('summary', {})
    report.append("\nMONTE CARLO RESULTS:")
    report.append(f"  Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
    if 'max_altitude' in summary:
        report.append(f"  Max Altitude: {summary['max_altitude'].get('mean', 0):.1f} ± {summary['max_altitude'].get('std', 0):.1f} m")
    if 'roll_rms' in summary:
        report.append(f"  Roll RMS: {summary['roll_rms'].get('mean', 0):.2f}° (target < 5°)")

# Load RL results
rl_file = results_dir / "rl_evaluation.json"
if rl_file.exists():
    with open(rl_file) as f:
        data = json.load(f)
    report.append("\nRL CONTROLLER RESULTS:")
    report.append(f"  Mean Reward: {data.get('mean_reward', 0):.2f}")
    report.append(f"  Success Rate: {data.get('success_rate', 0)*100:.1f}%")
    report.append(f"  Roll RMS: {data.get('mean_roll_rms', 0):.2f}°")

report.append("\n" + "=" * 60)

print("\n".join(report))

# Save report
report_file = results_dir / "analysis_report.txt"
with open(report_file, 'w') as f:
    f.write("\n".join(report))
print(f"\nReport saved to: {report_file}")
EOF

echo ""
echo "=========================================="
echo "ANALYSIS PIPELINE COMPLETE"
echo "Date: $(date)"
echo "Results in: ${RESULTS_DIR}"
echo "=========================================="
