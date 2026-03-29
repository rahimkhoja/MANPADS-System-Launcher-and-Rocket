#!/bin/bash
#SBATCH --job-name=rocket-full
#SBATCH --output=../results/full_%j.out
#SBATCH --error=../results/full_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "COMPLETE ROCKET ANALYSIS PIPELINE"
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

# Phase 1: Baseline smoke test
echo ""
echo "--- Phase 1: Baseline ---"
python -u rocket_dynamics.py

# Phase 2: Kalman filter
echo ""
echo "--- Phase 2: Kalman Filter ---"
python -u kalman_filter.py

# Phase 3: PID optimisation
echo ""
echo "--- Phase 3: PID Optimizer ---"
python -u pid_optimizer.py \
    --method de \
    --wind-speeds 0 2 5 \
    --num-runs 2 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --population 40 \
    --generations 60 \
    --output-dir "${RESULTS_DIR}"

# Phase 4: Monte Carlo
echo ""
echo "--- Phase 4: Monte Carlo ---"
GAINS_FILE="${RESULTS_DIR}/best_gains.json"
GAINS_ARG=""
[ -f "${GAINS_FILE}" ] && GAINS_ARG="--gains-file ${GAINS_FILE}"

python -u monte_carlo.py \
    --runs 2000 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --seed 42 \
    ${GAINS_ARG} \
    --output-dir "${RESULTS_DIR}"

# Phase 5: Summary
echo ""
echo "--- Results Summary ---"
python -u -c "
import json, pathlib, sys
d = pathlib.Path('${RESULTS_DIR}')
for f in sorted(d.glob('*.json')):
    print(f'  {f.name}')
bg = d / 'best_gains.json'
if bg.exists():
    data = json.load(open(bg))
    g = data.get('gains', {})
    print(f'Best gains: Kp_r={g.get(\"Kp_roll\",\"?\")}, Ki_r={g.get(\"Ki_roll\",\"?\")}, Kd_r={g.get(\"Kd_roll\",\"?\")}  '
          f'Kp_p={g.get(\"Kp_pitch\",\"?\")}, Ki_p={g.get(\"Ki_pitch\",\"?\")}, Kd_p={g.get(\"Kd_pitch\",\"?\")}')
    print(f'Score: {data.get(\"score\",\"?\")}')
mc = d / 'monte_carlo_results.json'
if mc.exists():
    data = json.load(open(mc))
    s = data.get('summary', {})
    print(f'MC success rate: {s.get(\"success_rate\",0)*100:.1f}%')
"

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE: $(date)"
echo "=========================================="
