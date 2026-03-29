#!/bin/bash
#SBATCH --job-name=design-sweep
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=results/sweep_%j.out
#SBATCH --error=results/sweep_%j.err

module load StdEnv/2023
module load scipy-stack/2024a

export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR/.."

echo "=== Aerodynamic Design Sweep ==="
echo "Start: $(date)"

python -u src/design_sweep.py \
    --output "results/sweep_results_${SLURM_JOB_ID}.json" \
    --generate-stl "results/candidate_stls_${SLURM_JOB_ID}/" \
    --top 5

echo "Done: $(date)"
