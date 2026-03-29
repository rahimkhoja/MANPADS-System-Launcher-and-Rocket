#!/bin/bash
#SBATCH --job-name=rocket-cfd
# Note: set --account if required on your cluster
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=32G
#SBATCH --output=results/cfd_%j.out
#SBATCH --error=results/cfd_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# Lmod / OpenFOAM bashrc require bash (not /bin/sh)
if [ -f /etc/profile.d/lmod.sh ]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/lmod.sh
fi

module purge
module load StdEnv/2023
module load openfoam/v2412
module load scipy-stack/2024a

REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"
cd "$SLURM_SUBMIT_DIR/.."

STL="${1:-${REPO_ROOT}/cad/rocket_assembly.stl}"
RESULTS_DIR="results/cfd_baseline_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "=== Rocket CFD Sweep ==="
echo "STL:   $STL"
echo "Procs: $SLURM_NTASKS"
echo "Start: $(date)"

for VELOCITY in 20 40 60; do
    for AOA in 0 2 5 10; do
        CASE_DIR="${RESULTS_DIR}/v${VELOCITY}_a${AOA}"
        echo "--- V=${VELOCITY} m/s, AoA=${AOA} deg ---"
        python -u cfd/run_case.py \
            --stl "$STL" \
            --velocity "$VELOCITY" \
            --aoa "$AOA" \
            --output "$CASE_DIR" \
            --nprocs "$SLURM_NTASKS" \
            --rocket-length 0.669 \
            --body-radius 0.0375 \
            --serial-mesh
    done
done

echo ""
echo "=== Collecting Results ==="
python3 -u -c "
import json, glob, os
results = []
for f in sorted(glob.glob('${RESULTS_DIR}/*/cfd_results.json')):
    with open(f) as fh:
        r = json.load(fh)
        r['case'] = os.path.basename(os.path.dirname(f))
        results.append(r)
        print(f'{r[\"case\"]:15s}  Cd={r.get(\"Cd\",0):.4f}  Cl={r.get(\"Cl\",0):.4f}  Cm={r.get(\"Cm\",0):.4f}')

with open('${RESULTS_DIR}/all_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
print(f'Saved {len(results)} results to ${RESULTS_DIR}/all_results.json')
"

echo "Done: $(date)"
