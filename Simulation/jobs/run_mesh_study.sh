#!/bin/bash
#SBATCH --job-name=mesh-study
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=96G
#SBATCH --output=results/mesh_study_%j.out
#SBATCH --error=results/mesh_study_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMPI_MCA_mpi_cuda_support=0

if [ -f /etc/profile.d/lmod.sh ]; then
    source /etc/profile.d/lmod.sh
fi

module purge
module load StdEnv/2023
module load openfoam/v2412
module load scipy-stack/2024a

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

STL_CANDIDATE="${1:-../../cad/rocket_baseline_clean.stl}"
VELOCITY="${2:-40}"
AOA="${3:-0}"

resolve_stl_path() {
    local candidate="${1}"
    for p in "$candidate" "$WORK_DIR/$candidate" "$WORK_DIR/../$candidate" "$WORK_DIR/../../$candidate"; do
        if [ -f "$p" ]; then echo "$p"; return; fi
    done
}

STL="$(resolve_stl_path "$STL_CANDIDATE")"
if [ -z "$STL" ] || [ ! -f "$STL" ]; then
    echo "ERROR: Cannot find STL: $STL_CANDIDATE" >&2
    exit 1
fi
STL="$(cd "$(dirname "$STL")" && echo "$(pwd)/$(basename "$STL")")"

REPO_ROOT=""
if [[ "$STL" == */cad/* ]]; then
    REPO_ROOT="${STL%/cad/*}"
elif [[ "$STL" == */Simulation/* ]]; then
    REPO_ROOT="${STL%/Simulation/*}"
elif [ -d "$WORK_DIR/../Simulation" ]; then
    REPO_ROOT="$(cd "$WORK_DIR/.." && pwd)"
else
    REPO_ROOT="$(cd "$WORK_DIR" && pwd)"
fi

cd "$REPO_ROOT/Simulation"
RESULTS_DIR="$REPO_ROOT/Simulation/results/mesh_study_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "=== Mesh Independence Study ==="
echo "STL:      $STL"
echo "V=${VELOCITY} m/s, AoA=${AOA} deg"
echo "Procs:    $SLURM_NTASKS"
echo "Start:    $(date)"
echo ""

for LEVEL in 1 2 3; do
    CASE_DIR="${RESULTS_DIR}/level_${LEVEL}"
    echo "--- Refinement Level ${LEVEL} ---"
    python -u cfd/run_case.py \
        --stl "$STL" \
        --velocity "$VELOCITY" \
        --aoa "$AOA" \
        --output "$CASE_DIR" \
        --nprocs "$SLURM_NTASKS" \
        --rocket-length 0.669 \
        --body-radius 0.0375 \
        --refinement-level "$LEVEL" \
        --serial-mesh
    echo ""
done

echo "=== Mesh Study Results ==="
python3 -u -c "
import json, os

results = []
for level in [1, 2, 3]:
    f = '${RESULTS_DIR}/level_{}/cfd_results.json'.format(level)
    if os.path.exists(f):
        with open(f) as fh:
            r = json.load(fh)
            r['level'] = level
            results.append(r)
            print(f'Level {level}: Cd={r.get(\"Cd\",0):.6f}  Cl={r.get(\"Cl\",0):.6f}  CmPitch={r.get(\"CmPitch\",0):.6f}')
    else:
        print(f'Level {level}: FAILED (no results)')

with open('${RESULTS_DIR}/mesh_study_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)

if len(results) >= 3:
    cd = [r['Cd'] for r in results]
    h = [1, 0.5, 0.25]
    p = abs((cd[1]-cd[0])/(cd[2]-cd[1])) if abs(cd[2]-cd[1]) > 1e-10 else float('inf')
    import math
    order = math.log2(p) if p > 0 and p != float('inf') else 0
    extrap = cd[2] + (cd[2]-cd[1])/(2**order - 1) if order > 0 else cd[2]
    print(f'')
    print(f'Richardson extrapolation:')
    print(f'  Apparent order: {order:.2f}')
    print(f'  Grid-converged Cd: {extrap:.6f}')
"

echo ""
echo "Done: $(date)"
