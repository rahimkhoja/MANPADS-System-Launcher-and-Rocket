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
export OMPI_MCA_mpi_cuda_support=0

# Lmod / OpenFOAM bashrc require bash (not /bin/sh)
if [ -f /etc/profile.d/lmod.sh ]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/lmod.sh
fi

module purge
module load StdEnv/2023
module load openfoam/v2412
module load scipy-stack/2024a

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

resolve_stl_path() {
    local candidate="${1}"
    if [ -f "$candidate" ]; then
        echo "$candidate"
        return
    fi

    local search_paths=(
        "$candidate"
        "$WORK_DIR/$candidate"
        "$WORK_DIR/../$candidate"
        "$WORK_DIR/../../$candidate"
        "$candidate"
    )

    for p in "${search_paths[@]}"; do
        if [ -f "$p" ]; then
            echo "$p"
            return
        fi
    done
}

STL_CANDIDATE="${1:-cad/rocket_assembly.stl}"
STL="$(resolve_stl_path "$STL_CANDIDATE")"

if [ -z "$STL" ] || [ ! -f "$STL" ]; then
    echo "ERROR: Cannot find STL: $STL_CANDIDATE" >&2
    exit 1
fi

STL="$(cd "$(dirname "$STL")" && echo "$(pwd)/$(basename "$STL")")"

# Infer repository root from where the STL lives, then fallback to submit directory layout.
REPO_ROOT=""
if [[ "$STL" == */cad/* ]]; then
    REPO_ROOT="${STL%/cad/*}"
elif [[ "$STL" == */Simulation/* ]]; then
    REPO_ROOT="${STL%/Simulation/*}"
elif [ -d "$WORK_DIR/../Simulation" ]; then
    REPO_ROOT="$(cd "$WORK_DIR/.." && pwd)"
elif [ -d "$WORK_DIR/../../Simulation" ]; then
    REPO_ROOT="$(cd "$WORK_DIR/../.." && pwd)"
else
    REPO_ROOT="$(cd "$WORK_DIR" && pwd)"
fi

cd "$REPO_ROOT/Simulation"
RESULTS_DIR="$REPO_ROOT/Simulation/results/cfd_baseline_${SLURM_JOB_ID}"
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
