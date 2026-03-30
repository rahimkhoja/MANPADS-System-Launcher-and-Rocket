#!/bin/bash
#SBATCH --job-name=canard-sweep
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=96G
#SBATCH --output=results/canard_sweep_%j.out
#SBATCH --error=results/canard_sweep_%j.err

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

REPO_ROOT=""
if [ -d "$WORK_DIR/../Simulation" ]; then
    REPO_ROOT="$(cd "$WORK_DIR/.." && pwd)"
elif [ -d "$WORK_DIR/../../Simulation" ]; then
    REPO_ROOT="$(cd "$WORK_DIR/../.." && pwd)"
else
    REPO_ROOT="$(cd "$WORK_DIR" && pwd)"
fi

cd "$REPO_ROOT/Simulation"
RESULTS_DIR="$REPO_ROOT/Simulation/results/canard_sweep_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

echo "=== Canard Deflection CFD Sweep ==="
echo "Start: $(date)"
echo ""

VELOCITY=40
AOA=0

for DEFL in 0 5 10 15; do
    STL_PATH="${RESULTS_DIR}/rocket_cd${DEFL}.stl"
    CASE_DIR="${RESULTS_DIR}/defl_${DEFL}"

    echo "--- Canard deflection: ${DEFL} deg ---"
    echo "Generating STL..."
    python -u -c "
import sys; sys.path.insert(0, 'src')
from rocket_geometry import generate_rocket
mesh = generate_rocket(
    nose_shape='vonkarman', nose_length=0.120, body_length=0.549,
    body_radius=0.0375, boat_tail=True,
    fin_count=4, fin_root=0.070, fin_tip=0.035, fin_span=0.060,
    fin_sweep=30.0, fin_thickness=0.002, fin_position=0.02,
    canard_count=4, canard_root=0.045, canard_tip=0.0225, canard_span=0.020,
    canard_sweep=0.0, canard_thickness=0.001,
    canard_deflection_deg=${DEFL}.0,
)
mesh.export('${STL_PATH}')
print(f'  Generated: {len(mesh.faces)} faces, watertight={mesh.is_watertight}')
"

    python -u cfd/run_case.py \
        --stl "$STL_PATH" \
        --velocity "$VELOCITY" \
        --aoa "$AOA" \
        --output "$CASE_DIR" \
        --nprocs "$SLURM_NTASKS" \
        --rocket-length 0.669 \
        --body-radius 0.0375 \
        --serial-mesh
    echo ""
done

echo "=== Canard Sweep Results ==="
python3 -u -c "
import json, os

results = []
for defl in [0, 5, 10, 15]:
    f = '${RESULTS_DIR}/defl_{}/cfd_results.json'.format(defl)
    if os.path.exists(f):
        with open(f) as fh:
            r = json.load(fh)
            r['canard_deflection'] = defl
            results.append(r)
            print(f'delta={defl:3d} deg:  Cd={r.get(\"Cd\",0):.4f}  Cl={r.get(\"Cl\",0):.6f}  CmPitch={r.get(\"CmPitch\",0):.6f}')
    else:
        print(f'delta={defl:3d} deg: FAILED')

with open('${RESULTS_DIR}/canard_sweep_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)

if len(results) >= 2:
    cl = [r.get('Cl', 0) for r in results]
    defl = [r['canard_deflection'] for r in results]
    if len(results) >= 2:
        dcl_dd = (cl[-1] - cl[0]) / (defl[-1] - defl[0]) if defl[-1] != defl[0] else 0
        print(f'')
        print(f'Control effectiveness dCl/d_delta = {dcl_dd:.6f} /deg')
"

echo ""
echo "Done: $(date)"
