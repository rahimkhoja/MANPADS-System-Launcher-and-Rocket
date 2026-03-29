# Aerodynamic Analysis and Design Improvement

## Overview

This document describes the aerodynamic analysis pipeline used to improve the
rocket's all-weather performance. Starting from the original CAD geometry
(measured from STL exports), we applied the Barrowman method to sweep 38,880
design configurations, identified optimal fin/canard/nose parameters, and
validated the improvement through 6-DOF simulation with Monte Carlo analysis.

## Problem Statement

The original rocket achieved only **10.4% Monte Carlo success rate** under
0–5 m/s wind conditions. Root cause analysis revealed inadequate roll control
authority: the canards barely protruded beyond the 75 mm body tube, producing
insufficient torque to counter crosswind disturbances during the 1.2 s motor
burn.

## Baseline Geometry (from CAD STL)

Dimensions extracted from `cad/rocket_assembly.stl` using trimesh:

| Parameter | Value |
|---|---|
| Total length | 587 mm |
| Body OD | 75 mm |
| Nose length | 38 mm (ogive) |
| Nose shape | Short ogive |
| Canard protrusion | ~2.6 mm beyond body |
| Fin protrusion | ~2.6 mm beyond body |
| Launcher tube ID | 95 mm (from `cad/launcher_tube.stl`) |

**Barrowman analysis of baseline:**
- CN_alpha total: 2.05 /rad
- CP: 124 mm from nose
- Stability: **-4.33 calibres** (unstable without large fins deployed)
- Cd0: 0.70
- Roll authority: 0.000007 Nm/(rad·Pa)

## Analysis Methods

### Barrowman Method (`Simulation/src/barrowman.py`)

Implementation of the Barrowman (1966) subsonic aerodynamic prediction method:

- **Normal force slopes** for nose, fins, and canards using the corrected
  Barrowman equation:
  ```
  CNa = K_fb × 4N(s/d)² / (1 + sqrt(1 + (AR/(2·cos(Λ)))²))
  ```
  where K_fb is the fin-body interference factor, N is fin count, s is
  semi-span, d is body diameter, AR is aspect ratio, and Λ is sweep angle.

- **Center of pressure** via weighted average of component CPs.

- **Drag estimation** using turbulent flat-plate skin friction (Schlichting),
  base drag, nose pressure drag, and fin leading-edge bluntness drag.

- **Roll authority** computed as canard count × planform area × 3D lift slope
  × moment arm.

Supports conical, ogive, Von Karman, and parabolic nose cones; trapezoidal
fins and canards with arbitrary sweep.

### Parametric Geometry Generator (`Simulation/src/rocket_geometry.py`)

Generates watertight STL meshes from parametric inputs using trimesh:
- Nose cone (conical, ogive, Von Karman)
- Cylindrical body tube with optional boat tail
- Trapezoidal fin sets at arbitrary positions
- Flat-plate canard sets

### OpenFOAM CFD (`Simulation/cfd/`)

Steady-state RANS simulation using k-omega SST turbulence model:
- `snappyHexMesh` for mesh generation around STL geometry
- `simpleFoam` for incompressible steady-state solution
- Parametric Python driver (`run_case.py`) sets velocity, AoA, and STL
- Extracts Cd, Cl, Cm from converged force coefficients

## Design Sweep Results

The design sweep (`Simulation/src/design_sweep.py`) evaluated **38,880
configurations** varying:

| Parameter | Range |
|---|---|
| Nose shape | conical, ogive, Von Karman |
| Nose length | 60 – 120 mm |
| Canard span | 10 – 30 mm |
| Canard root chord | 25 – 45 mm |
| Canard position | 60 – 100 mm from nose |
| Fin span | 30 – 60 mm |
| Fin root chord | 50 – 90 mm |
| Fin sweep | 30° – 60° |
| Boat tail | yes / no |

### Scoring Criteria

1. **Must fit** within 200 mm deployed OD (folding fins)
2. **Stability** ≥ 1.0 calibres (CP aft of CG)
3. **Maximise** roll authority (canard torque per radian)
4. **Minimise** drag (Cd0)

### Winning Configuration

| Parameter | Baseline | Improved | Change |
|---|---|---|---|
| Nose shape | ogive (38mm) | Von Karman (120mm) | 3.2× longer |
| Fin span | 2.6 mm | 60 mm | 23× larger |
| Fin root chord | ~20 mm | 70 mm | 3.5× larger |
| Fin sweep | ~45° | 30° | Lower sweep |
| Canard span | 2.6 mm | 20 mm | 7.7× larger |
| Canard root chord | ~20 mm | 45 mm | 2.3× larger |
| Boat tail | No | Yes (30mm, 60mm end ∅) | Added |
| **Stability** | **-4.33 cal** | **1.06 cal** | **Stable** |
| **CNa** | **2.05** | **8.21 /rad** | **4.0×** |
| **Cd0** | **0.70** | **0.64** | **-9%** |
| **Roll authority** | **7e-6** | **3e-4** | **40.8×** |

## Key Design Insights

1. **The body dominates**: A 75 mm diameter, 587 mm long body creates
   substantial destabilising lift (CN_alpha = 2.0 from nose alone). Without
   proportionally large fins, the rocket is deeply unstable.

2. **Folding fins must be aggressive**: The original 2.6 mm fin protrusion
   contributes almost nothing. To overcome the body destabilisation, fins need
   60 mm semi-span (total deployed OD ~195 mm).

3. **Longer nose reduces drag**: Extending from 38 mm ogive to 120 mm Von
   Karman reduces nose pressure drag substantially while barely affecting CP.

4. **Canard sizing for roll authority**: The 40× improvement in roll authority
   comes primarily from the 7.7× increase in canard span (torque scales with
   span × chord × lift slope).

5. **Boat tail reduces base drag**: A 30 mm boat tail taper reduces the base
   drag coefficient by approximately 25%.

## Simulation Validation

The improved design parameters are integrated into `Simulation/src/rocket_dynamics.py`
as the default `RocketParameters`. PID gains were re-optimised using differential
evolution on the Eureka cluster, followed by a 2000-run Monte Carlo analysis.

See `Simulation/results/` for the latest optimisation and Monte Carlo output
files.

## Reproducing the Results

```bash
# Run the design sweep (or re-use JSON + export STLs only)
cd Simulation/src
python design_sweep.py --output ../results/sweep_results.json --generate-stl ../results/candidate_stls/ --top 5
python design_sweep.py --from-json ../results/sweep_results.json --generate-stl ../results/candidate_stls --top 5

# Run the Barrowman calculator standalone
python barrowman.py

# Generate a parametric rocket STL
python rocket_geometry.py --nose vonkarman --nose-length 0.120 --fin-span 0.060 --canard-span 0.020 --boat-tail --output improved_rocket.stl

# Run CFD (requires OpenFOAM; --serial-mesh recommended on clusters)
cd ../cfd
python run_case.py --stl ../../cad/rocket_assembly.stl --velocity 40 --aoa 0 \
  --output baseline_cfd --serial-mesh

# Submit cluster jobs
cd ../jobs
sbatch run_design_sweep.sh
sbatch run_pid_optimizer.sh
sbatch --dependency=afterok:<pid_job_id> run_monte_carlo.sh
sbatch run_cfd.sh ../../cad/rocket_assembly.stl
```

## References

- J.S. Barrowman, "The Theoretical Prediction of the Center of Pressure",
  SRAD, 1966.
- OpenFOAM v2412, k-omega SST turbulence model.
- Schlichting, H., "Boundary Layer Theory", 7th ed., for skin friction
  correlations.
