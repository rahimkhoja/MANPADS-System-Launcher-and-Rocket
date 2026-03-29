# Simulation Framework

A 6-DOF (six degrees of freedom) simulation of the canard-stabilised rocket, used
to optimise PID gains and validate flight reliability under uncertainty.

---

## Architecture

```
Simulation/
├── src/
│   ├── rocket_dynamics.py   # 6-DOF physics engine
│   ├── pid_optimizer.py     # DE + grid (roll-only or --six-axis)
│   ├── monte_carlo.py       # Statistical flight analysis
│   ├── design_sweep.py      # Barrowman sweep + STL export
│   ├── barrowman.py         # Barrowman aerodynamics
│   ├── rocket_geometry.py   # Parametric STL (trimesh)
│   ├── kalman_filter.py     # EKF implementation + C++ header generator
│   └── rl_controller.py     # Reinforcement learning (PPO) controller
├── jobs/
│   ├── run_validation.sh    # Quick < 2 min smoke test
│   ├── run_pid_optimizer.sh # Gain optimisation job
│   ├── run_monte_carlo.sh   # Monte Carlo analysis job
│   ├── run_rl_training.sh   # RL training job (GPU)
│   └── run_full_analysis.sh # Full pipeline
├── results/                 # Output directory for all runs
└── requirements.txt         # Python dependencies
```

---

## Physics Model (`rocket_dynamics.py`)

### Coordinate System
NED (North-East-Down). The body x-axis points along the rocket's longitudinal axis
(out the nose). Altitude is `-position[2]`.

### Forces Modelled
| Force              | Description                                         |
|--------------------|-----------------------------------------------------|
| Thrust             | Trapezoidal profile **scaled** so ∫F dt = 10 N·s (C-class) |
| Gravity            | Transformed to body frame via rotation matrix       |
| Aerodynamic drag   | Axial, based on Cd0 + incremental canard drag       |
| Normal force       | CNa-based restoring force from body + fins          |
| Canard lift        | Per-canard flat-plate lift from deflection angle     |

### Moments Modelled
| Moment             | Description                                         |
|--------------------|-----------------------------------------------------|
| Roll/pitch/yaw (canard) | Differential [L,R,U,D] deflections → moments via `mix_canards` |
| Roll damping       | Clp × q̄Sd × (d/2V) × p                            |
| Pitch restoring    | CNa × stability_margin × α × q̄SL                   |
| Pitch damping      | Cmq × q̄SL × (d/2V) × q                            |
| Yaw restoring      | CNa × stability_margin × β × q̄SL                   |
| Yaw damping        | Cnr × q̄SL × (d/2V) × r                            |

### Wind Model
Dryden-style turbulence with a first-order filter (time constant τ = 0.5 s) applied
to white noise. No per-timestep RNG reseeding — the state is continuous.

### Integration
4th-order Runge-Kutta at 200 Hz (dt = 0.005 s). Quaternion-based attitude to avoid
gimbal lock.

---

## Controller

`CanardController` runs independent PIDs on roll, pitch, and yaw (set pitch/yaw
gains to zero for roll-only, matching V4/V5 firmware). D-term uses body angular
rate in deg/s (same convention as the rocket firmware). Outputs are mixed to four
surfaces:

`[L,R,U,D] = mix(roll_cmd, pitch_cmd, yaw_cmd)` with saturation on the 4-vector.

`pitch_rms_burn` in results is RMS deviation from **launch pitch** during burn.

---

## Running Locally

```bash
cd Simulation/src

# Single simulation
python rocket_dynamics.py

# Small grid search
python pid_optimizer.py --method grid --grid-points 5 --wind-speeds 0 3 --num-runs 1

# Small Monte Carlo
python monte_carlo.py --runs 100 --seed 42

# Six-axis PID tuning (roll + pitch in objective)
python pid_optimizer.py --method de --six-axis --workers 8

# Re-export top STLs from saved sweep (no full 38k re-sweep)
python design_sweep.py --from-json ../results/sweep_results.json \
  --generate-stl ../results/candidate_stls --top 5

# Generate Kalman filter C++ header
python kalman_filter.py
```

---

## Running on SLURM Cluster

```bash
cd Simulation/jobs

# Quick validation (< 2 minutes, 4 CPUs)
sbatch run_validation.sh

# PID optimisation (DE, ~1-2 hours, 16 CPUs)
sbatch run_pid_optimizer.sh

# Monte Carlo (2000 runs, ~1 hour, 16 CPUs)
sbatch run_monte_carlo.sh

# Optional: six-axis PID (overwrites best_gains.json — merge with roll-only if needed)
sbatch run_pid_sixaxis.sh

# CFD sweep (bash + Lmod + --serial-mesh inside driver)
sbatch run_cfd.sh ../../cad/rocket_assembly.stl

# Full pipeline (optimise → MC → report)
sbatch run_full_analysis.sh
```

Monitor with `squeue -u $USER` and check output in `Simulation/results/`.

---

## PID Optimisation

### Differential Evolution
- Population: 40, Generations: 60 (defaults; jobs may use 30×40)
- Wind speeds: 0, 2, 5 m/s crosswind
- Objective: minimise roll RMS during burn; with `--six-axis`, roll RMS + 0.25× pitch RMS (burn)
- Roll-only bounds: Kp_roll ∈ [0.05, 5.0], Ki_roll ∈ [0, 0.5], Kd_roll ∈ [0.05, 5.0]
- Six-axis adds Kp/Ki/Kd_pitch with yaw slaved to 0.2× pitch gains in the optimiser

### Grid Search
For verification. Tests a grid of Kp × Ki × Kd values.

Results are saved to `results/best_gains.json`.

---

## Monte Carlo Analysis

- Randomises: mass, CG, inertia, drag, thrust, wind speed/direction, launch angle
- Success criteria: altitude > 20 m, roll RMS during burn < 15°; if `Kp_pitch > 0.01`, also pitch RMS (burn) < 30°
- Default: 2000 runs with parallel execution
- Reports success rate, altitude distribution, roll statistics; **`elapsed_seconds`** is recorded correctly in the JSON summary

Results are saved to `results/monte_carlo_results.json`.

---

## Key Parameters (Improved Design)

| Parameter        | Value    | Source                               |
|------------------|----------|--------------------------------------|
| Mass             | 200 g    | BOM estimate                         |
| Length           | 669 mm   | STL measurement (120mm nose + 549mm) |
| Diameter         | 75 mm    | STL measurement                      |
| Stability margin | 1.06 cal | Barrowman analysis                   |
| Motor impulse    | 10 N·s   | C-class solid                        |
| Peak thrust      | 15 N     | Motor data                           |
| Burn time        | 1.2 s    | Motor data                           |
| Canard area      | 9.0 cm²  | Per canard (20×45 mm planform, improved) |
| Canard span      | 20 mm    | Semi-span beyond body                |
| Fin span         | 60 mm    | Semi-span, folding                   |
| CNa              | 8.2/rad  | Barrowman                            |
| Cd0              | 0.64     | Barrowman (Von Karman + boat tail)   |

See [AERODYNAMICS.md](AERODYNAMICS.md) for the full design sweep methodology
and the comparison between baseline and improved configurations.

---

## Barrowman Aerodynamics

The `Simulation/src/barrowman.py` module implements the Barrowman (1966) method
for rapid aerodynamic coefficient estimation. It can evaluate a complete rocket
configuration in under 1 ms, enabling grid sweeps of tens of thousands of
designs.

```bash
python Simulation/src/barrowman.py
```

## OpenFOAM CFD

For higher-fidelity validation, the `Simulation/cfd/` directory provides an
automated OpenFOAM pipeline:

```bash
# Set up and run a single case (add --serial-mesh if parallel snappyHexMesh crashes)
python Simulation/cfd/run_case.py --stl cad/rocket_assembly.stl --velocity 40 --aoa 5 --serial-mesh

# Submit a full sweep on SLURM (script enables --serial-mesh and bash+Lmod)
cd Simulation/jobs
sbatch run_cfd.sh ../../cad/rocket_assembly.stl
```

See the [AERODYNAMICS.md](AERODYNAMICS.md) documentation for details.
