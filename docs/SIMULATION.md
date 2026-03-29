# Simulation Framework

A 6-DOF (six degrees of freedom) simulation of the canard-stabilised rocket, used
to optimise PID gains and validate flight reliability under uncertainty.

---

## Architecture

```
Simulation/
├── src/
│   ├── rocket_dynamics.py   # 6-DOF physics engine
│   ├── pid_optimizer.py     # Differential evolution + grid search
│   ├── monte_carlo.py       # Statistical flight analysis
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
| Thrust             | Time-varying solid motor profile (C-class, ~15 N peak) |
| Gravity            | Transformed to body frame via rotation matrix       |
| Aerodynamic drag   | Axial, based on Cd0 + incremental canard drag       |
| Normal force       | CNa-based restoring force from body + fins          |
| Canard lift        | Per-canard flat-plate lift from deflection angle     |

### Moments Modelled
| Moment             | Description                                         |
|--------------------|-----------------------------------------------------|
| Roll (canard)      | N × F_canard × arm for all 4 canards               |
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

Roll-only PD(I) controller matching the real firmware:

```
output = Kp × roll_angle + Ki × ∫roll_angle + Kd × roll_rate
```

All 4 canards deflect by the same amount, producing pure roll torque. The D-term
uses the gyro rate directly (not differentiated error), matching `main.cpp` line 107.

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

# Full pipeline (optimise → MC → report)
sbatch run_full_analysis.sh
```

Monitor with `squeue -u $USER` and check output in `Simulation/results/`.

---

## PID Optimisation

### Differential Evolution
- Population: 40, Generations: 60
- Wind speeds: 0, 2, 5 m/s crosswind
- Objective: minimise roll RMS during motor burn phase
- Parameters: Kp_roll ∈ [0.1, 3.0], Ki_roll ∈ [0, 0.5], Kd_roll ∈ [0.02, 1.5]

### Grid Search
For verification. Tests a grid of Kp × Ki × Kd values.

Results are saved to `results/best_gains.json`.

---

## Monte Carlo Analysis

- Randomises: mass, CG, inertia, drag, thrust, wind speed/direction, launch angle
- Success criteria: altitude > 20 m **and** roll RMS during burn < 15°
- Default: 2000 runs with parallel execution
- Reports success rate, altitude distribution, roll statistics

Results are saved to `results/monte_carlo_results.json`.

---

## Key Parameters

| Parameter        | Value   | Source                           |
|------------------|---------|----------------------------------|
| Mass             | 200 g   | BOM estimate                     |
| Length           | 30 cm   | CAD                              |
| Diameter         | 40 mm   | Tube spec                        |
| Stability margin | 2 cm    | OpenRocket (1.0 calibre)         |
| Motor impulse    | 10 N·s  | C-class solid                    |
| Peak thrust      | 15 N    | Motor data                       |
| Burn time        | 1.2 s   | Motor data                       |
| Canard area      | 6 cm²   | Per canard (~20×30 mm)           |
| CNa              | 8.0/rad | OpenRocket full-rocket slope     |
| Cd0              | 0.45    | Including base drag              |
