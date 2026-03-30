# MANPADS Rocket & Launcher Prototype

> **Fork notice:** This is an enhanced fork of the original project by
> [Rahim Khoja](https://github.com/rahimkhoja). The simulation framework,
> firmware improvements, and documentation were developed with the assistance
> of [Cursor](https://cursor.com), an AI-powered code editor, running
> analysis on the Digital Research Alliance of Canada's Eureka HPC cluster.

---

## 30 Second Overview

[![30 Second Overview](https://img.youtube.com/vi/zFn__6_LdTc/hqdefault.jpg)](https://www.youtube.com/shorts/zFn__6_LdTc)

## Full System Overview (5 Minutes)

[![Full System Overview](https://img.youtube.com/vi/DDO2EvXyncE/hqdefault.jpg)](https://www.youtube.com/watch?v=DDO2EvXyncE&t=59s)

---

## Project Overview

A proof-of-concept prototype of a low-cost rocket launcher and guided rocket
system built using consumer electronics and 3D-printed components.

The rocket uses folding fins and canard stabilisation controlled by an onboard
ESP32 flight computer and MPU6050 IMU. The launcher integrates GPS, compass,
and barometric sensors to provide telemetry.

Total hardware cost: approximately **$96**.

---

## What's New in This Fork

### 6-DOF Flight Simulation
A complete rigid-body simulation modelling aerodynamic forces, motor thrust,
wind disturbance (Dryden turbulence), and quaternion-based attitude dynamics.
Used to optimise PID gains and validate flight reliability.

### PID Gain Optimisation
Differential evolution over roll PID (and optional **six-axis** roll+pitch
objective via `pid_optimizer.py --six-axis`). Expanded gain bounds (Kd up to 5).
Cluster: `run_pid_optimizer.sh` (roll-only default), `run_pid_sixaxis.sh` for
pitch-augmented tuning.

### Monte Carlo Reliability Analysis
Statistical assessment of flight performance under parameter uncertainty
(mass, thrust, wind, launch angle, sensor noise). Reports success rates
and confidence intervals.

### Extended Kalman Filter (V5 Firmware)
Drift-free attitude estimation fusing gyroscope and accelerometer data.
Automatic apogee detection. Backward-compatible with the V4 launcher.

### Aerodynamic Design Improvement
Barrowman method sweep of 38,880 configurations to find optimal fin, canard,
and nose cone geometry. Von Karman nose, larger folding fins, and boat tail
improved stability from -4.3 to +1.1 calibres and increased roll authority by
40x. Candidate STLs generated parametrically for CFD validation.

### OpenFOAM CFD Integration
Automated steady-state RANS (k-omega SST) pipeline with snappyHexMesh,
parametric Python driver, and SLURM job scripts. **`run_cfd.sh` uses
`--serial-mesh`** (serial snappyHexMesh, parallel `simpleFoam`) and sources
Lmod under bash for reliable OpenFOAM modules on Eureka.

### Rail Launcher Redesign
The improved rocket's canards exceed the original 95 mm tube bore. A parametric
rail launcher (two parallel 2020 aluminium extrusion rails + base plate) replaces
the tube design, with nylon rail buttons on the rocket body.

### Material Recommendations
See [MATERIALS.md](docs/MATERIALS.md) for recommended materials per component.

### HPC Integration
SLURM job scripts for the Eureka cluster with proper module loading,
parallel execution, and automated pipelines.

---

## Repository Structure

```
├── Firmware/
│   ├── Rocket/src/            # ESP32 flight computer (V4 + V5)
│   ├── Launcher/src/          # ESP32 ground station
│   └── dashboard.py           # Python GUI
├── Simulation/
│   ├── src/                   # 6-DOF sim, optimiser, Monte Carlo, Barrowman
│   ├── cfd/                   # OpenFOAM templates and CFD driver
│   ├── jobs/                  # SLURM job scripts
│   └── results/               # Output data and candidate STLs
├── cad/                       # STL exports (Unix-friendly names)
│   └── fusion360/             # Original Fusion 360 .f3z archives
├── Mechanical/                # Legacy CAD location
├── OpenRocket/                # Aerodynamic stability analysis
└── docs/
    ├── AERODYNAMICS.md        # Barrowman analysis, design sweep, CFD
    ├── WIRING.md              # Pin assignments (verified against firmware)
    ├── SIMULATION.md          # Simulation framework guide
    ├── FIRMWARE_IMPROVEMENTS.md  # V4 → V5 changelog
    ├── CLUSTER_GUIDE.md       # Eureka HPC usage guide
    └── CONTRIBUTING.md        # Fork details, credits, workflow
```

---

## Quick Start

### Run the Simulation Locally
```bash
cd Simulation
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy trimesh matplotlib
python src/rocket_dynamics.py
```
Motor **total impulse is enforced at 10 N·s** (thrust curve is scaled); expect
lower apogee than an unscaled 15 N plateau used in older runs.

### Run on Eureka Cluster
```bash
cd Simulation/jobs
sbatch run_validation.sh       # Quick smoke test (< 2 min)
sbatch run_full_analysis.sh    # Full pipeline (optimise → Monte Carlo)
```

### Build the Firmware
```bash
cd Firmware/Rocket
pio run                        # Compile
pio run -t upload              # Flash to ESP32
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [AERODYNAMICS.md](docs/AERODYNAMICS.md) | Barrowman analysis, design sweep, CFD pipeline |
| [SIMULATION.md](docs/SIMULATION.md) | Simulation physics, running instructions, parameters |
| [FIRMWARE_IMPROVEMENTS.md](docs/FIRMWARE_IMPROVEMENTS.md) | V4 → V5 firmware changes |
| [CLUSTER_GUIDE.md](docs/CLUSTER_GUIDE.md) | Eureka SLURM cluster usage |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Fork details, credits, AI workflow |
| [MATERIALS.md](docs/MATERIALS.md) | Recommended materials per component |
| [WIRING.md](docs/WIRING.md) | Pin assignments and telemetry protocol |

---

## Full Development Media
https://drive.google.com/drive/folders/17zpks6_R59H0iXJaGkTrtp1SzIFFAQtY?usp=drive_link

---

## Credits

- **Original project:** [rahimkhoja/MANPADS-System-Launcher-and-Rocket](https://github.com/rahimkhoja/MANPADS-System-Launcher-and-Rocket)
- **Fork maintainer:** Rahim Khoja ([@rahimkhoja](https://github.com/rahimkhoja))
- **AI-assisted development:** [Cursor](https://cursor.com)
- **HPC resources:** Digital Research Alliance of Canada — Eureka cluster

*Last updated: March 2026*
