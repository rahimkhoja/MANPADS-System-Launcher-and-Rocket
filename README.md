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
Differential evolution search over roll control gains (Kp, Ki, Kd), tested
across multiple wind conditions. Runs on multi-core SLURM nodes.

### Monte Carlo Reliability Analysis
Statistical assessment of flight performance under parameter uncertainty
(mass, thrust, wind, launch angle, sensor noise). Reports success rates
and confidence intervals.

### Extended Kalman Filter (V5 Firmware)
Drift-free attitude estimation fusing gyroscope and accelerometer data.
Automatic apogee detection. Backward-compatible with the V4 launcher.

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
│   ├── src/                   # 6-DOF simulation, optimiser, Monte Carlo
│   ├── jobs/                  # SLURM job scripts
│   └── results/               # Output data
├── Mechanical/                # Fusion 360 CAD files
├── OpenRocket/                # Aerodynamic stability analysis
└── docs/
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
cd Simulation/src
pip install numpy scipy matplotlib
python rocket_dynamics.py
```

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
| [SIMULATION.md](docs/SIMULATION.md) | Simulation physics, running instructions, parameters |
| [FIRMWARE_IMPROVEMENTS.md](docs/FIRMWARE_IMPROVEMENTS.md) | V4 → V5 firmware changes |
| [CLUSTER_GUIDE.md](docs/CLUSTER_GUIDE.md) | Eureka SLURM cluster usage |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Fork details, credits, AI workflow |
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
