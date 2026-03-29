# Contributing

## About This Fork

This repository is a fork of
[rahimkhoja/MANPADS-System-Launcher-and-Rocket](https://github.com/rahimkhoja/MANPADS-System-Launcher-and-Rocket),
the original $96 canard-stabilised rocket project.

**Fork maintainer:** Rahim Khoja ([@rahimkhoja](https://github.com/rahimkhoja))

### What This Fork Adds

- **6-DOF flight simulation** — full rigid-body physics with aerodynamic forces,
  motor thrust profiles, wind disturbance, and quaternion-based attitude
- **PID gain optimisation** — differential evolution and grid search to find
  optimal roll control gains under wind conditions
- **Monte Carlo reliability analysis** — statistical assessment of flight
  performance across parameter uncertainties
- **Extended Kalman Filter** — drift-free attitude estimation for the ESP32
  flight computer, with both Python and C++ implementations
- **Improved firmware (V5)** — EKF integration, automatic apogee detection,
  backward-compatible with V4 launcher
- **HPC integration** — SLURM job scripts for running optimisation and analysis
  on the Eureka cluster
- **Corrected documentation** — fixed pin assignments, added simulation docs,
  cluster guide, and firmware changelog

---

## AI-Assisted Development

This fork was developed with the assistance of **Cursor**, an AI-powered code
editor. The AI pair-programming workflow was used for:

- Writing and debugging the 6-DOF simulation physics
- Iterating on aerodynamic model corrections (wind model, canard moments,
  stability derivatives)
- Diagnosing and fixing Python multiprocessing/pickling issues
- Generating the EKF C++ header from the Python reference implementation
- Creating and debugging SLURM job scripts for the Eureka cluster
- Writing documentation

All code was reviewed and validated by running on the cluster. The simulation
results (gain optimisation, Monte Carlo analysis) are reproducible using the
scripts in `Simulation/jobs/`.

---

## Repository Structure

```
├── Firmware/
│   ├── Rocket/src/
│   │   ├── main.cpp              # V4 original firmware
│   │   ├── main_improved.cpp     # V5 improved firmware
│   │   └── attitude_ekf.h        # EKF header (generated)
│   ├── Launcher/src/main.cpp     # Launcher firmware
│   └── dashboard.py              # Tkinter ground station GUI
├── Simulation/
│   ├── src/                      # Python simulation code
│   ├── jobs/                     # SLURM job scripts
│   └── results/                  # Output data
├── Mechanical/                   # CAD files
├── OpenRocket/                   # Stability simulations
└── docs/                         # Documentation
```

---

## How to Reproduce Results

1. Clone the repository on the Eureka cluster (or any system with Python 3.11+
   and NumPy/SciPy)
2. Run the validation:
   ```bash
   cd Simulation/jobs
   sbatch run_validation.sh    # On cluster
   # or
   cd Simulation/src && python rocket_dynamics.py   # Locally
   ```
3. Run the full pipeline:
   ```bash
   sbatch run_full_analysis.sh
   ```
4. Results appear in `Simulation/results/`

See [SIMULATION.md](SIMULATION.md) and [CLUSTER_GUIDE.md](CLUSTER_GUIDE.md) for
detailed instructions.

---

## Development Setup

### Simulation (Python)
```bash
pip install numpy scipy matplotlib torch
cd Simulation/src
python rocket_dynamics.py
```

### Firmware (PlatformIO)
```bash
cd Firmware/Rocket
pio run                # Compile
pio run -t upload      # Flash
```

---

## Code Style

- Python: standard library conventions, type hints where helpful
- C++: Arduino conventions with PlatformIO
- Commits: short imperative messages describing the change
- Documentation: Markdown in `docs/`, keep in sync with code
