# Eureka Cluster Guide

Instructions for running the rocket simulation framework on the Digital Research
Alliance of Canada (formerly Compute Canada) Eureka cluster.

---

## Environment

| Component    | Details                        |
|--------------|--------------------------------|
| Scheduler    | SLURM                          |
| Software     | CVMFS module system            |
| StdEnv       | StdEnv/2023                    |
| Python       | 3.11.5 (module)                |
| NumPy/SciPy  | scipy-stack/2024a (module)     |
| GPUs         | Available via `--gres=gpu:1`   |
| CUDA         | 12.2 (module)                  |

---

## Quick Start

```bash
# Load the software stack
module load StdEnv/2023 python/3.11.5 scipy-stack/2024a

# Create a virtual environment (one-time)
python -m venv --system-site-packages ~/rocket_venv
source ~/rocket_venv/bin/activate

# Navigate to the simulation
cd MANPADS-System-Launcher-and-Rocket/Simulation/jobs

# Run the quick validation (< 2 minutes)
sbatch run_validation.sh
```

---

## SLURM Job Scripts

| Script                  | CPUs | Time   | Description                     |
|-------------------------|------|--------|---------------------------------|
| `run_validation.sh`     | 4    | 10 min | Smoke test — verify everything works |
| `run_pid_optimizer.sh`  | 16   | 2 hr   | Grid + DE roll-only gain tuning |
| `run_pid_sixaxis.sh`    | 16   | 2 hr   | DE with roll+pitch objective (`--six-axis`) |
| `run_monte_carlo.sh`    | 16   | 2 hr   | 2000-run statistical analysis   |
| `run_cfd.sh`            | 16   | 6 hr   | OpenFOAM matrix (`--serial-mesh` in driver) |
| `run_design_sweep.sh`   | 1    | 30 min | Barrowman sweep (+ optional STLs) |
| `run_rl_training.sh`    | 8+GPU| 4 hr   | PPO reinforcement learning      |
| `run_full_analysis.sh`  | 16   | 6 hr   | Complete pipeline               |

### Submitting Jobs
```bash
cd Simulation/jobs
sbatch run_validation.sh          # Submit
squeue -u $USER                   # Check status
scancel <job_id>                  # Cancel
```

### Viewing Output
```bash
# Real-time output
tail -f ../results/validate_<job_id>.out

# Error log
cat ../results/validate_<job_id>.err
```

---

## Module Commands

```bash
module avail python              # List available Python versions
module avail scipy               # List scipy-stack versions
module load StdEnv/2023          # Base environment
module load python/3.11.5        # Python interpreter
module load scipy-stack/2024a    # NumPy, SciPy, matplotlib, etc.
module load cuda/12.2            # CUDA toolkit (for GPU jobs)
module list                      # Show loaded modules
module purge                     # Unload all
```

---

## Virtual Environment

SLURM jobs create a temporary venv in `$SLURM_TMPDIR` (fast local SSD) for each
job. For interactive work, create a persistent venv:

```bash
module load StdEnv/2023 python/3.11.5 scipy-stack/2024a
python -m venv --system-site-packages ~/rocket_venv
source ~/rocket_venv/bin/activate
pip install torch  # If needed for RL training
```

The `--system-site-packages` flag gives the venv access to the cluster's
pre-built NumPy/SciPy/matplotlib, which are compiled with optimised BLAS.

---

## Interactive Testing

For quick debugging without submitting a job:

```bash
module load StdEnv/2023 python/3.11.5 scipy-stack/2024a
cd MANPADS-System-Launcher-and-Rocket/Simulation/src

# Run a single simulation
python rocket_dynamics.py

# Small optimizer test
python pid_optimizer.py --method grid --grid-points 3 --num-runs 1 --wind-speeds 0 2
```

Note: the login node is shared. Keep interactive runs short (< 5 minutes) or
use `salloc` to get an interactive compute node.

---

## Parallel Execution

The optimizer and Monte Carlo scripts support parallel execution via Python's
`ProcessPoolExecutor`. The `--workers` flag controls the number of parallel
processes:

```bash
# Use all allocated CPUs
python pid_optimizer.py --workers ${SLURM_CPUS_PER_TASK}
```

Each worker runs an independent simulation, so scaling is nearly linear up to
the number of physical cores.

---

## Results

All output goes to `Simulation/results/`:

| File                         | Contents                              |
|------------------------------|---------------------------------------|
| `best_gains.json`            | Optimised Kp, Ki, Kd gains           |
| `opt_history_de.json`        | Full DE convergence history           |
| `monte_carlo_results.json`   | MC statistics + per-run data          |
| `*.out` / `*.err`            | SLURM job stdout/stderr              |

---

## Troubleshooting

| Problem                          | Solution                              |
|----------------------------------|---------------------------------------|
| `ModuleNotFoundError: scipy`     | `module load scipy-stack/2024a`       |
| No output from job               | Add `export PYTHONUNBUFFERED=1`       |
| Job killed (OOM)                 | Increase `--mem`                      |
| `pickle` error in parallel       | Ensure worker functions are at module level |
| Stale results                    | Check `--output-dir` path             |
