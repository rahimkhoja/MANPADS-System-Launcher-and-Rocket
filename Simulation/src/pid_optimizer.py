"""
PID Gain Optimizer
==================
Roll-only or roll+pitch gain tuning via grid search or differential evolution.
Objective: minimise roll RMS during motor burn (optionally add pitch error).
"""

import numpy as np
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Any
import itertools

from rocket_dynamics import (
    RocketSimulator, CanardController, WindModel,
    RocketParameters, MotorProfile,
)


def _evaluate_gains(gains: Dict[str, float], wind_speeds, gust_frac: float,
                    runs_per_wind: int, base_seed: int,
                    six_axis_cost: bool) -> float:
    ctrl = CanardController(
        Kp_roll=gains.get('Kp_roll', 0.5),
        Ki_roll=gains.get('Ki_roll', 0.0),
        Kd_roll=gains.get('Kd_roll', 0.2),
        Kp_pitch=gains.get('Kp_pitch', 0.0),
        Ki_pitch=gains.get('Ki_pitch', 0.0),
        Kd_pitch=gains.get('Kd_pitch', 0.0),
        Kp_yaw=gains.get('Kp_yaw', 0.0),
        Ki_yaw=gains.get('Ki_yaw', 0.0),
        Kd_yaw=gains.get('Kd_yaw', 0.0),
    )
    costs = []
    for wi, ws in enumerate(wind_speeds):
        for ri in range(runs_per_wind):
            seed = base_seed + wi * 100 + ri
            wind = WindModel(
                base_velocity=np.array([ws, 0.0, 0.0]),
                gust_intensity=ws * gust_frac,
            )
            wind.seed(seed)
            sim = RocketSimulator(controller=ctrl, wind=wind)
            r = sim.run(duration=10.0, dt=0.005)
            if not r:
                costs.append(1e3)
                continue
            rb = r.get('roll_rms_burn', r.get('roll_rms', 1e3))
            if six_axis_cost:
                pb = r.get('pitch_rms_burn', 0.0)
                costs.append(rb + 0.25 * pb)
            else:
                costs.append(rb)
    return float(np.mean(costs))


def _worker(args):
    """Parallel worker: (gains_dict, wind_speeds, gust_frac, runs, seed, six_axis_cost)."""
    gains, wind_speeds, gust_frac, runs_per_wind, base_seed, six_axis_cost = args
    return _evaluate_gains(gains, wind_speeds, gust_frac, runs_per_wind,
                           base_seed, six_axis_cost)


def _gains_vec_to_dict_roll_only(ind: np.ndarray, names: List[str]) -> Dict[str, float]:
    return {n: float(v) for n, v in zip(names, ind)}


def _gains_vec_to_dict_six_axis(vec: np.ndarray) -> Dict[str, float]:
    Kp_r, Ki_r, Kd_r, Kp_p, Ki_p, Kd_p = [float(x) for x in vec]
    return {
        'Kp_roll': Kp_r, 'Ki_roll': Ki_r, 'Kd_roll': Kd_r,
        'Kp_pitch': Kp_p, 'Ki_pitch': Ki_p, 'Kd_pitch': Kd_p,
        'Kp_yaw': 0.2 * Kp_p, 'Ki_yaw': 0.2 * Ki_p, 'Kd_yaw': 0.2 * Kd_p,
    }


class PIDOptimizer:

    def __init__(self, wind_speeds=None, num_runs_per_config=2,
                 num_workers=None, gust_frac=0.3, six_axis=False):
        self.wind_speeds = wind_speeds or [0, 2, 5]
        self.num_runs = num_runs_per_config
        self.num_workers = num_workers
        self.gust_frac = gust_frac
        self.six_axis = six_axis
        self.history = []

    def _pack(self, gains: Dict[str, float], seed=0):
        return (gains, self.wind_speeds, self.gust_frac,
                self.num_runs, seed, self.six_axis)

    def objective(self, gains: Dict[str, float], seed=0) -> float:
        return _worker(self._pack(gains, seed))

    # ------------------------------------------------------------------
    def grid_search(self, ranges: dict) -> Tuple[dict, float]:
        Kp_vals = ranges.get('Kp_roll', [0.5])
        Ki_vals = ranges.get('Ki_roll', [0.0])
        Kd_vals = ranges.get('Kd_roll', [0.2])

        combos = list(itertools.product(Kp_vals, Ki_vals, Kd_vals))
        print(f"Grid search: {len(combos)} combinations", flush=True)

        args_list = [
            self._pack({'Kp_roll': kp, 'Ki_roll': ki, 'Kd_roll': kd})
            for kp, ki, kd in combos
        ]

        if self.num_workers and self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                scores = list(ex.map(_worker, args_list))
        else:
            scores = [_worker(a) for a in args_list]

        best_idx = int(np.argmin(scores))
        kp, ki, kd = combos[best_idx]
        best = {'Kp_roll': kp, 'Ki_roll': ki, 'Kd_roll': kd}

        for combo, score in zip(combos, scores):
            self.history.append({
                'gains': {'Kp_roll': combo[0], 'Ki_roll': combo[1], 'Kd_roll': combo[2]},
                'score': score,
            })

        print(f"Best grid: score={scores[best_idx]:.3f}  {best}", flush=True)
        return best, scores[best_idx]

    # ------------------------------------------------------------------
    def differential_evolution(self, bounds: dict,
                               pop_size=40, generations=60,
                               F=0.7, CR=0.8) -> Tuple[dict, float]:
        names = list(bounds.keys())
        lo = np.array([bounds[n][0] for n in names])
        hi = np.array([bounds[n][1] for n in names])
        D = len(names)

        pop = np.random.uniform(lo, hi, (pop_size, D))

        def vec_to_gains(ind: np.ndarray) -> Dict[str, float]:
            if self.six_axis and D == 6:
                return _gains_vec_to_dict_six_axis(ind)
            return _gains_vec_to_dict_roll_only(ind, names)

        def to_args(ind, seed=0):
            return self._pack(vec_to_gains(ind), seed)

        print(f"DE: pop={pop_size}, gen={generations}, F={F}, CR={CR}", flush=True)

        if self.num_workers and self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                fit = np.array(list(ex.map(_worker, [to_args(p) for p in pop])))
        else:
            fit = np.array([_worker(to_args(p)) for p in pop])

        bi = int(np.argmin(fit))
        best_ind = pop[bi].copy()
        best_fit = fit[bi]
        print(f"  Init best: {best_fit:.3f}  {vec_to_gains(best_ind)}", flush=True)

        for gen in range(generations):
            trials = np.empty_like(pop)
            for i in range(pop_size):
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lo, hi)
                mask = np.random.rand(D) < CR
                if not mask.any():
                    mask[np.random.randint(D)] = True
                trials[i] = np.where(mask, mutant, pop[i])

            seed_base = (gen + 1) * 1000
            if self.num_workers and self.num_workers > 1:
                with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                    t_fit = np.array(list(ex.map(
                        _worker, [to_args(t, seed_base) for t in trials])))
            else:
                t_fit = np.array([_worker(to_args(t, seed_base)) for t in trials])

            improved = t_fit < fit
            pop[improved] = trials[improved]
            fit[improved] = t_fit[improved]

            bi = int(np.argmin(fit))
            if fit[bi] < best_fit:
                best_fit = fit[bi]
                best_ind = pop[bi].copy()

            if (gen + 1) % 5 == 0 or gen == 0:
                print(f"  Gen {gen+1:3d}: best={best_fit:.3f}  "
                      f"mean={np.mean(fit):.3f}  "
                      f"{vec_to_gains(best_ind)}", flush=True)

            self.history.append({
                'generation': gen,
                'best_fitness': float(best_fit),
                'mean_fitness': float(np.mean(fit)),
                'best_gains': vec_to_gains(best_ind),
            })

        best_gains = vec_to_gains(best_ind)
        return best_gains, float(best_fit)

    def save_results(self, path):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2, default=float)


def main():
    parser = argparse.ArgumentParser(description='PID Roll / Roll+Pitch Gain Optimizer')
    parser.add_argument('--method', choices=['grid', 'de'], default='de')
    parser.add_argument('--six-axis', action='store_true',
                        help='Optimise roll + pitch (6 DE vars); yaw slaved to pitch')
    parser.add_argument('--wind-speeds', type=float, nargs='+', default=[0, 2, 5])
    parser.add_argument('--num-runs', type=int, default=2)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--grid-points', type=int, default=8)
    parser.add_argument('--population', type=int, default=40)
    parser.add_argument('--generations', type=int, default=60)
    parser.add_argument('--output-dir', type=str, default='../results')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("PID Gain Optimizer — Canard-Stabilized Rocket", flush=True)
    print("=" * 60, flush=True)
    print(f"Wind speeds: {args.wind_speeds} m/s", flush=True)
    print(f"Runs per config per wind: {args.num_runs}", flush=True)
    print(f"Workers: {args.workers or 'serial'}", flush=True)
    print(f"Six-axis (roll+pitch cost): {args.six_axis}", flush=True)

    opt = PIDOptimizer(
        wind_speeds=args.wind_speeds,
        num_runs_per_config=args.num_runs,
        num_workers=args.workers,
        six_axis=args.six_axis,
    )

    t0 = time.time()

    if args.method == 'grid':
        if args.six_axis:
            print("Grid search with --six-axis is not supported; use --method de", flush=True)
            return
        ranges = {
            'Kp_roll': np.linspace(0.1, 2.0, args.grid_points).tolist(),
            'Ki_roll': np.linspace(0.0, 0.5, max(args.grid_points // 2, 3)).tolist(),
            'Kd_roll': np.linspace(0.1, 5.0, args.grid_points).tolist(),
        }
        best_gains, best_score = opt.grid_search(ranges)
    else:
        if args.six_axis:
            bounds = {
                'Kp_roll': (0.05, 5.0),
                'Ki_roll': (0.0, 0.5),
                'Kd_roll': (0.05, 5.0),
                'Kp_pitch': (0.0, 3.0),
                'Ki_pitch': (0.0, 0.3),
                'Kd_pitch': (0.0, 3.0),
            }
        else:
            bounds = {
                'Kp_roll': (0.05, 5.0),
                'Ki_roll': (0.0, 0.5),
                'Kd_roll': (0.05, 5.0),
            }
        best_gains, best_score = opt.differential_evolution(
            bounds, pop_size=args.population, generations=args.generations)

    elapsed = time.time() - t0

    print("\n" + "=" * 60, flush=True)
    print("OPTIMIZATION COMPLETE", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)
    print(f"Best objective: {best_score:.4f}", flush=True)
    for k, v in best_gains.items():
        print(f"  {k}: {v:.4f}", flush=True)

    suffix = 'sixaxis' if args.six_axis else args.method
    opt.save_results(str(Path(args.output_dir) / f"opt_history_{suffix}.json"))

    result_payload = {
        'gains': best_gains,
        'score': best_score,
        'method': suffix,
        'six_axis': args.six_axis,
        'wind_speeds': args.wind_speeds,
        'elapsed_seconds': elapsed,
    }

    gains_filename = "best_gains_sixaxis.json" if args.six_axis else "best_gains_roll.json"
    gains_path = Path(args.output_dir) / gains_filename
    with open(gains_path, 'w') as f:
        json.dump(result_payload, f, indent=2)

    with open(Path(args.output_dir) / "best_gains.json", 'w') as f:
        json.dump(result_payload, f, indent=2)

    print(f"Results saved to {gains_path} (and best_gains.json)", flush=True)


if __name__ == "__main__":
    main()
