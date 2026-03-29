"""
Monte Carlo Trajectory Analysis
===============================
Statistical analysis of rocket flight performance under parameter uncertainty.
Evaluates reliability using burn-phase roll RMS as the primary success metric.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from rocket_dynamics import (
    RocketSimulator, RocketParameters, MotorProfile,
    WindModel, CanardController, AtmosphereModel
)


@dataclass
class UncertaintyParameters:
    mass_std: float = 0.005
    cg_std: float = 0.003
    Ixx_std_pct: float = 0.05
    Iyy_std_pct: float = 0.05
    Cd0_std_pct: float = 0.10
    thrust_std_pct: float = 0.08
    burn_time_std_pct: float = 0.03
    gyro_bias_std: float = 0.005
    launch_angle_std: float = 1.0
    wind_speed_mean: float = 2.0
    wind_speed_std: float = 1.5
    wind_direction_std: float = 30.0
    gust_frac: float = 0.3
    servo_offset_std: float = 1.0


@dataclass
class MCResult:
    run_id: int
    max_altitude: float
    apogee_time: float
    flight_time: float
    max_speed: float
    roll_rms: float
    roll_rms_burn: float
    landing_distance: float
    success: bool
    wind_speed: float
    launch_pitch: float
    mass: float
    thrust: float


def _run_one(args):
    """Module-level worker for parallel execution."""
    (run_id, seed, gains_dict, unc_dict) = args
    rng = np.random.RandomState(seed)
    u = UncertaintyParameters(**unc_dict)

    rocket = RocketParameters()
    rocket.mass = max(0.1, rng.normal(rocket.mass, u.mass_std))
    rocket.cg_from_nose = rng.normal(rocket.cg_from_nose, u.cg_std)
    rocket.Ixx *= max(0.5, rng.normal(1.0, u.Ixx_std_pct))
    rocket.Iyy *= max(0.5, rng.normal(1.0, u.Iyy_std_pct))
    rocket.Izz *= max(0.5, rng.normal(1.0, u.Iyy_std_pct))
    rocket.Cd0 *= max(0.5, rng.normal(1.0, u.Cd0_std_pct))

    motor = MotorProfile()
    motor.peak_thrust *= max(0.5, rng.normal(1.0, u.thrust_std_pct))
    motor.burn_time *= max(0.5, rng.normal(1.0, u.burn_time_std_pct))

    ws = max(0, rng.normal(u.wind_speed_mean, u.wind_speed_std))
    wd = rng.normal(0, u.wind_direction_std) * np.pi / 180
    wind = WindModel(
        base_velocity=np.array([ws * np.cos(wd), ws * np.sin(wd), 0.0]),
        gust_intensity=ws * u.gust_frac,
    )
    wind.seed(seed)

    launch_pitch = 85.0 + rng.normal(0, u.launch_angle_std)

    ctrl = CanardController(
        Kp_roll=gains_dict.get('Kp_roll', 0.5),
        Ki_roll=gains_dict.get('Ki_roll', 0.0),
        Kd_roll=gains_dict.get('Kd_roll', 0.2),
        Kp_pitch=gains_dict.get('Kp_pitch', 0.0),
        Ki_pitch=gains_dict.get('Ki_pitch', 0.0),
        Kd_pitch=gains_dict.get('Kd_pitch', 0.0),
        Kp_yaw=gains_dict.get('Kp_yaw', 0.0),
        Ki_yaw=gains_dict.get('Ki_yaw', 0.0),
        Kd_yaw=gains_dict.get('Kd_yaw', 0.0),
    )

    try:
        sim = RocketSimulator(rocket=rocket, motor=motor, wind=wind, controller=ctrl)
        result = sim.run(duration=12.0, dt=0.005, launch_pitch=launch_pitch)
        if not result:
            raise ValueError("empty result")

        pos = result['position']
        landing_dist = float(np.sqrt(pos[-1, 0]**2 + pos[-1, 1]**2))
        rms_burn = result.get('roll_rms_burn', result['roll_rms'])
        pitch_burn = result.get('pitch_rms_burn', 0.0)
        has_pitch_ctrl = gains_dict.get('Kp_pitch', 0.0) > 0.01

        ok = result['max_altitude'] > 20 and rms_burn < 15.0
        if has_pitch_ctrl:
            ok = ok and pitch_burn < 30.0

        return MCResult(
            run_id=run_id,
            max_altitude=result['max_altitude'],
            apogee_time=result['apogee_time'],
            flight_time=result['flight_time'],
            max_speed=result['max_speed'],
            roll_rms=result['roll_rms'],
            roll_rms_burn=rms_burn,
            landing_distance=landing_dist,
            success=ok,
            wind_speed=ws,
            launch_pitch=launch_pitch,
            mass=rocket.mass,
            thrust=motor.peak_thrust,
        )
    except Exception as e:
        return MCResult(
            run_id=run_id, max_altitude=0, apogee_time=0, flight_time=0,
            max_speed=0, roll_rms=999, roll_rms_burn=999,
            landing_distance=999, success=False,
            wind_speed=ws, launch_pitch=launch_pitch,
            mass=rocket.mass, thrust=motor.peak_thrust,
        )


class MonteCarloAnalysis:

    def __init__(self, gains=None, uncertainty=None, num_workers=None):
        g = dict(gains or {'Kp_roll': 0.5, 'Ki_roll': 0.0, 'Kd_roll': 0.2})
        self.gains = g
        self.uncertainty = uncertainty or UncertaintyParameters()
        self.num_workers = num_workers
        self.results: List[MCResult] = []
        self._last_elapsed = 0.0

    def run_analysis(self, num_runs=1000, base_seed=42):
        print(f"Monte Carlo: {num_runs} runs, workers={self.num_workers or 'serial'}",
              flush=True)
        unc_d = asdict(self.uncertainty)
        args_list = [
            (i, base_seed + i, self.gains, unc_d)
            for i in range(num_runs)
        ]

        t0 = time.time()

        if self.num_workers and self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                futs = {ex.submit(_run_one, a): a[0] for a in args_list}
                done = 0
                for f in as_completed(futs):
                    self.results.append(f.result())
                    done += 1
                    if done % max(1, num_runs // 10) == 0:
                        sr = sum(r.success for r in self.results) / len(self.results)
                        print(f"  [{done}/{num_runs}] success={sr*100:.1f}%", flush=True)
        else:
            for i, a in enumerate(args_list):
                self.results.append(_run_one(a))
                if (i + 1) % max(1, num_runs // 10) == 0:
                    sr = sum(r.success for r in self.results) / len(self.results)
                    print(f"  [{i+1}/{num_runs}] success={sr*100:.1f}%", flush=True)

        elapsed = time.time() - t0
        self._last_elapsed = elapsed
        return self.compute_statistics(elapsed)

    def compute_statistics(self, elapsed=0.0):
        if not self.results:
            return {}
        ok = [r for r in self.results if r.success]

        def s(vals):
            a = np.array(vals)
            return {
                'mean': float(np.mean(a)), 'std': float(np.std(a)),
                'min': float(np.min(a)), 'max': float(np.max(a)),
                'p5': float(np.percentile(a, 5)),
                'p50': float(np.percentile(a, 50)),
                'p95': float(np.percentile(a, 95)),
            }

        summary = {
            'total_runs': len(self.results),
            'successful_runs': len(ok),
            'success_rate': len(ok) / len(self.results),
            'elapsed_seconds': elapsed,
        }
        if ok:
            summary.update({
                'max_altitude': s([r.max_altitude for r in ok]),
                'flight_time': s([r.flight_time for r in ok]),
                'max_speed': s([r.max_speed for r in ok]),
                'roll_rms': s([r.roll_rms for r in ok]),
                'roll_rms_burn': s([r.roll_rms_burn for r in ok]),
                'landing_distance': s([r.landing_distance for r in ok]),
            })
        return summary

    def print_summary(self, stats=None):
        stats = stats or self.compute_statistics()
        print("\n" + "=" * 60, flush=True)
        print("MONTE CARLO SUMMARY", flush=True)
        print("=" * 60, flush=True)
        print(f"Total: {stats['total_runs']}  "
              f"Success: {stats['successful_runs']} "
              f"({stats['success_rate']*100:.1f}%)  "
              f"Time: {stats.get('elapsed_seconds', 0):.1f}s", flush=True)

        for key in ['max_altitude', 'roll_rms_burn', 'roll_rms',
                     'max_speed', 'flight_time', 'landing_distance']:
            d = stats.get(key)
            if not d:
                continue
            unit = {'max_altitude': 'm', 'landing_distance': 'm',
                    'flight_time': 's', 'max_speed': 'm/s'}.get(key, '°')
            print(f"\n  {key}:", flush=True)
            print(f"    mean={d['mean']:.2f}{unit}  std={d['std']:.2f}  "
                  f"90%CI=[{d['p5']:.2f}, {d['p95']:.2f}]", flush=True)

    def save_results(self, path):
        data = {
            'summary': self.compute_statistics(self._last_elapsed),
            'gains': self.gains,
            'uncertainty': asdict(self.uncertainty),
            'runs': [asdict(r) for r in self.results],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=float)


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Analysis')
    parser.add_argument('--runs', type=int, default=500)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gains-file', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='../results')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.gains_file and Path(args.gains_file).exists():
        with open(args.gains_file) as f:
            data = json.load(f)
            gains = data.get('gains', data)
    else:
        gains = {'Kp_roll': 0.5, 'Ki_roll': 0.0, 'Kd_roll': 0.2}

    print("=" * 60, flush=True)
    print("Monte Carlo Trajectory Analysis", flush=True)
    print("=" * 60, flush=True)
    print(f"Gains: {gains}", flush=True)

    mc = MonteCarloAnalysis(gains=gains, num_workers=args.workers)
    stats = mc.run_analysis(num_runs=args.runs, base_seed=args.seed)
    mc.print_summary(stats)

    out = Path(args.output_dir) / 'monte_carlo_results.json'
    mc.save_results(str(out))
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
