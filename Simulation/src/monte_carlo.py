"""
Monte Carlo Trajectory Analysis
===============================
Statistical analysis of rocket flight performance under uncertainty.
Runs thousands of simulations with randomized parameters to assess reliability.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

from rocket_dynamics import (
    RocketSimulator, RocketParameters, MotorProfile, 
    WindModel, CanardController, AtmosphereModel
)


@dataclass
class UncertaintyParameters:
    """Define uncertainty ranges for Monte Carlo analysis."""
    
    mass_std: float = 0.005          # kg (±5g)
    cg_std: float = 0.003            # m (±3mm)
    Ixx_std_pct: float = 0.05        # 5% variation
    Iyy_std_pct: float = 0.05
    Cd0_std_pct: float = 0.10        # 10% drag uncertainty
    thrust_std_pct: float = 0.08     # 8% motor variation
    burn_time_std_pct: float = 0.03
    
    gyro_bias_std: float = 0.005     # rad/s (~0.3°/s, realistic for MPU6050)
    gyro_noise_std: float = 0.005    # rad/s
    accel_bias_std: float = 0.1      # m/s^2
    accel_noise_std: float = 0.3     # m/s^2
    
    launch_angle_std: float = 1.0    # degrees
    wind_speed_mean: float = 2.0     # m/s
    wind_speed_std: float = 1.0      # m/s
    wind_direction_std: float = 20   # degrees
    gust_intensity_pct: float = 0.2  # 20% of mean wind
    
    servo_offset_std: float = 1.0    # degrees center offset
    servo_rate_limit: float = 300    # deg/s


@dataclass
class MonteCarloResult:
    """Results from a single Monte Carlo run."""
    run_id: int
    max_altitude: float
    apogee_time: float
    flight_time: float
    max_speed: float
    roll_rms: float
    pitch_rms: float
    yaw_rms: float
    landing_distance: float
    stability_score: float
    success: bool
    parameters: dict


class MonteCarloAnalysis:
    """Run Monte Carlo trajectory analysis."""
    
    def __init__(self, 
                 base_gains: dict = None,
                 uncertainty: UncertaintyParameters = None,
                 num_workers: int = None):
        
        self.base_gains = base_gains or {
            'Kp_roll': 0.5, 'Ki_roll': 0.0, 'Kd_roll': 0.2,
            'Kp_pitch': 0.5, 'Ki_pitch': 0.0, 'Kd_pitch': 0.2,
            'Kp_yaw': 0.3, 'Ki_yaw': 0.0, 'Kd_yaw': 0.15
        }
        
        self.uncertainty = uncertainty or UncertaintyParameters()
        self.num_workers = num_workers
        self.results: List[MonteCarloResult] = []
        
    def sample_parameters(self, seed: int = None) -> Tuple[RocketParameters, 
                                                            MotorProfile, 
                                                            WindModel, 
                                                            dict]:
        """Sample random parameters from uncertainty distributions."""
        if seed is not None:
            np.random.seed(seed)
        
        u = self.uncertainty
        
        rocket = RocketParameters()
        rocket.mass = max(0.1, np.random.normal(rocket.mass, u.mass_std))
        rocket.cg_from_nose = np.random.normal(rocket.cg_from_nose, u.cg_std)
        rocket.Ixx *= np.random.normal(1.0, u.Ixx_std_pct)
        rocket.Iyy *= np.random.normal(1.0, u.Iyy_std_pct)
        rocket.Izz *= np.random.normal(1.0, u.Iyy_std_pct)
        rocket.Cd0 *= np.random.normal(1.0, u.Cd0_std_pct)
        
        motor = MotorProfile()
        motor.peak_thrust *= np.random.normal(1.0, u.thrust_std_pct)
        motor.burn_time *= np.random.normal(1.0, u.burn_time_std_pct)
        
        wind_speed = max(0, np.random.normal(u.wind_speed_mean, u.wind_speed_std))
        wind_dir = np.random.normal(0, u.wind_direction_std) * np.pi / 180
        
        wind = WindModel(
            base_velocity=np.array([
                wind_speed * np.cos(wind_dir),
                wind_speed * np.sin(wind_dir),
                0.0
            ]),
            gust_intensity=wind_speed * u.gust_intensity_pct
        )
        
        launch_pitch = 85 + np.random.normal(0, u.launch_angle_std)
        launch_yaw = np.random.normal(0, u.launch_angle_std)
        
        sensor_params = {
            'gyro_bias': np.random.normal(0, u.gyro_bias_std, 3),
            'accel_bias': np.random.normal(0, u.accel_bias_std, 3),
            'servo_offsets': np.random.normal(0, u.servo_offset_std, 4),
            'launch_pitch': launch_pitch,
            'launch_yaw': launch_yaw
        }
        
        return rocket, motor, wind, sensor_params
    
    def run_single(self, run_id: int, seed: int = None) -> MonteCarloResult:
        """Run a single Monte Carlo simulation."""
        rocket, motor, wind, sensor_params = self.sample_parameters(seed)
        
        controller = CanardController(**self.base_gains)
        
        sim = RocketSimulator(
            rocket=rocket,
            motor=motor,
            wind=wind,
            controller=controller
        )
        
        sim.reset()
        launch_pitch_rad = sensor_params['launch_pitch'] * np.pi / 180
        launch_yaw_rad = sensor_params['launch_yaw'] * np.pi / 180
        
        cr, sr = 1.0, 0.0
        cp, sp = np.cos(launch_pitch_rad/2), np.sin(launch_pitch_rad/2)
        cy, sy = np.cos(launch_yaw_rad/2), np.sin(launch_yaw_rad/2)
        
        sim.state.quaternion = np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy
        ])
        
        try:
            results = sim.run(duration=15.0, dt=0.005)
            
            landing_distance = np.sqrt(
                results['position'][-1, 0]**2 + 
                results['position'][-1, 1]**2
            )
            
            max_roll = np.max(np.abs(results['roll']))
            roll_rms = results['roll_rms']
            success = (
                results['max_altitude'] > 20 and
                roll_rms < 30  # Use RMS instead of max, 30° threshold
            )
            
            return MonteCarloResult(
                run_id=run_id,
                max_altitude=results['max_altitude'],
                apogee_time=results['apogee_time'],
                flight_time=results['flight_time'],
                max_speed=results['max_speed'],
                roll_rms=results['roll_rms'],
                pitch_rms=results['pitch_rms'],
                yaw_rms=np.sqrt(np.mean(results['yaw']**2)),
                landing_distance=landing_distance,
                stability_score=results['stability_score'],
                success=success,
                parameters={
                    'mass': rocket.mass,
                    'wind_speed': np.linalg.norm(wind.base_velocity),
                    'launch_pitch': sensor_params['launch_pitch'],
                    'thrust': motor.peak_thrust
                }
            )
            
        except Exception as e:
            return MonteCarloResult(
                run_id=run_id,
                max_altitude=0, apogee_time=0, flight_time=0, max_speed=0,
                roll_rms=999, pitch_rms=999, yaw_rms=999,
                landing_distance=999, stability_score=-999,
                success=False,
                parameters={'error': str(e)}
            )
    
    def run_analysis(self, num_runs: int = 1000, 
                     base_seed: int = None) -> Dict:
        """Run full Monte Carlo analysis."""
        print(f"Starting Monte Carlo Analysis: {num_runs} runs")
        print(f"Workers: {self.num_workers or 'sequential'}")
        
        start_time = time.time()
        
        if self.num_workers and self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                seeds = [base_seed + i if base_seed else None for i in range(num_runs)]
                futures = {
                    executor.submit(self.run_single, i, seeds[i]): i 
                    for i in range(num_runs)
                }
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    self.results.append(result)
                    
                    if (i + 1) % 100 == 0:
                        success_rate = sum(1 for r in self.results if r.success) / len(self.results)
                        print(f"  [{i+1}/{num_runs}] Success rate: {success_rate*100:.1f}%")
        else:
            for i in range(num_runs):
                seed = base_seed + i if base_seed else None
                result = self.run_single(i, seed)
                self.results.append(result)
                
                if (i + 1) % 100 == 0:
                    success_rate = sum(1 for r in self.results if r.success) / len(self.results)
                    print(f"  [{i+1}/{num_runs}] Success rate: {success_rate*100:.1f}%")
        
        elapsed = time.time() - start_time
        
        return self.compute_statistics(elapsed)
    
    def compute_statistics(self, elapsed_time: float = 0) -> Dict:
        """Compute statistical summary of results."""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r.success]
        
        def stats(values):
            arr = np.array(values)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'p5': float(np.percentile(arr, 5)),
                'p50': float(np.percentile(arr, 50)),
                'p95': float(np.percentile(arr, 95))
            }
        
        summary = {
            'total_runs': len(self.results),
            'successful_runs': len(successful),
            'success_rate': len(successful) / len(self.results),
            'elapsed_time': elapsed_time,
            
            'max_altitude': stats([r.max_altitude for r in successful]) if successful else {},
            'flight_time': stats([r.flight_time for r in successful]) if successful else {},
            'max_speed': stats([r.max_speed for r in successful]) if successful else {},
            'roll_rms': stats([r.roll_rms for r in successful]) if successful else {},
            'pitch_rms': stats([r.pitch_rms for r in successful]) if successful else {},
            'landing_distance': stats([r.landing_distance for r in successful]) if successful else {},
            'stability_score': stats([r.stability_score for r in successful]) if successful else {},
        }
        
        return summary
    
    def save_results(self, filename: str):
        """Save all results to JSON."""
        data = {
            'summary': self.compute_statistics(),
            'base_gains': self.base_gains,
            'uncertainty': asdict(self.uncertainty),
            'runs': [asdict(r) for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=float)
    
    def print_summary(self):
        """Print formatted summary."""
        stats = self.compute_statistics()
        
        print("\n" + "=" * 60)
        print("MONTE CARLO ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total Runs: {stats['total_runs']}")
        print(f"Successful: {stats['successful_runs']} ({stats['success_rate']*100:.1f}%)")
        print(f"Time: {stats['elapsed_time']:.1f}s")
        
        print("\n--- Performance Metrics (successful flights) ---")
        
        for metric in ['max_altitude', 'flight_time', 'max_speed', 
                       'roll_rms', 'landing_distance', 'stability_score']:
            if metric in stats and stats[metric]:
                s = stats[metric]
                unit = 'm' if metric in ['max_altitude', 'landing_distance'] else \
                       's' if metric == 'flight_time' else \
                       'm/s' if metric == 'max_speed' else \
                       '°' if 'rms' in metric else ''
                print(f"\n{metric}:")
                print(f"  Mean: {s['mean']:.2f} {unit}")
                print(f"  Std:  {s['std']:.2f} {unit}")
                print(f"  Range: [{s['min']:.2f}, {s['max']:.2f}] {unit}")
                print(f"  90% CI: [{s['p5']:.2f}, {s['p95']:.2f}] {unit}")


def run_sensitivity_analysis(base_gains: dict, num_runs: int = 500) -> Dict:
    """Analyze sensitivity to each uncertainty parameter."""
    print("Running Sensitivity Analysis")
    print("=" * 60)
    
    base_uncertainty = UncertaintyParameters()
    
    parameters_to_test = [
        ('mass_std', [0.005, 0.010, 0.020]),
        ('wind_speed_mean', [1.0, 3.0, 6.0, 10.0]),
        ('gyro_bias_std', [0.01, 0.02, 0.05]),
        ('launch_angle_std', [1.0, 2.0, 5.0]),
        ('thrust_std_pct', [0.05, 0.10, 0.20]),
    ]
    
    sensitivity_results = {}
    
    for param_name, values in parameters_to_test:
        print(f"\nTesting {param_name}...")
        param_results = []
        
        for value in values:
            uncertainty = UncertaintyParameters()
            setattr(uncertainty, param_name, value)
            
            mc = MonteCarloAnalysis(
                base_gains=base_gains,
                uncertainty=uncertainty,
                num_workers=4
            )
            
            stats = mc.run_analysis(num_runs=num_runs // len(values), base_seed=42)
            param_results.append({
                'value': value,
                'success_rate': stats['success_rate'],
                'roll_rms_mean': stats['roll_rms'].get('mean', 999) if stats['roll_rms'] else 999,
                'altitude_mean': stats['max_altitude'].get('mean', 0) if stats['max_altitude'] else 0
            })
        
        sensitivity_results[param_name] = param_results
    
    return sensitivity_results


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Trajectory Analysis')
    parser.add_argument('--runs', type=int, default=1000,
                       help='Number of Monte Carlo runs')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed')
    parser.add_argument('--gains-file', type=str, default=None,
                       help='JSON file with optimized gains')
    parser.add_argument('--output-dir', type=str, default='../results',
                       help='Output directory')
    parser.add_argument('--sensitivity', action='store_true',
                       help='Run sensitivity analysis')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.gains_file:
        with open(args.gains_file, 'r') as f:
            data = json.load(f)
            gains = data.get('gains', data)
    else:
        gains = {
            'Kp_roll': 0.5, 'Ki_roll': 0.0, 'Kd_roll': 0.2,
            'Kp_pitch': 0.5, 'Ki_pitch': 0.0, 'Kd_pitch': 0.2,
            'Kp_yaw': 0.3, 'Ki_yaw': 0.0, 'Kd_yaw': 0.15
        }
    
    print("Controller Gains:")
    for k, v in gains.items():
        print(f"  {k}: {v}")
    
    if args.sensitivity:
        results = run_sensitivity_analysis(gains, num_runs=args.runs)
        output_file = Path(args.output_dir) / 'sensitivity_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSensitivity results saved to: {output_file}")
    else:
        mc = MonteCarloAnalysis(
            base_gains=gains,
            num_workers=args.workers
        )
        
        mc.run_analysis(num_runs=args.runs, base_seed=args.seed)
        mc.print_summary()
        
        output_file = Path(args.output_dir) / 'monte_carlo_results.json'
        mc.save_results(str(output_file))
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
