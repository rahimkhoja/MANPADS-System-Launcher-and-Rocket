"""
GPU-Accelerated PID Gain Optimizer
==================================
Uses parallel simulation on GPU to find optimal control gains.
Designed for SLURM cluster execution with multiple GPUs.
"""

import numpy as np
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import itertools

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

from rocket_dynamics import (
    RocketSimulator, CanardController, WindModel, 
    RocketParameters, MotorProfile, evaluate_controller
)

# Module-level function for multiprocessing pickling
def _eval_gains_worker(args):
    """Worker function for parallel gain evaluation."""
    gains_tuple, wind_speeds, num_runs = args
    param_names, values = gains_tuple
    gains = dict(zip(param_names, values))
    
    total_score = 0
    for wind in wind_speeds:
        results = evaluate_controller(gains, wind_speed=wind, num_runs=num_runs)
        roll_penalty = results['roll_rms'] * 2.0
        pitch_penalty = results['pitch_rms'] * 1.5
        altitude_bonus = -results['max_altitude'] * 0.1
        stability_bonus = -results['stability_score'] * 0.5
        total_score += roll_penalty + pitch_penalty + altitude_bonus + stability_bonus
    
    return total_score / len(wind_speeds)


class PIDOptimizer:
    """Optimize PID gains using grid search or evolutionary algorithms."""
    
    def __init__(self, 
                 wind_speeds: List[float] = [0, 2, 5],
                 num_runs_per_config: int = 3,
                 num_workers: int = None):
        self.wind_speeds = wind_speeds
        self.num_runs = num_runs_per_config
        self.num_workers = num_workers
        self.results_history = []
        
    def objective_function(self, gains: dict) -> float:
        """
        Compute objective function for a set of gains.
        Lower is better.
        """
        total_score = 0
        
        for wind in self.wind_speeds:
            results = evaluate_controller(
                gains, wind_speed=wind, num_runs=self.num_runs
            )
            roll_penalty = results['roll_rms'] * 2.0
            pitch_penalty = results['pitch_rms'] * 1.5
            altitude_bonus = -results['max_altitude'] * 0.1
            stability_bonus = -results['stability_score'] * 0.5
            
            total_score += roll_penalty + pitch_penalty + altitude_bonus + stability_bonus
        
        return total_score / len(self.wind_speeds)
    
    def grid_search(self, param_ranges: dict, 
                    progress_callback=None) -> Tuple[dict, float]:
        """
        Exhaustive grid search over parameter space.
        
        param_ranges: {
            'Kp_roll': [0.3, 0.5, 0.7, 1.0],
            'Kd_roll': [0.1, 0.2, 0.3],
            ...
        }
        """
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        all_combos = list(itertools.product(*param_values))
        total_combos = len(all_combos)
        
        print(f"Grid search: {total_combos} combinations")
        print(f"Parameters: {param_names}")
        
        best_gains = None
        best_score = float('inf')
        
        for i, combo in enumerate(all_combos):
            gains = dict(zip(param_names, combo))
            score = self.objective_function(gains)
            
            self.results_history.append({
                'gains': gains.copy(),
                'score': score
            })
            
            if score < best_score:
                best_score = score
                best_gains = gains.copy()
                print(f"  [{i+1}/{total_combos}] New best: {score:.3f} with {gains}")
            
            if progress_callback:
                progress_callback(i + 1, total_combos)
        
        return best_gains, best_score
    
    def parallel_grid_search(self, param_ranges: dict) -> Tuple[dict, float]:
        """Grid search with parallel evaluation."""
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        all_combos = list(itertools.product(*param_values))
        
        print(f"Parallel grid search: {len(all_combos)} combinations")
        
        def eval_combo(combo):
            gains = dict(zip(param_names, combo))
            score = self.objective_function(gains)
            return gains, score
        
        best_gains = None
        best_score = float('inf')
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(eval_combo, c): c for c in all_combos}
            
            for i, future in enumerate(as_completed(futures)):
                gains, score = future.result()
                self.results_history.append({'gains': gains, 'score': score})
                
                if score < best_score:
                    best_score = score
                    best_gains = gains
                    print(f"  [{i+1}/{len(all_combos)}] New best: {score:.3f}")
        
        return best_gains, best_score
    
    def differential_evolution(self, param_bounds: dict,
                               population_size: int = 50,
                               generations: int = 100,
                               mutation_factor: float = 0.8,
                               crossover_prob: float = 0.7) -> Tuple[dict, float]:
        """
        Differential Evolution optimization with parallel evaluation.
        More efficient for continuous parameter spaces.
        
        param_bounds: {
            'Kp_roll': (0.1, 2.0),
            'Kd_roll': (0.05, 1.0),
            ...
        }
        """
        param_names = list(param_bounds.keys())
        bounds = np.array(list(param_bounds.values()))
        n_params = len(param_names)
        
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1], 
            size=(population_size, n_params)
        )
        
        # Parallel initial fitness evaluation
        print(f"DE Optimization: {generations} generations, pop={population_size}", flush=True)
        print(f"Evaluating initial population...", flush=True)
        
        def make_args(ind):
            return ((param_names, tuple(ind)), self.wind_speeds, self.num_runs)
        
        if self.num_workers and self.num_workers > 1:
            args_list = [make_args(ind) for ind in population]
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                fitness = np.array(list(executor.map(_eval_gains_worker, args_list)))
        else:
            fitness = np.array([self.objective_function(dict(zip(param_names, ind))) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        print(f"Initial best: {best_fitness:.3f}", flush=True)
        
        for gen in range(generations):
            # Generate all trial vectors first
            trials = []
            trial_indices = []
            
            for i in range(population_size):
                candidates = [j for j in range(population_size) if j != i]
                a, b, c = population[np.random.choice(candidates, 3, replace=False)]
                
                mutant = a + mutation_factor * (b - c)
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
                
                crossover = np.random.rand(n_params) < crossover_prob
                if not np.any(crossover):
                    crossover[np.random.randint(n_params)] = True
                
                trial = np.where(crossover, mutant, population[i])
                trials.append(trial)
                trial_indices.append(i)
            
            # Parallel trial evaluation
            if self.num_workers and self.num_workers > 1:
                args_list = [make_args(t) for t in trials]
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    trial_fitness_list = list(executor.map(_eval_gains_worker, args_list))
            else:
                trial_fitness_list = [self.objective_function(dict(zip(param_names, t))) for t in trials]
            
            # Update population with better individuals
            for i, (trial, trial_fitness) in enumerate(zip(trials, trial_fitness_list)):
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
            
            if (gen + 1) % 5 == 0:
                print(f"  Gen {gen+1}: best={best_fitness:.3f}, "
                      f"mean={np.mean(fitness):.3f}", flush=True)
                
            self.results_history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitness),
                'best_gains': dict(zip(param_names, best_individual))
            })
        
        return dict(zip(param_names, best_individual)), best_fitness
    
    def save_results(self, filename: str):
        """Save optimization results to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=float)
    
    def load_results(self, filename: str):
        """Load previous results."""
        with open(filename, 'r') as f:
            self.results_history = json.load(f)


def run_optimization(args):
    """Main optimization routine."""
    print("=" * 60)
    print("PID Gain Optimizer for Canard-Stabilized Rocket")
    print("=" * 60)
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Wind speeds: {args.wind_speeds}")
    print(f"Runs per config: {args.num_runs}")
    
    optimizer = PIDOptimizer(
        wind_speeds=args.wind_speeds,
        num_runs_per_config=args.num_runs,
        num_workers=args.workers
    )
    
    start_time = time.time()
    
    if args.method == 'grid':
        param_ranges = {
            'Kp_roll': np.linspace(0.2, 1.5, args.grid_points).tolist(),
            'Kd_roll': np.linspace(0.1, 0.8, args.grid_points).tolist(),
            'Kp_pitch': np.linspace(0.2, 1.5, args.grid_points).tolist(),
            'Kd_pitch': np.linspace(0.1, 0.8, args.grid_points).tolist(),
        }
        
        if args.workers and args.workers > 1:
            best_gains, best_score = optimizer.parallel_grid_search(param_ranges)
        else:
            best_gains, best_score = optimizer.grid_search(param_ranges)
            
    elif args.method == 'de':
        param_bounds = {
            'Kp_roll': (0.1, 2.0),
            'Ki_roll': (0.0, 0.5),
            'Kd_roll': (0.05, 1.0),
            'Kp_pitch': (0.1, 2.0),
            'Ki_pitch': (0.0, 0.5),
            'Kd_pitch': (0.05, 1.0),
        }
        
        best_gains, best_score = optimizer.differential_evolution(
            param_bounds,
            population_size=args.population,
            generations=args.generations
        )
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Best Score: {best_score:.4f}")
    print(f"Best Gains:")
    for k, v in best_gains.items():
        print(f"  {k}: {v:.4f}")
    
    results_file = Path(args.output_dir) / f"optimization_results_{args.method}.json"
    optimizer.save_results(str(results_file))
    print(f"\nResults saved to: {results_file}")
    
    best_file = Path(args.output_dir) / "best_gains.json"
    with open(best_file, 'w') as f:
        json.dump({
            'gains': best_gains,
            'score': float(best_score),
            'method': args.method,
            'wind_speeds': args.wind_speeds,
            'elapsed_time': elapsed
        }, f, indent=2)
    print(f"Best gains saved to: {best_file}")
    
    return best_gains, best_score


def main():
    parser = argparse.ArgumentParser(description='PID Gain Optimizer')
    parser.add_argument('--method', choices=['grid', 'de'], default='de',
                       help='Optimization method (grid=grid search, de=differential evolution)')
    parser.add_argument('--wind-speeds', type=float, nargs='+', default=[0, 3, 6],
                       help='Wind speeds to test (m/s)')
    parser.add_argument('--num-runs', type=int, default=3,
                       help='Number of simulation runs per configuration')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--grid-points', type=int, default=5,
                       help='Points per dimension for grid search')
    parser.add_argument('--population', type=int, default=30,
                       help='Population size for DE')
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations for DE')
    parser.add_argument('--output-dir', type=str, default='../results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    run_optimization(args)


if __name__ == "__main__":
    main()
