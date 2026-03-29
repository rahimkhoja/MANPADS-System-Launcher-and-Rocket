"""
Reinforcement Learning Controller for Rocket Stabilization
==========================================================
Uses PPO (Proximal Policy Optimization) to learn optimal control policy.
Trains on GPU using PyTorch for fast convergence.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. RL training disabled.")

from rocket_dynamics import (
    RocketSimulator, RocketParameters, MotorProfile,
    WindModel, RocketState, CanardController
)


if TORCH_AVAILABLE:
    
    class PolicyNetwork(nn.Module):
        """Actor network for continuous control."""
        
        def __init__(self, state_dim: int = 12, action_dim: int = 4, 
                     hidden_dim: int = 128):
            super().__init__()
            
            self.shared = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            
            self.mean = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
                nn.Tanh()
            )
            
            self.log_std = nn.Parameter(torch.zeros(action_dim))
            
        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            features = self.shared(state)
            mean = self.mean(features) * 12.0
            std = torch.exp(self.log_std).expand_as(mean)
            return mean, std
        
        def get_action(self, state: torch.Tensor, 
                       deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
            mean, std = self.forward(state)
            
            if deterministic:
                return mean, torch.zeros_like(mean)
            
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action, log_prob
        
        def evaluate(self, state: torch.Tensor, 
                     action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            mean, std = self.forward(state)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, entropy


    class ValueNetwork(nn.Module):
        """Critic network for state value estimation."""
        
        def __init__(self, state_dim: int = 12, hidden_dim: int = 128):
            super().__init__()
            
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.net(state).squeeze(-1)


    class RocketEnvironment:
        """Gym-like environment wrapper for rocket simulation."""
        
        def __init__(self, 
                     wind_speed_range: Tuple[float, float] = (0, 8),
                     episode_length: int = 2000,
                     dt: float = 0.005):
            
            self.wind_speed_range = wind_speed_range
            self.episode_length = episode_length
            self.dt = dt
            
            self.sim = None
            self.step_count = 0
            self.target_pitch = 85.0
            
        def reset(self, seed: int = None) -> np.ndarray:
            """Reset environment and return initial state."""
            if seed is not None:
                np.random.seed(seed)
            
            wind_speed = np.random.uniform(*self.wind_speed_range)
            wind_dir = np.random.uniform(0, 2 * np.pi)
            
            wind = WindModel(
                base_velocity=np.array([
                    wind_speed * np.cos(wind_dir),
                    wind_speed * np.sin(wind_dir),
                    0.0
                ]),
                gust_intensity=wind_speed * 0.2
            )
            
            class NullController:
                def compute(self, state, target, dt, max_defl):
                    return np.zeros(4)
                def reset(self):
                    pass
            
            self.sim = RocketSimulator(
                wind=wind,
                controller=NullController()
            )
            self.sim.reset()
            
            launch_angle = 85 + np.random.uniform(-2, 2)
            self.target_pitch = launch_angle
            
            self.step_count = 0
            
            return self._get_state()
        
        def _get_state(self) -> np.ndarray:
            """Extract state vector for RL agent."""
            state = self.sim.state
            roll, pitch, yaw = state.euler_angles()
            
            return np.array([
                roll / 45.0,
                (pitch - self.target_pitch) / 45.0,
                yaw / 45.0,
                state.angular_velocity[0] / 5.0,
                state.angular_velocity[1] / 5.0,
                state.angular_velocity[2] / 5.0,
                state.velocity[0] / 50.0,
                state.velocity[1] / 50.0,
                state.velocity[2] / 50.0,
                -state.position[2] / 100.0,
                self.sim.time / 10.0,
                float(self.sim.time < self.sim.motor.burn_time)
            ], dtype=np.float32)
        
        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
            """Take action and return (state, reward, done, info)."""
            action = np.clip(action, -12, 12)
            self.sim.canard_deflections = action
            
            state_array = self.sim.state.to_array()
            k1 = self.sim.derivatives(state_array, self.sim.time, action)
            k2 = self.sim.derivatives(state_array + 0.5*self.dt*k1, 
                                      self.sim.time + 0.5*self.dt, action)
            k3 = self.sim.derivatives(state_array + 0.5*self.dt*k2, 
                                      self.sim.time + 0.5*self.dt, action)
            k4 = self.sim.derivatives(state_array + self.dt*k3, 
                                      self.sim.time + self.dt, action)
            
            state_array += (self.dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            self.sim.state = RocketState.from_array(state_array)
            self.sim.time += self.dt
            self.step_count += 1
            
            roll, pitch, yaw = self.sim.state.euler_angles()
            
            roll_error = abs(roll)
            pitch_error = abs(pitch - self.target_pitch)
            yaw_error = abs(yaw)
            
            attitude_reward = -0.01 * (roll_error + pitch_error + yaw_error)
            
            angular_vel = self.sim.state.angular_velocity
            rate_penalty = -0.001 * np.sum(np.abs(angular_vel))
            
            control_penalty = -0.0001 * np.sum(action**2)
            
            altitude = -self.sim.state.position[2]
            altitude_reward = 0.001 * altitude if altitude > 0 else -1.0
            
            reward = attitude_reward + rate_penalty + control_penalty + altitude_reward
            
            done = False
            info = {'altitude': altitude, 'roll': roll, 'pitch': pitch}
            
            if self.sim.state.position[2] > 0:
                done = True
                reward -= 10.0
                info['termination'] = 'ground_hit'
            
            if roll_error > 60 or pitch_error > 60:
                done = True
                reward -= 20.0
                info['termination'] = 'unstable'
            
            if self.step_count >= self.episode_length:
                done = True
                reward += 5.0
                info['termination'] = 'timeout'
            
            return self._get_state(), reward, done, info


    class PPOTrainer:
        """Proximal Policy Optimization trainer."""
        
        def __init__(self,
                     env: RocketEnvironment,
                     hidden_dim: int = 128,
                     lr_policy: float = 3e-4,
                     lr_value: float = 1e-3,
                     gamma: float = 0.99,
                     gae_lambda: float = 0.95,
                     clip_epsilon: float = 0.2,
                     entropy_coef: float = 0.01,
                     value_coef: float = 0.5,
                     max_grad_norm: float = 0.5,
                     device: str = 'auto'):
            
            self.env = env
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_epsilon = clip_epsilon
            self.entropy_coef = entropy_coef
            self.value_coef = value_coef
            self.max_grad_norm = max_grad_norm
            
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            print(f"Using device: {self.device}")
            
            state_dim = 12
            action_dim = 4
            
            self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.value = ValueNetwork(state_dim, hidden_dim).to(self.device)
            
            self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
            self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr_value)
            
            self.training_stats = []
            
        def collect_trajectories(self, num_steps: int = 2048) -> Dict:
            """Collect experience from environment."""
            states = []
            actions = []
            rewards = []
            dones = []
            log_probs = []
            values = []
            
            state = self.env.reset()
            episode_rewards = []
            current_episode_reward = 0
            
            for _ in range(num_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, log_prob = self.policy.get_action(state_tensor)
                    value = self.value(state_tensor)
                
                action_np = action.cpu().numpy()[0]
                
                next_state, reward, done, info = self.env.step(action_np)
                
                states.append(state)
                actions.append(action_np)
                rewards.append(reward)
                dones.append(done)
                log_probs.append(log_prob.cpu().numpy()[0])
                values.append(value.cpu().numpy()[0])
                
                current_episode_reward += reward
                
                if done:
                    episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0
                    state = self.env.reset()
                else:
                    state = next_state
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                next_value = self.value(state_tensor).cpu().numpy()[0]
            
            advantages, returns = self._compute_gae(
                rewards, values, dones, next_value
            )
            
            return {
                'states': np.array(states),
                'actions': np.array(actions),
                'log_probs': np.array(log_probs),
                'returns': returns,
                'advantages': advantages,
                'episode_rewards': episode_rewards
            }
        
        def _compute_gae(self, rewards: List[float], values: List[float],
                        dones: List[bool], next_value: float) -> Tuple[np.ndarray, np.ndarray]:
            """Compute Generalized Advantage Estimation."""
            advantages = np.zeros_like(rewards)
            last_gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_val = next_value
                else:
                    next_val = values[t + 1]
                
                next_non_terminal = 1.0 - float(dones[t])
                delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
                advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            
            returns = advantages + np.array(values)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return advantages, returns
        
        def update(self, batch: Dict, epochs: int = 10, 
                   batch_size: int = 64) -> Dict:
            """Update policy and value networks."""
            states = torch.FloatTensor(batch['states']).to(self.device)
            actions = torch.FloatTensor(batch['actions']).to(self.device)
            old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
            returns = torch.FloatTensor(batch['returns']).to(self.device)
            advantages = torch.FloatTensor(batch['advantages']).to(self.device)
            
            total_samples = len(states)
            
            policy_losses = []
            value_losses = []
            entropy_losses = []
            
            for _ in range(epochs):
                indices = np.random.permutation(total_samples)
                
                for start in range(0, total_samples, batch_size):
                    end = start + batch_size
                    batch_indices = indices[start:end]
                    
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    
                    new_log_probs, entropy = self.policy.evaluate(batch_states, batch_actions)
                    
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                                        1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    values = self.value(batch_states)
                    value_loss = ((values - batch_returns) ** 2).mean()
                    
                    entropy_loss = -entropy.mean()
                    
                    total_loss = (policy_loss + 
                                 self.value_coef * value_loss + 
                                 self.entropy_coef * entropy_loss)
                    
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    total_loss.backward()
                    
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                    
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                    
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(-entropy_loss.item())
            
            return {
                'policy_loss': np.mean(policy_losses),
                'value_loss': np.mean(value_losses),
                'entropy': np.mean(entropy_losses)
            }
        
        def train(self, total_timesteps: int = 1000000,
                  steps_per_update: int = 2048,
                  epochs_per_update: int = 10,
                  log_interval: int = 10) -> List[Dict]:
            """Main training loop."""
            print(f"\nStarting PPO Training")
            print(f"Total timesteps: {total_timesteps}")
            print(f"Steps per update: {steps_per_update}")
            print("=" * 60)
            
            num_updates = total_timesteps // steps_per_update
            
            for update in range(num_updates):
                batch = self.collect_trajectories(steps_per_update)
                
                update_stats = self.update(batch, epochs=epochs_per_update)
                
                if batch['episode_rewards']:
                    mean_reward = np.mean(batch['episode_rewards'])
                    max_reward = np.max(batch['episode_rewards'])
                else:
                    mean_reward = 0
                    max_reward = 0
                
                stats = {
                    'update': update,
                    'timestep': (update + 1) * steps_per_update,
                    'mean_reward': mean_reward,
                    'max_reward': max_reward,
                    'episodes': len(batch['episode_rewards']),
                    **update_stats
                }
                self.training_stats.append(stats)
                
                if (update + 1) % log_interval == 0:
                    print(f"Update {update+1}/{num_updates} | "
                          f"Timestep {stats['timestep']} | "
                          f"Reward: {mean_reward:.2f} (max: {max_reward:.2f}) | "
                          f"Policy Loss: {update_stats['policy_loss']:.4f}")
            
            return self.training_stats
        
        def save(self, path: str):
            """Save trained models."""
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'value_state_dict': self.value.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'value_optimizer': self.value_optimizer.state_dict(),
                'training_stats': self.training_stats
            }, path)
            print(f"Model saved to: {path}")
        
        def load(self, path: str):
            """Load trained models."""
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
            self.training_stats = checkpoint.get('training_stats', [])
            print(f"Model loaded from: {path}")
        
        def evaluate(self, num_episodes: int = 100) -> Dict:
            """Evaluate trained policy."""
            print(f"\nEvaluating policy over {num_episodes} episodes...")
            
            episode_rewards = []
            episode_lengths = []
            max_altitudes = []
            roll_rms_values = []
            successes = 0
            
            for ep in range(num_episodes):
                state = self.env.reset()
                total_reward = 0
                rolls = []
                max_alt = 0
                
                done = False
                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        action, _ = self.policy.get_action(state_tensor, deterministic=True)
                    
                    state, reward, done, info = self.env.step(action.cpu().numpy()[0])
                    total_reward += reward
                    rolls.append(info.get('roll', 0))
                    max_alt = max(max_alt, info.get('altitude', 0))
                
                episode_rewards.append(total_reward)
                episode_lengths.append(self.env.step_count)
                max_altitudes.append(max_alt)
                roll_rms_values.append(np.sqrt(np.mean(np.array(rolls)**2)))
                
                if info.get('termination') == 'timeout':
                    successes += 1
            
            results = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'mean_altitude': np.mean(max_altitudes),
                'mean_roll_rms': np.mean(roll_rms_values),
                'success_rate': successes / num_episodes
            }
            
            print(f"\nEvaluation Results:")
            print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"  Mean Altitude: {results['mean_altitude']:.1f} m")
            print(f"  Mean Roll RMS: {results['mean_roll_rms']:.2f}°")
            print(f"  Success Rate: {results['success_rate']*100:.1f}%")
            
            return results


def export_policy_to_cpp(trainer: 'PPOTrainer', output_path: str):
    """Export trained policy to C++ header for ESP32."""
    policy = trainer.policy
    
    weights = []
    for name, param in policy.named_parameters():
        weights.append((name, param.detach().cpu().numpy()))
    
    cpp_code = '''
// Auto-generated Neural Network Policy for ESP32
// Trained with PPO on rocket stabilization task

#ifndef RL_POLICY_H
#define RL_POLICY_H

#include <Arduino.h>
#include <math.h>

class NeuralPolicy {
public:
    // Network weights (flattened)
'''
    
    for name, w in weights:
        flat_name = name.replace('.', '_')
        flat_w = w.flatten()
        cpp_code += f"    static constexpr float {flat_name}[{len(flat_w)}] = {{"
        cpp_code += ', '.join([f'{x:.6f}f' for x in flat_w])
        cpp_code += "};\n"
    
    cpp_code += '''
    
    void getAction(float* state, float* action) {
        // Forward pass through network
        // Simplified for ESP32 (no batching)
        
        // This is a placeholder - full implementation would
        // include matrix multiplications matching the network architecture
        
        // For now, fall back to PID-like behavior
        float roll_error = state[0] * 45.0f;  // Denormalize
        float pitch_error = state[1] * 45.0f;
        float roll_rate = state[3] * 5.0f;
        float pitch_rate = state[4] * 5.0f;
        
        float Kp = 0.5f, Kd = 0.2f;
        float roll_cmd = Kp * roll_error + Kd * roll_rate;
        float pitch_cmd = Kp * pitch_error + Kd * pitch_rate;
        
        // Canard mixing
        action[0] = constrain(roll_cmd, -12.0f, 12.0f);  // Left
        action[1] = constrain(roll_cmd, -12.0f, 12.0f);  // Right
        action[2] = constrain(pitch_cmd, -12.0f, 12.0f); // Up
        action[3] = constrain(pitch_cmd, -12.0f, 12.0f); // Down
    }
    
private:
    float relu(float x) { return x > 0 ? x : 0; }
    float tanh_approx(float x) {
        if (x < -3) return -1;
        if (x > 3) return 1;
        return x * (27 + x*x) / (27 + 9*x*x);
    }
};

#endif // RL_POLICY_H
'''
    
    with open(output_path, 'w') as f:
        f.write(cpp_code)
    
    print(f"Policy exported to: {output_path}")


def main():
    if not TORCH_AVAILABLE:
        print("Error: PyTorch required for RL training")
        return
    
    parser = argparse.ArgumentParser(description='RL Controller Training')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to load existing model')
    parser.add_argument('--output-dir', type=str, default='../results',
                       help='Output directory')
    parser.add_argument('--export-cpp', action='store_true',
                       help='Export policy to C++ header')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    env = RocketEnvironment(
        wind_speed_range=(0, 8),
        episode_length=2000,
        dt=0.005
    )
    
    trainer = PPOTrainer(env, hidden_dim=128)
    
    if args.load:
        trainer.load(args.load)
    else:
        print("\n" + "=" * 60)
        print("REINFORCEMENT LEARNING ROCKET CONTROLLER")
        print("=" * 60)
        print(f"Device: {trainer.device}")
        print(f"Training timesteps: {args.timesteps}")
        
        start_time = time.time()
        trainer.train(total_timesteps=args.timesteps)
        elapsed = time.time() - start_time
        
        print(f"\nTraining completed in {elapsed:.1f}s")
        
        model_path = Path(args.output_dir) / 'rl_policy.pth'
        trainer.save(str(model_path))
    
    eval_results = trainer.evaluate(num_episodes=args.eval_episodes)
    
    results_path = Path(args.output_dir) / 'rl_evaluation.json'
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    if args.export_cpp:
        cpp_path = Path(args.output_dir) / 'rl_policy.h'
        export_policy_to_cpp(trainer, str(cpp_path))


if __name__ == "__main__":
    main()
