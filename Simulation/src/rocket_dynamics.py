"""
6-DOF Rocket Flight Dynamics Simulation
========================================
High-fidelity simulation for canard-stabilized rocket with folding fins.
Supports GPU acceleration via CuPy for parameter optimization.

Based on specifications from the MANPADS prototype:
- Mass: ~200g (estimated from $96 BOM)
- 4 canard control surfaces
- MPU6050 IMU at 200Hz
- ESP32 flight computer
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable
import json

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False


@dataclass
class RocketParameters:
    """Physical parameters of the rocket."""
    mass: float = 0.200  # kg (200g)
    length: float = 0.30  # m (30cm)
    diameter: float = 0.04  # m (40mm tube)
    
    Ixx: float = 0.00005  # kg*m^2 (roll moment of inertia)
    Iyy: float = 0.0015   # kg*m^2 (pitch moment of inertia)
    Izz: float = 0.0015   # kg*m^2 (yaw moment of inertia)
    
    cg_from_nose: float = 0.18  # m (center of gravity from nose)
    cp_from_nose: float = 0.20  # m (center of pressure from nose)
    
    canard_area: float = 0.0004  # m^2 per canard (20mm x 20mm)
    canard_arm: float = 0.05     # m (distance from CG to canard)
    canard_max_deflection: float = 12.0  # degrees
    
    fin_area: float = 0.001  # m^2 per fin
    reference_area: float = 0.00126  # m^2 (pi * r^2)
    
    Cd0: float = 0.4      # Zero-lift drag coefficient
    Cd_canard: float = 0.1  # Additional drag per degree canard deflection
    Cla: float = 2.5      # Lift curve slope (per radian)
    Cma: float = -3.0     # Pitch moment coefficient (negative = stable)
    Clp: float = -0.5     # Roll damping coefficient
    Cmq: float = -2.0     # Pitch damping coefficient
    Cnr: float = -2.0     # Yaw damping coefficient


@dataclass
class MotorProfile:
    """Solid rocket motor thrust profile."""
    burn_time: float = 1.2  # seconds
    total_impulse: float = 10.0  # N*s (estimated C-class motor)
    peak_thrust: float = 15.0  # N
    propellant_mass: float = 0.020  # kg
    
    def thrust(self, t: float) -> float:
        """Thrust as function of time (fast ignition profile)."""
        if t < 0 or t > self.burn_time:
            return 0.0
        ramp_time = 0.02  # 20ms ignition ramp (realistic for solid motors)
        if t < ramp_time:
            return self.peak_thrust * (t / ramp_time)
        elif t > self.burn_time - 0.05:
            return self.peak_thrust * ((self.burn_time - t) / 0.05)
        return self.peak_thrust
    
    def mass_flow(self, t: float) -> float:
        """Propellant mass flow rate."""
        if 0 <= t <= self.burn_time:
            return self.propellant_mass / self.burn_time
        return 0.0


@dataclass
class AtmosphereModel:
    """Simple atmosphere model for low-altitude flights."""
    sea_level_density: float = 1.225  # kg/m^3
    sea_level_pressure: float = 101325  # Pa
    temperature_lapse: float = 0.0065  # K/m
    sea_level_temp: float = 288.15  # K
    
    def density(self, altitude: float) -> float:
        """Air density at altitude (valid up to ~11km)."""
        if altitude < 0:
            altitude = 0
        T = self.sea_level_temp - self.temperature_lapse * altitude
        return self.sea_level_density * (T / self.sea_level_temp) ** 4.256
    
    def speed_of_sound(self, altitude: float) -> float:
        """Speed of sound at altitude."""
        T = self.sea_level_temp - self.temperature_lapse * altitude
        return 20.05 * np.sqrt(T)


@dataclass
class WindModel:
    """Wind disturbance model."""
    base_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    gust_intensity: float = 0.0  # m/s RMS
    turbulence_scale: float = 50.0  # m (length scale)
    
    def velocity(self, position: np.ndarray, t: float) -> np.ndarray:
        """Wind velocity at position and time (NED frame)."""
        if self.gust_intensity == 0:
            return self.base_velocity.copy()
        np.random.seed(int(t * 1000) % 2**31)
        gust = np.random.randn(3) * self.gust_intensity
        return self.base_velocity + gust


@dataclass
class RocketState:
    """Complete state vector for 6-DOF simulation."""
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    mass: float = 0.220  # kg (including propellant)
    
    def euler_angles(self) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees."""
        q0, q1, q2, q3 = self.quaternion
        roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
        pitch = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1, 1))
        yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    
    def rotation_matrix(self) -> np.ndarray:
        """Body-to-inertial rotation matrix from quaternion."""
        q0, q1, q2, q3 = self.quaternion
        return np.array([
            [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
        ])
    
    def to_array(self) -> np.ndarray:
        """Flatten state to array for integration."""
        return np.concatenate([
            self.position, self.velocity, self.quaternion, 
            self.angular_velocity, [self.mass]
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'RocketState':
        """Reconstruct state from array."""
        state = cls()
        state.position = arr[0:3].copy()
        state.velocity = arr[3:6].copy()
        state.quaternion = arr[6:10].copy()
        state.quaternion /= np.linalg.norm(state.quaternion)
        state.angular_velocity = arr[10:13].copy()
        state.mass = arr[13]
        return state


class CanardController:
    """PID controller for canard-stabilized rocket."""
    
    def __init__(self, Kp_roll=0.5, Ki_roll=0.0, Kd_roll=0.2,
                 Kp_pitch=0.5, Ki_pitch=0.0, Kd_pitch=0.2,
                 Kp_yaw=0.5, Ki_yaw=0.0, Kd_yaw=0.2):
        self.gains = {
            'roll': {'Kp': Kp_roll, 'Ki': Ki_roll, 'Kd': Kd_roll},
            'pitch': {'Kp': Kp_pitch, 'Ki': Ki_pitch, 'Kd': Kd_pitch},
            'yaw': {'Kp': Kp_yaw, 'Ki': Ki_yaw, 'Kd': Kd_yaw}
        }
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)
        self.integral_limit = 10.0
        
    def compute(self, state: RocketState, target_attitude: np.ndarray, 
                dt: float, max_deflection: float = 12.0) -> np.ndarray:
        """
        Compute canard deflections [left, right, up, down] in degrees.
        target_attitude: [roll, pitch, yaw] in degrees
        """
        roll, pitch, yaw = state.euler_angles()
        angular_vel = state.angular_velocity * 180 / np.pi
        
        error = np.array([
            target_attitude[0] - roll,
            target_attitude[1] - pitch,
            target_attitude[2] - yaw
        ])
        
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, 
                                       -self.integral_limit, self.integral_limit)
        
        derivative = (error - self.last_error) / dt if dt > 0 else np.zeros(3)
        self.last_error = error.copy()
        
        outputs = np.zeros(3)
        for i, axis in enumerate(['roll', 'pitch', 'yaw']):
            g = self.gains[axis]
            outputs[i] = (g['Kp'] * error[i] + 
                         g['Ki'] * self.integral_error[i] + 
                         g['Kd'] * (-angular_vel[i]))
        
        roll_cmd, pitch_cmd, yaw_cmd = outputs
        roll_cmd = np.clip(roll_cmd, -max_deflection, max_deflection)
        pitch_cmd = np.clip(pitch_cmd, -max_deflection, max_deflection)
        yaw_cmd = np.clip(yaw_cmd, -max_deflection, max_deflection)
        
        left = roll_cmd + yaw_cmd
        right = roll_cmd - yaw_cmd
        up = pitch_cmd + yaw_cmd
        down = pitch_cmd - yaw_cmd
        
        deflections = np.array([left, right, up, down])
        return np.clip(deflections, -max_deflection, max_deflection)
    
    def reset(self):
        """Reset controller state."""
        self.integral_error = np.zeros(3)
        self.last_error = np.zeros(3)


class RocketSimulator:
    """6-DOF rocket flight simulator."""
    
    def __init__(self, rocket: RocketParameters = None,
                 motor: MotorProfile = None,
                 atmosphere: AtmosphereModel = None,
                 wind: WindModel = None,
                 controller: CanardController = None,
                 use_gpu: bool = False):
        
        self.rocket = rocket or RocketParameters()
        self.motor = motor or MotorProfile()
        self.atmosphere = atmosphere or AtmosphereModel()
        self.wind = wind or WindModel()
        self.controller = controller or CanardController()
        
        self.xp = cp if (use_gpu and GPU_AVAILABLE) else np
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        self.state = RocketState()
        self.state.mass = self.rocket.mass + self.motor.propellant_mass
        self.time = 0.0
        self.history = []
        self.canard_deflections = np.zeros(4)
        
    def reset(self, initial_state: RocketState = None):
        """Reset simulation to initial conditions."""
        if initial_state:
            self.state = initial_state
        else:
            self.state = RocketState()
            self.state.mass = self.rocket.mass + self.motor.propellant_mass
            q_launch = self._euler_to_quaternion(0, 85, 0)
            self.state.quaternion = q_launch
            self.state.position = np.array([0.0, 0.0, -1.0])  # Start 1m above ground (NED: -Z is up)
        
        self.time = 0.0
        self.history = []
        self.controller.reset()
        self.canard_deflections = np.zeros(4)
        
    def _euler_to_quaternion(self, roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
        """Convert Euler angles to quaternion."""
        r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
        cr, sr = np.cos(r/2), np.sin(r/2)
        cp, sp = np.cos(p/2), np.sin(p/2)
        cy, sy = np.cos(y/2), np.sin(y/2)
        
        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy
        ])
    
    def compute_forces_and_moments(self, state: RocketState, 
                                   canard_deflections: np.ndarray,
                                   t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute total forces and moments in body frame."""
        R = state.rotation_matrix()
        R_inv = R.T
        
        altitude = -state.position[2]
        rho = self.atmosphere.density(altitude)
        
        wind_inertial = self.wind.velocity(state.position, t)
        vel_air_inertial = state.velocity - wind_inertial
        vel_air_body = R_inv @ vel_air_inertial
        
        V = np.linalg.norm(vel_air_body)
        
        if V < 0.1:
            alpha = 0.0
            beta = 0.0
        else:
            alpha = np.arctan2(vel_air_body[2], vel_air_body[0])
            beta = np.arcsin(np.clip(vel_air_body[1] / V, -1, 1))
        
        q_bar = 0.5 * rho * V**2
        S = self.rocket.reference_area
        
        Cd = self.rocket.Cd0 + self.rocket.Cd_canard * np.mean(np.abs(canard_deflections))
        drag = -q_bar * S * Cd
        
        lift_alpha = q_bar * S * self.rocket.Cla * alpha
        lift_beta = q_bar * S * self.rocket.Cla * beta
        
        F_aero_body = np.array([
            drag,
            -lift_beta,
            -lift_alpha
        ])
        
        canard_lift_coef = 0.05
        left, right, up, down = np.radians(canard_deflections)
        
        F_canard = np.array([
            0,
            q_bar * self.rocket.canard_area * canard_lift_coef * (left - right),
            q_bar * self.rocket.canard_area * canard_lift_coef * (up - down)
        ])
        
        thrust = self.motor.thrust(t)
        F_thrust = np.array([thrust, 0, 0])
        
        g_body = R_inv @ np.array([0, 0, 9.81])
        F_gravity = state.mass * g_body
        
        F_total = F_aero_body + F_canard + F_thrust + F_gravity
        
        p, q, r = state.angular_velocity
        L = self.rocket.length
        
        M_aero = np.array([
            q_bar * S * L * self.rocket.Clp * p,
            q_bar * S * L * (self.rocket.Cma * alpha + self.rocket.Cmq * q),
            q_bar * S * L * (self.rocket.Cma * beta + self.rocket.Cnr * r)
        ])
        
        arm = self.rocket.canard_arm
        M_canard = np.array([
            q_bar * self.rocket.canard_area * arm * canard_lift_coef * (left + right),
            q_bar * self.rocket.canard_area * arm * canard_lift_coef * (up + down),
            q_bar * self.rocket.canard_area * arm * canard_lift_coef * (left - right) * 0.5
        ])
        
        M_total = M_aero + M_canard
        
        return F_total, M_total
    
    def derivatives(self, state_array: np.ndarray, t: float, 
                   canard_deflections: np.ndarray) -> np.ndarray:
        """Compute state derivatives for integration."""
        state = RocketState.from_array(state_array)
        
        F, M = self.compute_forces_and_moments(state, canard_deflections, t)
        
        # Clamp forces and moments for numerical stability
        F = np.clip(F, -1e6, 1e6)
        M = np.clip(M, -1e3, 1e3)
        
        R = state.rotation_matrix()
        
        pos_dot = state.velocity
        vel_dot = R @ (F / max(state.mass, 0.01))
        vel_dot = np.clip(vel_dot, -1e4, 1e4)
        
        p, q, r = np.clip(state.angular_velocity, -100, 100)
        q0, q1, q2, q3 = state.quaternion
        quat_dot = 0.5 * np.array([
            -q1*p - q2*q - q3*r,
            q0*p + q2*r - q3*q,
            q0*q + q3*p - q1*r,
            q0*r + q1*q - q2*p
        ])
        
        I = np.diag([self.rocket.Ixx, self.rocket.Iyy, self.rocket.Izz])
        omega = np.clip(state.angular_velocity, -100, 100)
        try:
            omega_dot = np.linalg.solve(I, M - np.cross(omega, I @ omega))
        except np.linalg.LinAlgError:
            omega_dot = np.zeros(3)
        omega_dot = np.clip(omega_dot, -1e4, 1e4)
        
        mass_dot = -self.motor.mass_flow(t)
        
        return np.concatenate([pos_dot, vel_dot, quat_dot, omega_dot, [mass_dot]])
    
    def step(self, dt: float, target_attitude: np.ndarray = None):
        """Advance simulation by dt seconds."""
        if target_attitude is None:
            target_attitude = np.array([0.0, 0.0, 0.0])
        
        self.canard_deflections = self.controller.compute(
            self.state, target_attitude, dt, self.rocket.canard_max_deflection
        )
        
        state_array = self.state.to_array()
        
        k1 = self.derivatives(state_array, self.time, self.canard_deflections)
        k2 = self.derivatives(state_array + 0.5*dt*k1, self.time + 0.5*dt, self.canard_deflections)
        k3 = self.derivatives(state_array + 0.5*dt*k2, self.time + 0.5*dt, self.canard_deflections)
        k4 = self.derivatives(state_array + dt*k3, self.time + dt, self.canard_deflections)
        
        state_array += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        self.state = RocketState.from_array(state_array)
        self.time += dt
        
        self.history.append({
            'time': self.time,
            'position': self.state.position.copy(),
            'velocity': self.state.velocity.copy(),
            'euler': self.state.euler_angles(),
            'angular_velocity': self.state.angular_velocity.copy(),
            'canards': self.canard_deflections.copy(),
            'altitude': -self.state.position[2],
            'speed': np.linalg.norm(self.state.velocity),
            'mass': self.state.mass
        })
        
        return self.state.position[2] > 0  # NED: positive Z is below ground
    
    def run(self, duration: float, dt: float = 0.005, 
            target_attitude: np.ndarray = None) -> dict:
        """Run complete simulation."""
        self.reset()
        
        ground_hit = False
        while self.time < duration and not ground_hit:
            ground_hit = self.step(dt, target_attitude)
        
        return self.get_results()
    
    def get_results(self) -> dict:
        """Package simulation results."""
        if not self.history:
            return {}
        
        times = np.array([h['time'] for h in self.history])
        positions = np.array([h['position'] for h in self.history])
        altitudes = np.array([h['altitude'] for h in self.history])
        speeds = np.array([h['speed'] for h in self.history])
        euler = np.array([h['euler'] for h in self.history])
        canards = np.array([h['canards'] for h in self.history])
        
        max_alt_idx = np.argmax(altitudes)
        
        return {
            'time': times,
            'position': positions,
            'altitude': altitudes,
            'speed': speeds,
            'roll': euler[:, 0],
            'pitch': euler[:, 1],
            'yaw': euler[:, 2],
            'canards': canards,
            'max_altitude': altitudes[max_alt_idx],
            'apogee_time': times[max_alt_idx],
            'flight_time': times[-1],
            'max_speed': np.max(speeds),
            'roll_rms': np.sqrt(np.mean(euler[:, 0]**2)),
            'pitch_rms': np.sqrt(np.mean((euler[:, 1] - 85)**2)),
            'stability_score': 100 - np.sqrt(np.mean(euler[:, 0]**2)) - 
                              np.sqrt(np.mean((euler[:, 1] - 85)**2))
        }


def evaluate_controller(gains: dict, wind_speed: float = 0.0,
                       num_runs: int = 1, seed: int = None) -> dict:
    """Evaluate controller performance with given gains."""
    if seed is not None:
        np.random.seed(seed)
    
    controller = CanardController(
        Kp_roll=gains.get('Kp_roll', 0.5),
        Ki_roll=gains.get('Ki_roll', 0.0),
        Kd_roll=gains.get('Kd_roll', 0.2),
        Kp_pitch=gains.get('Kp_pitch', 0.5),
        Ki_pitch=gains.get('Ki_pitch', 0.0),
        Kd_pitch=gains.get('Kd_pitch', 0.2)
    )
    
    wind = WindModel(
        base_velocity=np.array([wind_speed, 0.0, 0.0]),
        gust_intensity=wind_speed * 0.2
    )
    
    results = []
    for _ in range(num_runs):
        sim = RocketSimulator(controller=controller, wind=wind)
        result = sim.run(duration=10.0, dt=0.005)
        results.append(result)
    
    avg_results = {
        'max_altitude': np.mean([r['max_altitude'] for r in results]),
        'roll_rms': np.mean([r['roll_rms'] for r in results]),
        'pitch_rms': np.mean([r['pitch_rms'] for r in results]),
        'stability_score': np.mean([r['stability_score'] for r in results]),
        'flight_time': np.mean([r['flight_time'] for r in results])
    }
    
    return avg_results


if __name__ == "__main__":
    print("Rocket 6-DOF Simulator")
    print("=" * 50)
    print(f"GPU Available: {GPU_AVAILABLE}")
    
    sim = RocketSimulator()
    results = sim.run(duration=8.0, dt=0.005)
    
    print(f"\nFlight Results:")
    print(f"  Max Altitude: {results['max_altitude']:.1f} m")
    print(f"  Apogee Time: {results['apogee_time']:.2f} s")
    print(f"  Max Speed: {results['max_speed']:.1f} m/s")
    print(f"  Flight Time: {results['flight_time']:.2f} s")
    print(f"  Roll RMS: {results['roll_rms']:.2f}°")
    print(f"  Pitch RMS: {results['pitch_rms']:.2f}°")
    print(f"  Stability Score: {results['stability_score']:.1f}")
