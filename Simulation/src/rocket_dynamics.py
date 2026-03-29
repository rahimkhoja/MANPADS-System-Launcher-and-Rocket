"""
6-DOF Rocket Flight Dynamics Simulation
========================================
Canard-stabilized rocket with folding fins.

Physical system modeled:
- ~200g airframe with 4 canard surfaces in '+' configuration
- All canards deflect identically for roll-only control
- Folding tail fins provide passive pitch/yaw stability
- C-class solid motor (~10 N*s total impulse)
- MPU6050 IMU sampled at 200 Hz, ESP32 flight computer
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
    """Physical parameters derived from CAD and OpenRocket analysis."""
    mass: float = 0.200
    length: float = 0.30
    diameter: float = 0.04

    Ixx: float = 0.00005   # kg*m^2 roll (thin cylinder)
    Iyy: float = 0.0015    # kg*m^2 pitch
    Izz: float = 0.0015    # kg*m^2 yaw

    cg_from_nose: float = 0.18
    cp_from_nose: float = 0.22  # 2cm stability margin (1.0 cal)

    canard_area: float = 0.0006   # m^2 per canard (~20x30mm)
    canard_arm: float = 0.04      # m, moment arm from body axis to canard lift centre
    num_canards: int = 4
    canard_max_deflection: float = 12.0
    canard_Cla: float = 3.0       # per-radian lift slope for a small flat plate

    fin_area: float = 0.001
    reference_area: float = 0.00126  # pi * (0.02)^2

    Cd0: float = 0.45
    Cd_canard: float = 0.005  # incremental drag per degree deflection
    CNa: float = 8.0          # normal force slope of full rocket (per radian)

    Clp: float = -0.08        # roll damping from fins (dimensionless)
    Cmq: float = -8.0         # pitch damping coefficient
    Cnr: float = -8.0         # yaw damping coefficient


@dataclass
class MotorProfile:
    """Solid rocket motor thrust profile."""
    burn_time: float = 1.2
    total_impulse: float = 10.0
    peak_thrust: float = 15.0
    propellant_mass: float = 0.020

    def thrust(self, t: float) -> float:
        if t < 0 or t > self.burn_time:
            return 0.0
        ramp = 0.02
        if t < ramp:
            return self.peak_thrust * (t / ramp)
        if t > self.burn_time - 0.05:
            return self.peak_thrust * ((self.burn_time - t) / 0.05)
        return self.peak_thrust

    def mass_flow(self, t: float) -> float:
        if 0 <= t <= self.burn_time:
            return self.propellant_mass / self.burn_time
        return 0.0


@dataclass
class AtmosphereModel:
    sea_level_density: float = 1.225
    sea_level_temp: float = 288.15
    temperature_lapse: float = 0.0065

    def density(self, altitude: float) -> float:
        T = self.sea_level_temp - self.temperature_lapse * max(altitude, 0)
        return self.sea_level_density * (T / self.sea_level_temp) ** 4.256


@dataclass
class WindModel:
    """Dryden-style turbulence: filtered white noise with temporal correlation."""
    base_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gust_intensity: float = 0.0
    tau: float = 0.5  # time constant (seconds)

    def __post_init__(self):
        self._gust_state = np.zeros(3)
        self._rng = np.random.RandomState()

    def seed(self, s: int):
        self._rng = np.random.RandomState(s)
        self._gust_state = np.zeros(3)

    def velocity(self, position: np.ndarray, t: float, dt: float = 0.005) -> np.ndarray:
        if self.gust_intensity == 0:
            return self.base_velocity.copy()
        alpha = dt / max(self.tau, dt)
        white = self._rng.randn(3) * self.gust_intensity
        self._gust_state = (1 - alpha) * self._gust_state + alpha * white
        return self.base_velocity + self._gust_state


@dataclass
class RocketState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 0.220

    def euler_angles(self) -> Tuple[float, float, float]:
        q0, q1, q2, q3 = self.quaternion
        roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
        pitch = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1, 1))
        yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    def rotation_matrix(self) -> np.ndarray:
        q0, q1, q2, q3 = self.quaternion
        return np.array([
            [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
            [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
            [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
        ])

    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.position, self.velocity, self.quaternion,
            self.angular_velocity, [self.mass]
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'RocketState':
        s = cls()
        s.position = arr[0:3].copy()
        s.velocity = arr[3:6].copy()
        s.quaternion = arr[6:10].copy()
        norm = np.linalg.norm(s.quaternion)
        if norm > 1e-8:
            s.quaternion /= norm
        s.angular_velocity = arr[10:13].copy()
        s.mass = arr[13]
        return s


class CanardController:
    """
    Roll-only PD controller matching the original firmware.
    output = Kp * roll_angle + Kd * roll_rate
    Applied identically to all 4 canards.
    """

    def __init__(self, Kp_roll=0.5, Ki_roll=0.0, Kd_roll=0.2, **_kwargs):
        self.Kp = Kp_roll
        self.Ki = Ki_roll
        self.Kd = Kd_roll
        self.integral = 0.0
        self.integral_limit = 15.0

    def compute(self, state: RocketState, _target, dt: float,
                max_deflection: float = 12.0) -> np.ndarray:
        roll_deg = state.euler_angles()[0]
        roll_rate_deg = state.angular_velocity[0] * 180.0 / np.pi

        self.integral += roll_deg * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        output = self.Kp * roll_deg + self.Ki * self.integral + self.Kd * roll_rate_deg
        deflection = np.clip(output, -max_deflection, max_deflection)
        return np.full(4, deflection)

    def reset(self):
        self.integral = 0.0


class RocketSimulator:

    def __init__(self, rocket=None, motor=None, atmosphere=None,
                 wind=None, controller=None, use_gpu=False):
        self.rocket = rocket or RocketParameters()
        self.motor = motor or MotorProfile()
        self.atmosphere = atmosphere or AtmosphereModel()
        self.wind = wind or WindModel()
        self.controller = controller or CanardController()
        self.state = RocketState()
        self.state.mass = self.rocket.mass + self.motor.propellant_mass
        self.time = 0.0
        self.dt = 0.005
        self.history = []
        self.canard_deflections = np.zeros(4)

    def reset(self, initial_state=None, launch_pitch=85.0):
        if initial_state:
            self.state = initial_state
        else:
            self.state = RocketState()
            self.state.mass = self.rocket.mass + self.motor.propellant_mass
            self.state.quaternion = self._euler_to_quat(0, launch_pitch, 0)
            self.state.position = np.array([0.0, 0.0, -1.0])
        self.time = 0.0
        self.history = []
        self.controller.reset()
        self.canard_deflections = np.zeros(4)

    @staticmethod
    def _euler_to_quat(r_deg, p_deg, y_deg) -> np.ndarray:
        r, p, y = np.radians([r_deg, p_deg, y_deg])
        cr, sr = np.cos(r/2), np.sin(r/2)
        cp, sp = np.cos(p/2), np.sin(p/2)
        cy, sy = np.cos(y/2), np.sin(y/2)
        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy
        ])

    # ------------------------------------------------------------------
    def compute_forces_and_moments(self, state, canard_defl, t):
        R = state.rotation_matrix()
        Rt = R.T
        rk = self.rocket

        altitude = -state.position[2]
        rho = self.atmosphere.density(altitude)

        wind_ned = self.wind.velocity(state.position, t, self.dt)
        v_air_ned = state.velocity - wind_ned
        v_body = Rt @ v_air_ned
        V = np.linalg.norm(v_body)

        if V < 0.1:
            alpha = beta = 0.0
        else:
            alpha = np.arctan2(v_body[2], v_body[0])
            beta = np.arcsin(np.clip(v_body[1] / V, -1, 1))

        q_bar = 0.5 * rho * V**2
        S = rk.reference_area
        L = rk.length
        d = rk.diameter

        # --- Axial force (drag) ---
        avg_defl = np.mean(np.abs(canard_defl))
        Cd = rk.Cd0 + rk.Cd_canard * avg_defl
        F_axial = -q_bar * S * Cd

        # --- Normal force from body + fins (restoring) ---
        stability_margin = rk.cp_from_nose - rk.cg_from_nose
        F_N_alpha = q_bar * S * rk.CNa * alpha
        F_N_beta = q_bar * S * rk.CNa * beta

        # --- Canard forces ---
        # All 4 canards deflect identically; each produces lift perpendicular
        # to body x-axis, offset from the roll axis by canard_arm.
        defl_rad = np.radians(canard_defl[0])  # all identical
        F_canard_per = q_bar * rk.canard_area * rk.canard_Cla * defl_rad

        # --- Thrust ---
        thrust = self.motor.thrust(t)

        # --- Gravity in body frame ---
        g_body = Rt @ np.array([0.0, 0.0, 9.81])

        F_body = np.array([
            F_axial + thrust,
            -F_N_beta,
            -F_N_alpha
        ]) + state.mass * g_body

        # --- Moments ---
        p_r, q_r, r_r = state.angular_velocity
        nondim = d / (2 * max(V, 0.5))

        # Pitch/yaw restoring + damping
        M_pitch = -q_bar * S * L * (rk.CNa * stability_margin / L * alpha + rk.Cmq * nondim * q_r)
        M_yaw   = q_bar * S * L * (rk.CNa * stability_margin / L * beta + rk.Cnr * nondim * r_r)

        # Roll: canard torque + fin damping
        M_roll_canard = rk.num_canards * F_canard_per * rk.canard_arm
        M_roll_damp = q_bar * S * d * rk.Clp * nondim * p_r

        M_body = np.array([
            M_roll_canard + M_roll_damp,
            M_pitch,
            M_yaw
        ])

        return F_body, M_body

    # ------------------------------------------------------------------
    def derivatives(self, sa, t, canard_defl):
        state = RocketState.from_array(sa)
        F, M = self.compute_forces_and_moments(state, canard_defl, t)

        F = np.clip(F, -1e5, 1e5)
        M = np.clip(M, -1e2, 1e2)

        R = state.rotation_matrix()
        pos_dot = state.velocity
        vel_dot = R @ (F / max(state.mass, 0.05))

        p, q, r = np.clip(state.angular_velocity, -50, 50)
        q0, q1, q2, q3 = state.quaternion
        quat_dot = 0.5 * np.array([
            -q1*p - q2*q - q3*r,
             q0*p + q2*r - q3*q,
             q0*q + q3*p - q1*r,
             q0*r + q1*q - q2*p
        ])

        Ixx, Iyy, Izz = self.rocket.Ixx, self.rocket.Iyy, self.rocket.Izz
        omega = np.clip(state.angular_velocity, -50, 50)
        omega_dot = np.array([
            (M[0] - (Izz - Iyy) * omega[1] * omega[2]) / Ixx,
            (M[1] - (Ixx - Izz) * omega[0] * omega[2]) / Iyy,
            (M[2] - (Iyy - Ixx) * omega[0] * omega[1]) / Izz,
        ])
        omega_dot = np.clip(omega_dot, -1e4, 1e4)

        mass_dot = -self.motor.mass_flow(t)
        return np.concatenate([pos_dot, vel_dot, quat_dot, omega_dot, [mass_dot]])

    # ------------------------------------------------------------------
    def step(self, dt, target_attitude=None):
        if target_attitude is None:
            target_attitude = np.zeros(3)
        self.dt = dt

        self.canard_deflections = self.controller.compute(
            self.state, target_attitude, dt, self.rocket.canard_max_deflection
        )

        sa = self.state.to_array()
        k1 = self.derivatives(sa, self.time, self.canard_deflections)
        k2 = self.derivatives(sa + 0.5*dt*k1, self.time + 0.5*dt, self.canard_deflections)
        k3 = self.derivatives(sa + 0.5*dt*k2, self.time + 0.5*dt, self.canard_deflections)
        k4 = self.derivatives(sa + dt*k3, self.time + dt, self.canard_deflections)
        sa += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        self.state = RocketState.from_array(sa)
        self.time += dt

        roll, pitch, yaw = self.state.euler_angles()
        self.history.append({
            'time': self.time,
            'position': self.state.position.copy(),
            'velocity': self.state.velocity.copy(),
            'euler': (roll, pitch, yaw),
            'angular_velocity': self.state.angular_velocity.copy(),
            'canards': self.canard_deflections.copy(),
            'altitude': -self.state.position[2],
            'speed': np.linalg.norm(self.state.velocity),
            'mass': self.state.mass,
            'thrust': self.motor.thrust(self.time),
        })
        return self.state.position[2] > 0

    # ------------------------------------------------------------------
    def run(self, duration=10.0, dt=0.005, target_attitude=None, launch_pitch=85.0):
        self.reset(launch_pitch=launch_pitch)
        hit = False
        while self.time < duration and not hit:
            hit = self.step(dt, target_attitude)
        return self.get_results()

    # ------------------------------------------------------------------
    def get_results(self):
        if not self.history:
            return {}

        times = np.array([h['time'] for h in self.history])
        altitudes = np.array([h['altitude'] for h in self.history])
        speeds = np.array([h['speed'] for h in self.history])
        euler = np.array([h['euler'] for h in self.history])
        canards = np.array([h['canards'] for h in self.history])
        positions = np.array([h['position'] for h in self.history])

        burn_mask = times <= self.motor.burn_time
        coast_mask = ~burn_mask

        max_alt_idx = np.argmax(altitudes)

        roll_all = euler[:, 0]
        roll_burn = roll_all[burn_mask] if np.any(burn_mask) else roll_all

        return {
            'time': times,
            'position': positions,
            'altitude': altitudes,
            'speed': speeds,
            'roll': roll_all,
            'pitch': euler[:, 1],
            'yaw': euler[:, 2],
            'canards': canards,
            'max_altitude': float(altitudes[max_alt_idx]),
            'apogee_time': float(times[max_alt_idx]),
            'flight_time': float(times[-1]),
            'max_speed': float(np.max(speeds)),
            'roll_rms': float(np.sqrt(np.mean(roll_all**2))),
            'roll_rms_burn': float(np.sqrt(np.mean(roll_burn**2))),
            'burn_time': float(self.motor.burn_time),
        }


def evaluate_controller(gains: dict, wind_speed=0.0, num_runs=1, seed=None):
    """Evaluate controller across multiple runs."""
    if seed is not None:
        np.random.seed(seed)

    controller = CanardController(
        Kp_roll=gains.get('Kp_roll', 0.5),
        Ki_roll=gains.get('Ki_roll', 0.0),
        Kd_roll=gains.get('Kd_roll', 0.2),
    )

    wind = WindModel(
        base_velocity=np.array([wind_speed, 0.0, 0.0]),
        gust_intensity=wind_speed * 0.3,
    )

    results = []
    for i in range(num_runs):
        wind.seed(seed + i if seed else i)
        sim = RocketSimulator(controller=controller, wind=wind)
        r = sim.run(duration=10.0, dt=0.005)
        results.append(r)

    return {
        'max_altitude': float(np.mean([r['max_altitude'] for r in results])),
        'roll_rms': float(np.mean([r['roll_rms'] for r in results])),
        'roll_rms_burn': float(np.mean([r['roll_rms_burn'] for r in results])),
        'flight_time': float(np.mean([r['flight_time'] for r in results])),
    }


if __name__ == "__main__":
    print("Rocket 6-DOF Simulator")
    print("=" * 50)
    print(f"GPU Available: {GPU_AVAILABLE}")

    print("\n--- No wind ---")
    sim = RocketSimulator()
    r = sim.run(duration=10.0)
    print(f"  Max Altitude: {r['max_altitude']:.1f} m")
    print(f"  Max Speed:    {r['max_speed']:.1f} m/s")
    print(f"  Roll RMS:     {r['roll_rms']:.2f}°  (burn: {r['roll_rms_burn']:.2f}°)")
    print(f"  Apogee Time:  {r['apogee_time']:.2f} s")

    print("\n--- 3 m/s crosswind ---")
    wind = WindModel(base_velocity=np.array([3.0, 0, 0]), gust_intensity=1.0)
    wind.seed(42)
    sim2 = RocketSimulator(wind=wind)
    r2 = sim2.run(duration=10.0)
    print(f"  Max Altitude: {r2['max_altitude']:.1f} m")
    print(f"  Roll RMS:     {r2['roll_rms']:.2f}°  (burn: {r2['roll_rms_burn']:.2f}°)")
