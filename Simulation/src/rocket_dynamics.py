"""
6-DOF Rocket Flight Dynamics Simulation
========================================
Canard-stabilized rocket with folding fins.

Physical system modeled:
- ~200g airframe with 4 canard surfaces in '+' configuration
- Differential mixing: roll + pitch + yaw commands (firmware may use roll-only)
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
    """Physical parameters from Barrowman sweep of improved design.

    Improved design (candidate #1 from design_sweep.py):
      Von Karman nose 120mm, body 549mm (75mm OD), boat tail 30mm,
      4x folding fins 60mm span / 70mm root chord / 30deg sweep,
      4x canards 20mm span / 45mm root chord at 100mm from nose.
    Barrowman predictions: CNa=8.21, CP=430mm, stability=1.06 cal, Cd0=0.64.
    """
    mass: float = 0.200
    length: float = 0.669          # m (120mm nose + 549mm body)
    diameter: float = 0.075        # m (75 mm OD)

    Ixx: float = 0.00014          # kg*m^2 roll  (m*r^2/2)
    Iyy: float = 0.0074           # kg*m^2 pitch (m*L^2/12 = 0.2*0.669^2/12)
    Izz: float = 0.0074           # kg*m^2 yaw

    cg_from_nose: float = 0.35    # m (assembly heavy aft, unchanged)
    cp_from_nose: float = 0.430   # m (Barrowman: 429.5mm)

    nose_length: float = 0.120    # m (120mm Von Karman, up from 38mm)
    body_length: float = 0.549    # m (unchanged)

    # Planform ~20 mm span x 45 mm chord (sweep winner)
    canard_area: float = 0.0009   # m^2 per canard
    canard_arm: float = 0.0475    # m, body radius + half canard span (20 mm)
    canard_from_nose: float = 0.10  # m from nose tip (100 mm)
    num_canards: int = 4
    canard_max_deflection: float = 12.0
    canard_Cla: float = 3.5       # improved AR gives better lift slope

    fin_area: float = 0.00315     # m^2 per fin (70mm x 90mm avg span area)
    fin_from_nose: float = 0.59   # m, near aft end
    num_fins: int = 4
    reference_area: float = 0.00442  # m^2, pi * (0.0375)^2

    Cd0: float = 0.64             # Barrowman estimate (improved nose)
    Cd_canard: float = 0.005      # incremental drag per degree deflection
    CNa: float = 8.2              # Barrowman: 8.21 /rad

    Clp: float = -0.15            # larger fins = more roll damping
    Cmq: float = -12.0            # larger fins = more pitch damping
    Cnr: float = -12.0            # larger fins = more yaw damping

    aero_table: object = field(default=None, repr=False)

    @property
    def canard_pitch_arm(self) -> float:
        """Moment arm (m) from canard normal force to CG along body x."""
        return max(0.01, self.cg_from_nose - self.canard_from_nose)


@dataclass
class MotorProfile:
    """Solid rocket motor thrust profile (scaled so time-integral = total_impulse)."""
    burn_time: float = 1.2
    total_impulse: float = 10.0
    peak_thrust: float = 15.0
    propellant_mass: float = 0.020
    _thrust_scale: float = field(default=1.0, repr=False, init=False)

    def __post_init__(self):
        ramp, tail = 0.02, 0.05
        plateau = max(self.burn_time - ramp - tail, 0.0)
        raw_integral = self.peak_thrust * (0.5 * ramp + plateau + 0.5 * tail)
        if raw_integral > 1e-9:
            object.__setattr__(self, '_thrust_scale', self.total_impulse / raw_integral)
        else:
            object.__setattr__(self, '_thrust_scale', 1.0)

    def _thrust_unscaled(self, t: float) -> float:
        if t < 0 or t > self.burn_time:
            return 0.0
        ramp = 0.02
        if t < ramp:
            return self.peak_thrust * (t / ramp)
        if t > self.burn_time - 0.05:
            return self.peak_thrust * ((self.burn_time - t) / 0.05)
        return self.peak_thrust

    def thrust(self, t: float) -> float:
        return self._thrust_unscaled(t) * self._thrust_scale

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


def mix_canards(roll_cmd: float, pitch_cmd: float, yaw_cmd: float,
                max_abs: float) -> np.ndarray:
    """Map roll/pitch/yaw surface commands (deg) to [L,R,U,D] deflections."""
    d = np.array([
        roll_cmd - pitch_cmd + yaw_cmd,
        roll_cmd - pitch_cmd - yaw_cmd,
        roll_cmd + pitch_cmd - yaw_cmd,
        roll_cmd + pitch_cmd + yaw_cmd,
    ], dtype=float)
    m = np.max(np.abs(d))
    if m > max_abs and m > 1e-9:
        d *= max_abs / m
    return d


def _wrap_deg(err: float) -> float:
    while err > 180.0:
        err -= 360.0
    while err < -180.0:
        err += 360.0
    return err


class CanardController:
    """
    PID on roll, pitch, yaw with D-term from body angular rates (deg/s),
    matching firmware convention for roll. Outputs [L,R,U,D] via mix_canards.
    Set pitch/yaw gains to zero for roll-only behaviour.
    """

    def __init__(
        self,
        Kp_roll=0.5, Ki_roll=0.0, Kd_roll=0.2,
        Kp_pitch=0.0, Ki_pitch=0.0, Kd_pitch=0.0,
        Kp_yaw=0.0, Ki_yaw=0.0, Kd_yaw=0.0,
        **_kwargs,
    ):
        self.Kp_r = Kp_roll
        self.Ki_r = Ki_roll
        self.Kd_r = Kd_roll
        self.Kp_p = Kp_pitch
        self.Ki_p = Ki_pitch
        self.Kd_p = Kd_pitch
        self.Kp_y = Kp_yaw
        self.Ki_y = Ki_yaw
        self.Kd_y = Kd_yaw
        self.ir = self.ip = self.iy = 0.0
        self.integral_limit = 15.0

    def reset(self):
        self.ir = self.ip = self.iy = 0.0

    def compute(self, state: RocketState, target_deg: np.ndarray, dt: float,
                max_deflection: float = 12.0) -> np.ndarray:
        roll_deg, pitch_deg, yaw_deg = state.euler_angles()
        p, q, r = state.angular_velocity
        roll_rate_deg = p * 180.0 / np.pi
        pitch_rate_deg = q * 180.0 / np.pi
        yaw_rate_deg = r * 180.0 / np.pi

        tr, tp, ty = float(target_deg[0]), float(target_deg[1]), float(target_deg[2])
        e_r = _wrap_deg(roll_deg - tr)
        e_p = _wrap_deg(pitch_deg - tp)
        e_y = _wrap_deg(yaw_deg - ty)

        self.ir += e_r * dt
        self.ip += e_p * dt
        self.iy += e_y * dt
        lim = self.integral_limit
        self.ir = float(np.clip(self.ir, -lim, lim))
        self.ip = float(np.clip(self.ip, -lim, lim))
        self.iy = float(np.clip(self.iy, -lim, lim))

        r_cmd = self.Kp_r * e_r + self.Ki_r * self.ir + self.Kd_r * roll_rate_deg
        p_cmd = self.Kp_p * e_p + self.Ki_p * self.ip + self.Kd_p * pitch_rate_deg
        y_cmd = self.Kp_y * e_y + self.Ki_y * self.iy + self.Kd_y * yaw_rate_deg

        return mix_canards(r_cmd, p_cmd, y_cmd, max_deflection)


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
        self._launch_pitch = 85.0

    def reset(self, initial_state=None, launch_pitch=85.0):
        self._launch_pitch = float(launch_pitch)
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
        alpha_deg = np.degrees(alpha)
        beta_deg = np.degrees(beta)

        if rk.aero_table is not None:
            Cd = rk.aero_table.Cd(V, alpha_deg) + rk.Cd_canard * avg_defl
            CNa_eff = rk.CNa
            if abs(alpha_deg) > 0.1:
                Cl_cfd = rk.aero_table.Cl(V, alpha_deg)
                CNa_eff = Cl_cfd / alpha if abs(alpha) > 1e-6 else rk.CNa
        else:
            Cd = rk.Cd0 + rk.Cd_canard * avg_defl
            CNa_eff = rk.CNa

        F_axial = -q_bar * S * Cd

        stability_margin = rk.cp_from_nose - rk.cg_from_nose
        F_N_alpha = q_bar * S * CNa_eff * alpha
        F_N_beta = q_bar * S * CNa_eff * beta

        # --- Canard control (differential [L,R,U,D], deg) ---
        dL, dR, dU, dD = [float(x) for x in canard_defl[:4]]
        dr = np.pi / 180.0
        qSc = q_bar * rk.canard_area * rk.canard_Cla * dr
        arm_r = rk.canard_arm
        arm_p = rk.canard_pitch_arm
        M_roll_canard = arm_r * qSc * (dL - dR + dD - dU)
        M_pitch_canard = arm_p * qSc * (dU + dD - dL - dR)
        M_yaw_canard = arm_r * qSc * (dL - dR - dU + dD)

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

        M_pitch = (-q_bar * S * L * (CNa_eff * stability_margin / L * alpha + rk.Cmq * nondim * q_r)
                   + M_pitch_canard)
        M_yaw = (q_bar * S * L * (CNa_eff * stability_margin / L * beta + rk.Cnr * nondim * r_r)
                 + M_yaw_canard)

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
            target_attitude = np.array([0.0, self._launch_pitch, 0.0])
        else:
            target_attitude = np.asarray(target_attitude, dtype=float).reshape(3)
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
        pitch_all = euler[:, 1]
        pitch_burn = pitch_all[burn_mask] if np.any(burn_mask) else pitch_all
        lp = getattr(self, '_launch_pitch', 85.0)
        pitch_err_burn = pitch_burn - lp

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
            'pitch_rms_burn': float(np.sqrt(np.mean(pitch_err_burn ** 2))),
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
        Kp_pitch=gains.get('Kp_pitch', 0.0),
        Ki_pitch=gains.get('Ki_pitch', 0.0),
        Kd_pitch=gains.get('Kd_pitch', 0.0),
        Kp_yaw=gains.get('Kp_yaw', 0.0),
        Ki_yaw=gains.get('Ki_yaw', 0.0),
        Kd_yaw=gains.get('Kd_yaw', 0.0),
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
        'pitch_rms_burn': float(np.mean([r.get('pitch_rms_burn', 0.0) for r in results])),
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
