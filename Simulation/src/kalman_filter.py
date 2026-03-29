"""
Extended Kalman Filter for Rocket State Estimation
==================================================
Fuses gyroscope and accelerometer data for accurate attitude estimation.
Eliminates gyro drift and provides smooth state estimates.

Designed to run on ESP32 with fixed-point optimization options.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class IMUMeasurement:
    """Raw IMU measurement packet."""
    gyro: np.ndarray      # rad/s [p, q, r]
    accel: np.ndarray     # m/s^2 [ax, ay, az]
    timestamp: float      # seconds


class AttitudeEKF:
    """
    Extended Kalman Filter for 3-axis attitude estimation.
    
    State vector: [roll, pitch, yaw, gyro_bias_x, gyro_bias_y, gyro_bias_z]
    """
    
    def __init__(self, 
                 process_noise_gyro: float = 0.001,
                 process_noise_bias: float = 0.0001,
                 measurement_noise_accel: float = 0.5,
                 initial_bias: np.ndarray = None):
        
        self.state = np.zeros(6)
        
        self.P = np.eye(6) * 0.1
        self.P[3:6, 3:6] = np.eye(3) * 0.01
        
        self.Q = np.diag([
            process_noise_gyro, process_noise_gyro, process_noise_gyro,
            process_noise_bias, process_noise_bias, process_noise_bias
        ])
        
        self.R = np.eye(2) * measurement_noise_accel
        
        if initial_bias is not None:
            self.state[3:6] = initial_bias
            
        self.last_time = None
        self.initialized = False
        
    def reset(self, roll: float = 0, pitch: float = 0, yaw: float = 0,
              gyro_bias: np.ndarray = None):
        """Reset filter state."""
        self.state[0] = roll
        self.state[1] = pitch
        self.state[2] = yaw
        if gyro_bias is not None:
            self.state[3:6] = gyro_bias
        else:
            self.state[3:6] = 0
        
        self.P = np.eye(6) * 0.1
        self.P[3:6, 3:6] = np.eye(3) * 0.01
        self.initialized = True
        
    def predict(self, gyro: np.ndarray, dt: float):
        """
        Prediction step using gyroscope measurements.
        
        gyro: Angular velocity [p, q, r] in rad/s
        dt: Time step in seconds
        """
        roll, pitch, yaw = self.state[0:3]
        bias = self.state[3:6]
        
        gyro_corrected = gyro - bias
        p, q, r = gyro_corrected
        
        cr, sr = np.cos(roll), np.sin(roll)
        cp, tp = np.cos(pitch), np.tan(pitch)
        
        if abs(cp) < 0.001:
            cp = 0.001 * np.sign(cp) if cp != 0 else 0.001
            tp = np.tan(np.arccos(cp))
        
        roll_dot = p + sr * tp * q + cr * tp * r
        pitch_dot = cr * q - sr * r
        yaw_dot = (sr / cp) * q + (cr / cp) * r
        
        self.state[0] += roll_dot * dt
        self.state[1] += pitch_dot * dt
        self.state[2] += yaw_dot * dt
        
        self.state[0] = self._wrap_angle(self.state[0])
        self.state[2] = self._wrap_angle(self.state[2])
        self.state[1] = np.clip(self.state[1], -np.pi/2 + 0.01, np.pi/2 - 0.01)
        
        F = self._compute_jacobian_F(self.state, gyro_corrected, dt)
        
        self.P = F @ self.P @ F.T + self.Q * dt
        
    def update(self, accel: np.ndarray):
        """
        Update step using accelerometer measurements.
        
        accel: Acceleration [ax, ay, az] in m/s^2
        """
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 0.1:
            return
            
        accel_normalized = accel / accel_norm
        
        roll_meas = np.arctan2(accel_normalized[1], accel_normalized[2])
        pitch_meas = np.arctan2(-accel_normalized[0], 
                                np.sqrt(accel_normalized[1]**2 + accel_normalized[2]**2))
        
        z = np.array([roll_meas, pitch_meas])
        
        roll, pitch = self.state[0], self.state[1]
        h = np.array([roll, pitch])
        
        H = np.zeros((2, 6))
        H[0, 0] = 1  # d(roll_pred)/d(roll)
        H[1, 1] = 1  # d(pitch_pred)/d(pitch)
        
        y = z - h
        y[0] = self._wrap_angle(y[0])
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state += K @ y
        
        self.state[0] = self._wrap_angle(self.state[0])
        self.state[1] = np.clip(self.state[1], -np.pi/2 + 0.01, np.pi/2 - 0.01)
        self.state[2] = self._wrap_angle(self.state[2])
        
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        
    def process_imu(self, measurement: IMUMeasurement) -> Tuple[float, float, float]:
        """
        Process a complete IMU measurement.
        Returns (roll, pitch, yaw) in radians.
        """
        if self.last_time is None:
            self.last_time = measurement.timestamp
            accel = measurement.accel
            accel_norm = np.linalg.norm(accel)
            if accel_norm > 0.1:
                accel_n = accel / accel_norm
                roll = np.arctan2(accel_n[1], accel_n[2])
                pitch = np.arctan2(-accel_n[0], 
                                   np.sqrt(accel_n[1]**2 + accel_n[2]**2))
                self.reset(roll, pitch, 0)
            return self.get_attitude()
        
        dt = measurement.timestamp - self.last_time
        self.last_time = measurement.timestamp
        
        if dt <= 0 or dt > 1.0:
            return self.get_attitude()
        
        self.predict(measurement.gyro, dt)
        
        self.update(measurement.accel)
        
        return self.get_attitude()
    
    def get_attitude(self) -> Tuple[float, float, float]:
        """Get current attitude estimate in radians."""
        return self.state[0], self.state[1], self.state[2]
    
    def get_attitude_degrees(self) -> Tuple[float, float, float]:
        """Get current attitude estimate in degrees."""
        return tuple(np.degrees(self.state[0:3]))
    
    def get_gyro_bias(self) -> np.ndarray:
        """Get estimated gyro bias."""
        return self.state[3:6].copy()
    
    def get_covariance(self) -> np.ndarray:
        """Get state covariance matrix."""
        return self.P.copy()
    
    def _compute_jacobian_F(self, state: np.ndarray, gyro: np.ndarray, 
                           dt: float) -> np.ndarray:
        """Compute state transition Jacobian."""
        roll, pitch = state[0], state[1]
        p, q, r = gyro
        
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        tp = np.tan(pitch)
        
        if abs(cp) < 0.001:
            cp = 0.001
            
        F = np.eye(6)
        
        F[0, 0] = 1 + dt * (cr * tp * q - sr * tp * r)
        F[0, 1] = dt * (sr * q + cr * r) / (cp * cp)
        F[0, 3] = -dt
        F[0, 4] = -dt * sr * tp
        F[0, 5] = -dt * cr * tp
        
        F[1, 0] = dt * (-sr * q - cr * r)
        F[1, 4] = -dt * cr
        F[1, 5] = dt * sr
        
        F[2, 0] = dt * (cr * q - sr * r) / cp
        F[2, 1] = dt * (sr * q + cr * r) * sp / (cp * cp)
        F[2, 4] = -dt * sr / cp
        F[2, 5] = -dt * cr / cp
        
        return F
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class ComplementaryFilter:
    """
    Simple complementary filter as a lightweight alternative.
    Good for resource-constrained systems.
    """
    
    def __init__(self, alpha: float = 0.98):
        """
        alpha: Weight for gyro integration (0.98 = trust gyro 98%)
        """
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0
        self.last_time = None
        
    def reset(self, roll: float = 0, pitch: float = 0):
        """Reset filter state."""
        self.roll = roll
        self.pitch = pitch
        
    def update(self, gyro: np.ndarray, accel: np.ndarray, 
               dt: float) -> Tuple[float, float]:
        """
        Update filter with new measurements.
        
        gyro: Angular velocity [p, q, r] in rad/s
        accel: Acceleration [ax, ay, az] in m/s^2
        dt: Time step in seconds
        
        Returns (roll, pitch) in radians.
        """
        p, q, r = gyro
        
        roll_gyro = self.roll + p * dt
        pitch_gyro = self.pitch + q * dt
        
        accel_norm = np.linalg.norm(accel)
        if accel_norm > 0.1:
            accel_n = accel / accel_norm
            roll_accel = np.arctan2(accel_n[1], accel_n[2])
            pitch_accel = np.arctan2(-accel_n[0], 
                                     np.sqrt(accel_n[1]**2 + accel_n[2]**2))
        else:
            roll_accel = roll_gyro
            pitch_accel = pitch_gyro
        
        self.roll = self.alpha * roll_gyro + (1 - self.alpha) * roll_accel
        self.pitch = self.alpha * pitch_gyro + (1 - self.alpha) * pitch_accel
        
        return self.roll, self.pitch
    
    def get_attitude_degrees(self) -> Tuple[float, float]:
        """Get attitude in degrees."""
        return np.degrees(self.roll), np.degrees(self.pitch)


def generate_cpp_kalman_filter():
    """Generate C++ code for ESP32 implementation."""
    
    cpp_code = '''
// Auto-generated Kalman Filter for ESP32
// Optimized for MPU6050 IMU

#ifndef ATTITUDE_EKF_H
#define ATTITUDE_EKF_H

#include <Arduino.h>
#include <math.h>

class AttitudeEKF {
public:
    // State: [roll, pitch, yaw, bias_x, bias_y, bias_z]
    float state[6] = {0};
    float P[6][6];  // Covariance matrix
    
    // Tuning parameters
    float Q_gyro = 0.001f;
    float Q_bias = 0.0001f;
    float R_accel = 0.5f;
    
    AttitudeEKF() {
        reset();
    }
    
    void reset() {
        for (int i = 0; i < 6; i++) {
            state[i] = 0;
            for (int j = 0; j < 6; j++) {
                P[i][j] = (i == j) ? 0.1f : 0;
            }
        }
        P[3][3] = P[4][4] = P[5][5] = 0.01f;
    }
    
    void predict(float gx, float gy, float gz, float dt) {
        // Correct gyro with bias estimate
        float p = gx - state[3];
        float q = gy - state[4];
        float r = gz - state[5];
        
        float roll = state[0];
        float pitch = state[1];
        
        float cr = cosf(roll), sr = sinf(roll);
        float cp = cosf(pitch);
        float tp = tanf(pitch);
        
        if (fabsf(cp) < 0.001f) cp = 0.001f;
        
        // Euler angle rates
        float roll_dot = p + sr * tp * q + cr * tp * r;
        float pitch_dot = cr * q - sr * r;
        float yaw_dot = (sr / cp) * q + (cr / cp) * r;
        
        state[0] += roll_dot * dt;
        state[1] += pitch_dot * dt;
        state[2] += yaw_dot * dt;
        
        // Wrap angles
        state[0] = wrapAngle(state[0]);
        state[2] = wrapAngle(state[2]);
        state[1] = constrain(state[1], -M_PI_2 + 0.01f, M_PI_2 - 0.01f);
        
        // Simplified covariance update (diagonal approximation for speed)
        for (int i = 0; i < 3; i++) P[i][i] += Q_gyro * dt;
        for (int i = 3; i < 6; i++) P[i][i] += Q_bias * dt;
    }
    
    void update(float ax, float ay, float az) {
        float accel_norm = sqrtf(ax*ax + ay*ay + az*az);
        if (accel_norm < 0.1f) return;
        
        ax /= accel_norm;
        ay /= accel_norm;
        az /= accel_norm;
        
        // Measured attitude from accelerometer
        float roll_meas = atan2f(ay, az);
        float pitch_meas = atan2f(-ax, sqrtf(ay*ay + az*az));
        
        // Innovation
        float y_roll = roll_meas - state[0];
        float y_pitch = pitch_meas - state[1];
        y_roll = wrapAngle(y_roll);
        
        // Kalman gain (simplified)
        float K_roll = P[0][0] / (P[0][0] + R_accel);
        float K_pitch = P[1][1] / (P[1][1] + R_accel);
        
        // State update
        state[0] += K_roll * y_roll;
        state[1] += K_pitch * y_pitch;
        
        // Covariance update
        P[0][0] *= (1 - K_roll);
        P[1][1] *= (1 - K_pitch);
        
        // Wrap angles
        state[0] = wrapAngle(state[0]);
        state[1] = constrain(state[1], -M_PI_2 + 0.01f, M_PI_2 - 0.01f);
    }
    
    void getRollPitchYaw(float &roll, float &pitch, float &yaw) {
        roll = state[0] * 180.0f / M_PI;
        pitch = state[1] * 180.0f / M_PI;
        yaw = state[2] * 180.0f / M_PI;
    }
    
    void getGyroBias(float &bx, float &by, float &bz) {
        bx = state[3];
        by = state[4];
        bz = state[5];
    }
    
private:
    float wrapAngle(float angle) {
        while (angle > M_PI) angle -= 2 * M_PI;
        while (angle < -M_PI) angle += 2 * M_PI;
        return angle;
    }
};

#endif // ATTITUDE_EKF_H
'''
    return cpp_code


def test_kalman_filter():
    """Test Kalman filter with simulated IMU data."""
    print("Testing Attitude EKF")
    print("=" * 50)
    
    ekf = AttitudeEKF(
        process_noise_gyro=0.001,
        process_noise_bias=0.0001,
        measurement_noise_accel=0.3
    )
    
    cf = ComplementaryFilter(alpha=0.98)
    
    true_roll = 0.0
    true_pitch = 0.3  # ~17 degrees
    gyro_bias = np.array([0.01, -0.02, 0.005])  # rad/s bias
    
    dt = 0.005  # 200 Hz
    duration = 10.0
    
    gyro_noise_std = 0.01
    accel_noise_std = 0.5
    
    ekf_errors = []
    cf_errors = []
    raw_errors = []
    
    raw_roll = 0.0
    
    for t in np.arange(0, duration, dt):
        true_roll = 0.1 * np.sin(2 * np.pi * 0.5 * t)
        true_pitch = 0.3 + 0.05 * np.sin(2 * np.pi * 0.3 * t)
        roll_rate = 0.1 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t)
        pitch_rate = 0.05 * 2 * np.pi * 0.3 * np.cos(2 * np.pi * 0.3 * t)
        
        gyro = np.array([roll_rate, pitch_rate, 0.0]) + gyro_bias
        gyro += np.random.randn(3) * gyro_noise_std
        
        g = 9.81
        accel = np.array([
            -g * np.sin(true_pitch),
            g * np.sin(true_roll) * np.cos(true_pitch),
            g * np.cos(true_roll) * np.cos(true_pitch)
        ])
        accel += np.random.randn(3) * accel_noise_std
        
        measurement = IMUMeasurement(gyro=gyro, accel=accel, timestamp=t)
        ekf_roll, ekf_pitch, _ = ekf.process_imu(measurement)
        
        cf_roll, cf_pitch = cf.update(gyro, accel, dt)
        
        raw_roll += gyro[0] * dt
        
        ekf_errors.append(np.sqrt((ekf_roll - true_roll)**2 + (ekf_pitch - true_pitch)**2))
        cf_errors.append(np.sqrt((cf_roll - true_roll)**2 + (cf_pitch - true_pitch)**2))
        raw_errors.append(abs(raw_roll - true_roll))
    
    print(f"EKF RMS Error: {np.sqrt(np.mean(np.array(ekf_errors)**2))*180/np.pi:.3f} deg")
    print(f"CF RMS Error: {np.sqrt(np.mean(np.array(cf_errors)**2))*180/np.pi:.3f} deg")
    print(f"Raw Gyro RMS Error: {np.sqrt(np.mean(np.array(raw_errors)**2))*180/np.pi:.3f} deg")
    print(f"\nEstimated Gyro Bias: {ekf.get_gyro_bias() * 180/np.pi} deg/s")
    print(f"True Gyro Bias: {gyro_bias * 180/np.pi} deg/s")
    
    return ekf


if __name__ == "__main__":
    test_kalman_filter()
    
    print("\n" + "=" * 50)
    print("Generating C++ implementation...")
    cpp_code = generate_cpp_kalman_filter()
    
    with open("attitude_ekf.h", "w") as f:
        f.write(cpp_code)
    print("Saved to attitude_ekf.h")
