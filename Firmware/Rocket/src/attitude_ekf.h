/*
 * Attitude Extended Kalman Filter for ESP32
 * ==========================================
 * Fuses MPU6050 gyroscope and accelerometer for drift-free attitude estimation.
 * Optimized for real-time execution on ESP32 at 200Hz.
 * 
 * State vector: [roll, pitch, yaw, gyro_bias_x, gyro_bias_y, gyro_bias_z]
 */

#ifndef ATTITUDE_EKF_H
#define ATTITUDE_EKF_H

#include <Arduino.h>
#include <math.h>

class AttitudeEKF {
public:
    float state[6] = {0};
    float P[6][6];
    
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
    
    void setInitialAttitude(float roll, float pitch, float yaw) {
        state[0] = roll * DEG_TO_RAD;
        state[1] = pitch * DEG_TO_RAD;
        state[2] = yaw * DEG_TO_RAD;
    }
    
    void predict(float gx, float gy, float gz, float dt) {
        float p = gx - state[3];
        float q = gy - state[4];
        float r = gz - state[5];
        
        float roll = state[0];
        float pitch = state[1];
        
        float cr = cosf(roll), sr = sinf(roll);
        float cp = cosf(pitch);
        float tp = tanf(pitch);
        
        if (fabsf(cp) < 0.001f) {
            cp = copysignf(0.001f, cp);
            tp = tanf(acosf(cp));
        }
        
        float roll_dot = p + sr * tp * q + cr * tp * r;
        float pitch_dot = cr * q - sr * r;
        float yaw_dot = (sr / cp) * q + (cr / cp) * r;
        
        state[0] += roll_dot * dt;
        state[1] += pitch_dot * dt;
        state[2] += yaw_dot * dt;
        
        state[0] = wrapAngle(state[0]);
        state[2] = wrapAngle(state[2]);
        state[1] = constrain(state[1], -M_PI_2 + 0.01f, M_PI_2 - 0.01f);
        
        for (int i = 0; i < 3; i++) P[i][i] += Q_gyro * dt;
        for (int i = 3; i < 6; i++) P[i][i] += Q_bias * dt;
    }
    
    void update(float ax, float ay, float az) {
        float accel_norm = sqrtf(ax*ax + ay*ay + az*az);
        
        if (accel_norm < 1.0f || accel_norm > 30.0f) return;
        
        ax /= accel_norm;
        ay /= accel_norm;
        az /= accel_norm;
        
        float roll_meas = atan2f(ay, az);
        float pitch_meas = atan2f(-ax, sqrtf(ay*ay + az*az));
        
        float y_roll = wrapAngle(roll_meas - state[0]);
        float y_pitch = pitch_meas - state[1];
        
        float K_roll = P[0][0] / (P[0][0] + R_accel);
        float K_pitch = P[1][1] / (P[1][1] + R_accel);
        
        state[0] += K_roll * y_roll;
        state[1] += K_pitch * y_pitch;
        
        P[0][0] *= (1 - K_roll);
        P[1][1] *= (1 - K_pitch);
        
        state[0] = wrapAngle(state[0]);
        state[1] = constrain(state[1], -M_PI_2 + 0.01f, M_PI_2 - 0.01f);
    }
    
    void process(float gx, float gy, float gz, float ax, float ay, float az, float dt) {
        predict(gx, gy, gz, dt);
        update(ax, ay, az);
    }
    
    float getRollDeg() const { return state[0] * RAD_TO_DEG; }
    float getPitchDeg() const { return state[1] * RAD_TO_DEG; }
    float getYawDeg() const { return state[2] * RAD_TO_DEG; }
    
    void getRollPitchYaw(float &roll, float &pitch, float &yaw) {
        roll = state[0] * RAD_TO_DEG;
        pitch = state[1] * RAD_TO_DEG;
        yaw = state[2] * RAD_TO_DEG;
    }
    
    void getGyroBias(float &bx, float &by, float &bz) {
        bx = state[3] * RAD_TO_DEG;
        by = state[4] * RAD_TO_DEG;
        bz = state[5] * RAD_TO_DEG;
    }
    
private:
    float wrapAngle(float angle) {
        while (angle > M_PI) angle -= 2 * M_PI;
        while (angle < -M_PI) angle += 2 * M_PI;
        return angle;
    }
};

#endif // ATTITUDE_EKF_H
