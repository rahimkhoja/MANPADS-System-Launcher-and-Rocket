# Firmware Improvements: V4 → V5

This document describes the changes made to the rocket flight computer firmware.

---

## Summary of Changes

| Feature                | V4 (Original)               | V5 (Improved)                    |
|------------------------|------------------------------|----------------------------------|
| Attitude estimation    | Raw gyro integration         | Extended Kalman Filter (EKF)     |
| Drift compensation     | None                         | Online gyro bias estimation      |
| Control axes           | Roll only                    | Roll only (PID, same structure)  |
| D-term source          | Gyro rate                    | Gyro rate (minus estimated bias) |
| Integral term          | None                         | Optional Ki with anti-windup     |
| Default gains          | Kp=0.5, Kd=0.2              | Kp=0.65, Ki=0.05, Kd=0.28       |
| Flight phase detection | Manual state machine         | Automatic apogee detection       |
| Telemetry format       | Fixed V4 format              | V4 (default) or V5 extended     |
| I2C clock              | Default (100 kHz)            | 400 kHz Fast Mode               |

---

## Extended Kalman Filter

The EKF fuses gyroscope and accelerometer data to produce drift-free attitude
estimates. It runs at the full IMU rate (~200 Hz) and estimates:

- Roll, pitch, yaw angles (quaternion internally)
- Gyroscope bias on all three axes

### Why it matters
The MPU6050 gyroscope has a typical bias instability of ~0.01°/s. With raw
integration (V4), this produces ~0.6°/minute of drift. Over a 10-second flight
this is small, but the EKF also filters high-frequency noise and provides
pitch/yaw estimates needed for future 3-axis control.

### Implementation
The EKF is implemented in `Firmware/Rocket/src/attitude_ekf.h` as a single header
(included by `main_improved.cpp`). `getGyroBiasX()` exposes the X gyro bias in
deg/s for telemetry and the D-term.

---

## PID Control

V5 keeps the same roll-only control architecture as V4: all four canards
deflect by the same amount. The PID output is:

```
output = Kp × roll + Ki × ∫roll + Kd × roll_rate
```

The D-term uses the gyro rate directly (with bias removed by the EKF), not
`d(error)/dt`. This avoids amplifying sensor noise through numerical
differentiation — a deliberate design choice inherited from V4.

### Gain Derivation
Default gains (Kp=0.65, Ki=0.05, Kd=0.28) were found via simulation-based search.
See `docs/SIMULATION.md` and `Simulation/results/best_gains.json` for cluster runs.

### Serial commands
- **`PID,Kp,Kd`** — V4-compatible (sets Ki=0).
- **`PID,Kp,Ki,Kd`** — full three-term set (V5).

### Battery telemetry (optional)
GPIO **36** (ADC1) with a 100k:100k divider can report bus voltage in extended
telemetry (`readBatteryVoltage()`). Define `BATTERY_ADC_PIN` to `-1` to disable.

---

## Apogee Detection

V5 adds automatic transition to RECOVERY state:

1. During FLIGHT, monitor axial accelerometer reading
2. If |accel_x| < 0.5g for 500 ms continuously → declare apogee
3. Set state to RECOVERY, centre all canards
4. Send `RECOVERY` message to launcher

This replaces the V4 behaviour where the rocket stays in FLIGHT until power-off.

---

## Telemetry

### V4 Compatible Mode (default)
```
DATA,ax,ay,az,roll,rate,offset,state,Kp,Kd,skew
```
This is identical to V4 and works with the existing launcher firmware and
dashboard without modification.

### V5 Extended Mode
```
DATA,roll,pitch,yaw,offset,state,Kp,Kd,biasX,Vbatt
```
(`Vbatt` in volts; 0 if ADC disabled.)
Activated by sending `TELEMETRY_V5` command. Returns to V4 format with
`TELEMETRY_V4`.

---

## Backward Compatibility

V5 is a drop-in replacement for V4:
- Same pin assignments
- Same serial protocol
- Same default telemetry format
- Same command set (ARM, IGNITE, CALIBRATE, PID with optional Ki)
- Added commands (DISARM, TELEMETRY_V5, TELEMETRY_V4) are additive

The launcher firmware does not need to be modified.

---

## Building

The firmware is a PlatformIO project:

```bash
cd Firmware/Rocket
pio run                    # Compile
pio run --target upload    # Flash to ESP32
pio device monitor         # Serial monitor
```

Requires: Arduino framework, Adafruit MPU6050, ESP32Servo libraries.
