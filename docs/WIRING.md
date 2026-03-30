# Wiring Reference

Pin assignments verified against the firmware source code (V4/V5).

---

## Rocket Flight Computer (ESP32 DevKit V1)

### Servo Connections
| Function       | GPIO Pin | Notes          |
|----------------|----------|----------------|
| Left Canard    | 26       | Left side      |
| Right Canard   | 25       | Right side     |
| Up Canard      | 27       | Top            |
| Down Canard    | 14       | Bottom         |
| Ignition Servo | 5        | Motor bay      |

### Servo Center Calibration
| Servo | Center Angle | Max Deflection |
|-------|-------------|----------------|
| Left  | 115°        | ±12°           |
| Right | 80°         | ±12°           |
| Up    | 80°         | ±12°           |
| Down  | 115°        | ±12°           |

### I2C Bus (MPU6050)
| Signal | GPIO Pin |
|--------|----------|
| SDA    | 21       |
| SCL    | 22       |

### UART to Launcher (Serial2)
| Signal | GPIO Pin | Baud Rate |
|--------|----------|-----------|
| TX2    | 17       | 115200    |
| RX2    | 16       | 115200    |

---

## Launcher Ground Station (ESP32 DevKit V1)

### I2C Bus (QMC5883L + BMP180)
| Signal | GPIO Pin |
|--------|----------|
| SDA    | 21       |
| SCL    | 22       |

### I2C Device Addresses
| Device    | Address | Notes |
|-----------|---------|-------|
| QMC5883L  | 0x0D    | Compass |
| BMP180    | 0x77    | Barometer / altimeter |

### GPS Module (Serial1)
| Signal | GPIO Pin | Baud Rate |
|--------|----------|-----------|
| RX1    | 4        | 9600      |

> GPS TX connects to ESP32 RX1 (GPIO 4). Pin 4 supports input on ESP32.

### UART to Rocket (Serial2)
| Signal | GPIO Pin | Baud Rate |
|--------|----------|-----------|
| TX2    | 17       | 115200    |
| RX2    | 16       | 115200    |

### Control Switches & Indicators
| Function        | GPIO Pin | Type              | Notes                    |
|-----------------|----------|-------------------|--------------------------|
| Arm Switch      | 5        | Toggle Switch     | INPUT_PULLUP, active LOW |
| Launch Button   | 18       | Momentary (N.O.)  | INPUT_PULLUP, active LOW |
| Status LED      | 23       | LED               | Digital output           |
| Buzzer          | 2        | Active Buzzer     | LOW-trigger              |
| Launcher Servo  | 19       | SG90 Micro Servo  | Launcher-side ignition   |

### WiFi Configuration
| Parameter  | Value                |
|------------|----------------------|
| Mode       | SoftAP (Access Point)|
| SSID       | `ROCKET_LAUNCHER`    |
| Password   | `launch_secure` (matches `Firmware/Launcher/src/main.cpp`) |
| IP Address | 192.168.4.1          |
| UDP Port   | 4444                 |

---

## Telemetry Protocol

### V4 Format (default)
| Direction       | Format                                                           |
|-----------------|------------------------------------------------------------------|
| Rocket → Launcher | `DATA,ax,ay,az,roll,rate,offset,state,Kp,Kd,skew`            |
| Launcher → PC   | `ENV,lat,lon,alt,gps_state`                                     |
| PC → Rocket     | `PID,Kp,Kd`                                                     |

### V5 Extended Format (opt-in via `TELEMETRY_V5` command)
| Direction       | Format                                                           |
|-----------------|------------------------------------------------------------------|
| Rocket → Launcher | `DATA,roll,pitch,yaw,offset,state,Kp,Ki,Kd,biasX,Vbatt`    |

### Commands
| Command         | Description                            |
|-----------------|----------------------------------------|
| `ARM`           | Arm the system (IDLE → ARMED)          |
| `DISARM`        | Disarm (any → IDLE) — V5 only         |
| `IGNITE`        | Fire ignition servo (ARMED → IGNITING) |
| `CALIBRATE`     | Re-calibrate IMU                       |
| `PID,Kp,Kd`    | Set PID gains (V4, Ki=0)               |
| `PID,Kp,Ki,Kd` | Set PID gains with integral term (V5)  |
| `TELEMETRY_V5`  | Switch to extended format — V5 only    |
| `TELEMETRY_V4`  | Switch back to V4 format — V5 only     |

---

## Changelog

- **2026-03**: Corrected pin assignments to match firmware source (Ignition Servo: 13→5,
  Arm Switch: 15→5, Launch Button: 4→18, LED: 2→23, Buzzer: 5→2, Launcher Servo: 13→19,
  GPS RX: 34→4). Fixed max deflection: ±15°→±12°. Added V5 telemetry format.
