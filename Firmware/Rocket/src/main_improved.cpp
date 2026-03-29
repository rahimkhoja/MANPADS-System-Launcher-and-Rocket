/*
 * ROCKET ESP32 CODE — V5 (Improved)
 * ==================================
 * Enhancements over V4:
 *   - Extended Kalman Filter for drift-free attitude estimation
 *   - PID roll control with D-term from gyro rate (matches V4 convention)
 *   - Automatic apogee detection → RECOVERY state
 *   - Backward-compatible telemetry (same DATA format as V4)
 *
 * All 4 canards still deflect identically for roll stabilisation.
 * Gains are derived from simulation optimisation on HPC cluster.
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ESP32Servo.h>
#include "attitude_ekf.h"

// ============== PIN DEFINITIONS (matches V4) ==============
const int RX2_PIN = 16;
const int TX2_PIN = 17;
const int IGNITE_SERVO_PIN = 5;

const int LEFT_SERVO_PIN = 26;
const int RIGHT_SERVO_PIN = 25;
const int UP_SERVO_PIN = 27;
const int DOWN_SERVO_PIN = 14;

// ============== SERVO CALIBRATION ==============
const int IGNITE_SERVO_ON = 150;
const int IGNITE_SERVO_OFF = 35;

const int LEFT_CENTER = 115;
const int RIGHT_CENTER = 80;
const int UP_CENTER = 80;
const int DOWN_CENTER = 115;
const int MAX_DEFLECTION = 12;

// ============== ROLL CONTROL GAINS ==============
float Kp = 0.65f;
float Ki = 0.05f;
float Kd = 0.28f;
float integralLimit = 15.0f;

// Optional LiPo sense: GPIO36 + 100k:100k divider to ADC (set -1 to disable)
#ifndef BATTERY_ADC_PIN
#define BATTERY_ADC_PIN 36
#endif

static float readBatteryVoltage() {
#if BATTERY_ADC_PIN >= 0
    int mv = analogReadMilliVolts(BATTERY_ADC_PIN);
    return (mv / 1000.0f) * 2.0f; // 2:1 divider
#else
    return 0.0f;
#endif
}

// ============== HARDWARE ==============
Servo igniteServo;
Servo leftServo, rightServo, upServo, downServo;
Adafruit_MPU6050 mpu;
AttitudeEKF ekf;

// ============== STATE MACHINE ==============
enum SystemState { IDLE, ARMED, IGNITING, FLIGHT, RECOVERY };
SystemState sysState = IDLE;

String cmdBuffer = "";

// Control variables
float rollIntegral = 0;
int lastServoOffset = 0;
float physical_skew_angle = 0.0f;

// Timing
unsigned long lastTime;
unsigned long lastTelemetrySent = 0;
unsigned long lastReadySent = 0;
unsigned long igniteStartTime = 0;
unsigned long flightStartTime = 0;

// Apogee detection
float peakAltitudeAccel = 0.0f;
unsigned long lowGStartTime = 0;
const unsigned long APOGEE_HOLD_MS = 500;
const float LOW_G_THRESHOLD = 4.9f; // m/s^2 (~0.5 g)

// Telemetry mode: false = V4 compatible, true = extended V5
bool extendedTelemetry = false;

// ============== FORWARD DECLARATIONS ==============
void calibrateSensors();
void resetController();
void setAllCanards(int offset);

// ============== SETUP ==============
void setup() {
    Serial.begin(115200);
    Serial2.begin(115200, SERIAL_8N1, RX2_PIN, TX2_PIN);
    Serial2.setTimeout(20);
    delay(1500);

    Wire.begin(21, 22);
    Wire.setClock(400000);

    if (!mpu.begin()) {
        Serial.println("MPU6050 not found!");
        while (1) delay(100);
    }

    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

#if BATTERY_ADC_PIN >= 0
    pinMode(BATTERY_ADC_PIN, INPUT);
#endif

    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);

    igniteServo.setPeriodHertz(50);
    igniteServo.attach(IGNITE_SERVO_PIN);
    igniteServo.write(IGNITE_SERVO_OFF);

    leftServo.setPeriodHertz(50);   leftServo.attach(LEFT_SERVO_PIN, 500, 2400);
    rightServo.setPeriodHertz(50);  rightServo.attach(RIGHT_SERVO_PIN, 500, 2400);
    upServo.setPeriodHertz(50);     upServo.attach(UP_SERVO_PIN, 500, 2400);
    downServo.setPeriodHertz(50);   downServo.attach(DOWN_SERVO_PIN, 500, 2400);

    setAllCanards(0);
    calibrateSensors();
    ekf.reset();

    lastTime = micros();
    Serial.println("Rocket V5 Ready — EKF + Roll PID");
}

// ============== MAIN LOOP ==============
void loop() {
    unsigned long now = micros();
    float dt = (now - lastTime) / 1000000.0f;
    if (dt <= 0 || dt > 0.1f) { lastTime = now; return; }
    lastTime = now;

    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);

    ekf.process(
        gyro.gyro.x, gyro.gyro.y, gyro.gyro.z,
        accel.acceleration.x, accel.acceleration.y, accel.acceleration.z,
        dt
    );

    float rollDeg = ekf.getRollDeg();
    float rollRateDeg = (gyro.gyro.x - ekf.getGyroBiasX()) * RAD_TO_DEG;

    switch (sysState) {
    case IDLE:
    case ARMED:
        setAllCanards(0);
        break;

    case IGNITING:
        if (millis() - igniteStartTime > 2500) {
            igniteServo.write(IGNITE_SERVO_OFF);
            sysState = FLIGHT;
            flightStartTime = millis();
            resetController();
            Serial2.println("IGNITED");
        }
        break;

    case FLIGHT: {
        // PID: P on angle, I on angle, D on gyro rate (not d(error)/dt)
        rollIntegral += rollDeg * dt;
        rollIntegral = constrain(rollIntegral, -integralLimit, integralLimit);

        float output = Kp * rollDeg + Ki * rollIntegral + Kd * rollRateDeg;
        int offset = constrain((int)output, -MAX_DEFLECTION, MAX_DEFLECTION);
        setAllCanards(offset);
        lastServoOffset = offset;

        // Apogee detection: axial accel < 0.5 g for > 500 ms
        float axialAccel = accel.acceleration.x;  // along body x-axis
        if (fabsf(axialAccel) < LOW_G_THRESHOLD) {
            if (lowGStartTime == 0) lowGStartTime = millis();
            if (millis() - lowGStartTime > APOGEE_HOLD_MS) {
                sysState = RECOVERY;
                setAllCanards(0);
                Serial2.println("RECOVERY");
            }
        } else {
            lowGStartTime = 0;
        }
        break;
    }

    case RECOVERY:
        setAllCanards(0);
        break;
    }

    // Telemetry at 20 Hz
    if (millis() - lastTelemetrySent >= 50) {
        if (extendedTelemetry) {
            // V5 extended: DATA,roll,pitch,yaw,offset,state,Kp,Kd,biasX
            float pitch = ekf.getPitchDeg();
            float yaw = ekf.getYawDeg();
            float bx = ekf.getGyroBiasX();
            const char* st = (sysState == IDLE) ? "IDLE" :
                             (sysState == ARMED) ? "ARMED" :
                             (sysState == IGNITING) ? "IGNITING" :
                             (sysState == FLIGHT) ? "FLIGHT" : "RECOVERY";
            char buf[128];
            float vb = readBatteryVoltage();
            snprintf(buf, sizeof(buf),
                "DATA,%.2f,%.2f,%.2f,%d,%s,%.2f,%.2f,%.3f,%.2f",
                rollDeg, pitch, yaw, lastServoOffset, st, Kp, Kd, bx, vb);
            Serial2.println(buf);
        } else {
            // V4 compatible: DATA,ax,ay,az,roll,rate,offset,state,Kp,Kd,skew
            char buf[160];
            snprintf(buf, sizeof(buf),
                "DATA,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%s,%.2f,%.2f,%.2f",
                accel.acceleration.x, accel.acceleration.y, accel.acceleration.z,
                rollDeg, rollRateDeg, lastServoOffset,
                (sysState == IDLE) ? "IDLE" :
                (sysState == ARMED) ? "ARMED" :
                (sysState == IGNITING) ? "IGNITING" :
                (sysState == FLIGHT) ? "FLIGHT" : "RECOVERY",
                Kp, Kd, physical_skew_angle);
            Serial2.println(buf);
        }
        lastTelemetrySent = millis();
    }

    // Heartbeat at 1 Hz
    if (sysState == IDLE && millis() - lastReadySent >= 1000) {
        Serial2.println("READY");
        lastReadySent = millis();
    }

    // Command processing
    while (Serial2.available()) {
        char c = Serial2.read();
        if (c == '\n') {
            cmdBuffer.trim();

            if (cmdBuffer == "ARM" && sysState == IDLE) {
                sysState = ARMED;
                resetController();
                Serial2.println("ACK:ARMED");
            }
            else if (cmdBuffer == "DISARM") {
                sysState = IDLE;
                Serial2.println("ACK:DISARMED");
            }
            else if (cmdBuffer == "IGNITE" && sysState == ARMED) {
                sysState = IGNITING;
                igniteStartTime = millis();
                igniteServo.write(IGNITE_SERVO_ON);
                Serial2.println("ACK:IGNITING");
            }
            else if (cmdBuffer == "CALIBRATE") {
                resetController();
                Serial2.println("ACK:CALIBRATED");
            }
            else if (cmdBuffer.startsWith("PID,")) {
                // PID,Kp,Kd  (V4 compat, Ki=0)  or  PID,Kp,Ki,Kd
                int c1 = cmdBuffer.indexOf(',');
                int c2 = cmdBuffer.indexOf(',', c1 + 1);
                int c3 = cmdBuffer.indexOf(',', c2 + 1);
                if (c1 > 0 && c2 > 0) {
                    Kp = cmdBuffer.substring(c1 + 1, c2).toFloat();
                    if (c3 > 0) {
                        Ki = cmdBuffer.substring(c2 + 1, c3).toFloat();
                        Kd = cmdBuffer.substring(c3 + 1).toFloat();
                    } else {
                        Ki = 0.0f;
                        Kd = cmdBuffer.substring(c2 + 1).toFloat();
                    }
                    Serial2.println("ACK:PID_SET");
                }
            }
            else if (cmdBuffer == "TELEMETRY_V5") {
                extendedTelemetry = true;
                Serial2.println("ACK:TELEMETRY_V5");
            }
            else if (cmdBuffer == "TELEMETRY_V4") {
                extendedTelemetry = false;
                Serial2.println("ACK:TELEMETRY_V4");
            }
            cmdBuffer = "";
        } else if (c != '\r') {
            cmdBuffer += c;
        }
    }
}

// ============== HELPERS ==============
void setAllCanards(int offset) {
    offset = constrain(offset, -MAX_DEFLECTION, MAX_DEFLECTION);
    leftServo.write(LEFT_CENTER + offset);
    rightServo.write(RIGHT_CENTER + offset);
    upServo.write(UP_CENTER + offset);
    downServo.write(DOWN_CENTER + offset);
}

void calibrateSensors() {
    Serial.println("Calibrating... keep still");
    delay(500);

    float sumAx = 0, sumAy = 0, sumAz = 0;
    const int samples = 200;

    for (int i = 0; i < samples; i++) {
        sensors_event_t a, g, t;
        mpu.getEvent(&a, &g, &t);
        sumAx += a.acceleration.x;
        sumAy += a.acceleration.y;
        sumAz += a.acceleration.z;
        delay(5);
    }

    float ay = sumAy / samples;
    float az = sumAz / samples;
    float ax = sumAx / samples;

    float initRoll = atan2(ay, az) * RAD_TO_DEG;
    float initPitch = atan2(-ax, sqrt(ay*ay + az*az)) * RAD_TO_DEG;
    physical_skew_angle = initRoll;

    ekf.setInitialAttitude(initRoll, initPitch, 0);
    Serial.printf("Calibrated — Roll: %.1f  Pitch: %.1f\n", initRoll, initPitch);
}

void resetController() {
    rollIntegral = 0;
    lastServoOffset = 0;
    lowGStartTime = 0;
    ekf.reset();
    calibrateSensors();
}
