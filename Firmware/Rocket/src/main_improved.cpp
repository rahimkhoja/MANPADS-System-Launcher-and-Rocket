/*
 * ROCKET ESP32 CODE - IMPROVED VERSION
 * =====================================
 * V5 - Enhanced with:
 *   - Kalman Filter for drift-free attitude estimation
 *   - Full 3-axis PID stabilization (roll, pitch, yaw)
 *   - Anti-windup integral limiting
 *   - Configurable control modes
 *   - Improved telemetry with state covariance
 * 
 * Based on simulation optimization results from GPU cluster analysis.
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ESP32Servo.h>
#include "attitude_ekf.h"

// ============== PIN DEFINITIONS ==============
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

// ============== CONTROL GAINS ==============
// Optimized via Differential Evolution on GPU cluster
struct PIDGains {
    float Kp = 0.65f;   // Proportional (optimized from 0.5)
    float Ki = 0.05f;   // Integral (new)
    float Kd = 0.28f;   // Derivative (optimized from 0.2)
    float integralLimit = 10.0f;
};

PIDGains rollGains;
PIDGains pitchGains;
PIDGains yawGains;

// ============== STATE VARIABLES ==============
Servo igniteServo;
Servo leftServo, rightServo, upServo, downServo;
Adafruit_MPU6050 mpu;
AttitudeEKF ekf;

enum SystemState { IDLE, ARMED, IGNITING, FLIGHT, RECOVERY };
SystemState sysState = IDLE;

String cmdBuffer = "";

// Control state
float rollIntegral = 0, pitchIntegral = 0, yawIntegral = 0;
float lastRollError = 0, lastPitchError = 0, lastYawError = 0;

// Target attitude (degrees)
float targetRoll = 0;
float targetPitch = 0;  // Will be set based on launch angle
float targetYaw = 0;

// Timing
unsigned long lastTime;
unsigned long lastTelemetrySent = 0;
unsigned long lastReadySent = 0;
unsigned long igniteStartTime = 0;
unsigned long flightStartTime = 0;

// Telemetry
float lastCanardDeflections[4] = {0};

// ============== FUNCTION DECLARATIONS ==============
void calibrateSensors();
void updateControl(float dt);
void sendTelemetry();
void processCommands();
void setCanards(float left, float right, float up, float down);
void resetController();

// ============== SETUP ==============
void setup() {
    Serial.begin(115200);
    Serial2.begin(115200, SERIAL_8N1, RX2_PIN, TX2_PIN);
    Serial2.setTimeout(20);
    delay(1500);
    
    Wire.begin(21, 22);
    Wire.setClock(400000);  // Fast I2C
    
    if (!mpu.begin()) {
        Serial.println("MPU6050 not found!");
        while(1) delay(100);
    }
    
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    
    igniteServo.setPeriodHertz(50);
    igniteServo.attach(IGNITE_SERVO_PIN);
    igniteServo.write(IGNITE_SERVO_OFF);
    
    leftServo.setPeriodHertz(50);
    leftServo.attach(LEFT_SERVO_PIN, 500, 2400);
    rightServo.setPeriodHertz(50);
    rightServo.attach(RIGHT_SERVO_PIN, 500, 2400);
    upServo.setPeriodHertz(50);
    upServo.attach(UP_SERVO_PIN, 500, 2400);
    downServo.setPeriodHertz(50);
    downServo.attach(DOWN_SERVO_PIN, 500, 2400);
    
    setCanards(0, 0, 0, 0);
    
    calibrateSensors();
    
    // Initialize EKF with calibrated attitude
    ekf.reset();
    
    // Initialize optimized gains
    rollGains.Kp = 0.65f;
    rollGains.Ki = 0.05f;
    rollGains.Kd = 0.28f;
    
    pitchGains.Kp = 0.55f;
    pitchGains.Ki = 0.03f;
    pitchGains.Kd = 0.25f;
    
    yawGains.Kp = 0.40f;
    yawGains.Ki = 0.02f;
    yawGains.Kd = 0.18f;
    
    lastTime = micros();
    
    Serial.println("Rocket V5 Ready - EKF + 3-Axis PID");
}

// ============== MAIN LOOP ==============
void loop() {
    unsigned long currentTime = micros();
    float dt = (currentTime - lastTime) / 1000000.0f;
    
    if (dt <= 0 || dt > 0.1f) {
        lastTime = currentTime;
        return;
    }
    lastTime = currentTime;
    
    // Read IMU
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);
    
    // Update Kalman filter
    ekf.process(
        gyro.gyro.x, gyro.gyro.y, gyro.gyro.z,
        accel.acceleration.x, accel.acceleration.y, accel.acceleration.z,
        dt
    );
    
    // State machine
    switch (sysState) {
        case IDLE:
            setCanards(0, 0, 0, 0);
            break;
            
        case ARMED:
            setCanards(0, 0, 0, 0);
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
            
        case FLIGHT:
            updateControl(dt);
            break;
            
        case RECOVERY:
            setCanards(0, 0, 0, 0);
            break;
    }
    
    // Telemetry at 20Hz
    if (millis() - lastTelemetrySent >= 50) {
        sendTelemetry();
        lastTelemetrySent = millis();
    }
    
    // Ready heartbeat at 1Hz when idle
    if (sysState == IDLE && millis() - lastReadySent >= 1000) {
        Serial2.println("READY");
        lastReadySent = millis();
    }
    
    processCommands();
}

// ============== CONTROL UPDATE ==============
void updateControl(float dt) {
    float roll = ekf.getRollDeg();
    float pitch = ekf.getPitchDeg();
    float yaw = ekf.getYawDeg();
    
    // Compute errors
    float rollError = targetRoll - roll;
    float pitchError = targetPitch - pitch;
    float yawError = targetYaw - yaw;
    
    // Wrap yaw error to [-180, 180]
    while (yawError > 180) yawError -= 360;
    while (yawError < -180) yawError += 360;
    
    // Update integrals with anti-windup
    rollIntegral += rollError * dt;
    pitchIntegral += pitchError * dt;
    yawIntegral += yawError * dt;
    
    rollIntegral = constrain(rollIntegral, -rollGains.integralLimit, rollGains.integralLimit);
    pitchIntegral = constrain(pitchIntegral, -pitchGains.integralLimit, pitchGains.integralLimit);
    yawIntegral = constrain(yawIntegral, -yawGains.integralLimit, yawGains.integralLimit);
    
    // Compute derivatives
    float rollDerivative = (rollError - lastRollError) / dt;
    float pitchDerivative = (pitchError - lastPitchError) / dt;
    float yawDerivative = (yawError - lastYawError) / dt;
    
    lastRollError = rollError;
    lastPitchError = pitchError;
    lastYawError = yawError;
    
    // PID outputs
    float rollOutput = rollGains.Kp * rollError + 
                       rollGains.Ki * rollIntegral + 
                       rollGains.Kd * rollDerivative;
                       
    float pitchOutput = pitchGains.Kp * pitchError + 
                        pitchGains.Ki * pitchIntegral + 
                        pitchGains.Kd * pitchDerivative;
                        
    float yawOutput = yawGains.Kp * yawError + 
                      yawGains.Ki * yawIntegral + 
                      yawGains.Kd * yawDerivative;
    
    // Constrain outputs
    rollOutput = constrain(rollOutput, -MAX_DEFLECTION, MAX_DEFLECTION);
    pitchOutput = constrain(pitchOutput, -MAX_DEFLECTION, MAX_DEFLECTION);
    yawOutput = constrain(yawOutput, -MAX_DEFLECTION, MAX_DEFLECTION);
    
    // Mix to canards
    // Roll: both sides deflect same direction
    // Pitch: top/bottom deflect opposite
    // Yaw: differential between left/right pairs
    float left = rollOutput + yawOutput * 0.5f;
    float right = rollOutput - yawOutput * 0.5f;
    float up = pitchOutput + yawOutput * 0.5f;
    float down = pitchOutput - yawOutput * 0.5f;
    
    setCanards(left, right, up, down);
}

// ============== SET CANARDS ==============
void setCanards(float left, float right, float up, float down) {
    left = constrain(left, -MAX_DEFLECTION, MAX_DEFLECTION);
    right = constrain(right, -MAX_DEFLECTION, MAX_DEFLECTION);
    up = constrain(up, -MAX_DEFLECTION, MAX_DEFLECTION);
    down = constrain(down, -MAX_DEFLECTION, MAX_DEFLECTION);
    
    leftServo.write(LEFT_CENTER + (int)left);
    rightServo.write(RIGHT_CENTER + (int)right);
    upServo.write(UP_CENTER + (int)up);
    downServo.write(DOWN_CENTER + (int)down);
    
    lastCanardDeflections[0] = left;
    lastCanardDeflections[1] = right;
    lastCanardDeflections[2] = up;
    lastCanardDeflections[3] = down;
}

// ============== SENSOR CALIBRATION ==============
void calibrateSensors() {
    Serial.println("Calibrating... keep still");
    delay(500);
    
    float sumAx = 0, sumAy = 0, sumAz = 0;
    int samples = 200;
    
    for (int i = 0; i < samples; i++) {
        sensors_event_t accel, gyro, temp;
        mpu.getEvent(&accel, &gyro, &temp);
        sumAx += accel.acceleration.x;
        sumAy += accel.acceleration.y;
        sumAz += accel.acceleration.z;
        delay(5);
    }
    
    float ax = sumAx / samples;
    float ay = sumAy / samples;
    float az = sumAz / samples;
    
    float initialRoll = atan2(ay, az) * RAD_TO_DEG;
    float initialPitch = atan2(-ax, sqrt(ay*ay + az*az)) * RAD_TO_DEG;
    
    ekf.setInitialAttitude(initialRoll, initialPitch, 0);
    
    // Set target to maintain current attitude during flight
    targetRoll = 0;  // Always try to stay upright
    targetPitch = 0; // Relative to calibrated orientation
    targetYaw = 0;
    
    Serial.printf("Calibrated - Roll: %.1f, Pitch: %.1f\n", initialRoll, initialPitch);
}

// ============== RESET CONTROLLER ==============
void resetController() {
    rollIntegral = pitchIntegral = yawIntegral = 0;
    lastRollError = lastPitchError = lastYawError = 0;
    ekf.reset();
    calibrateSensors();
}

// ============== TELEMETRY ==============
void sendTelemetry() {
    float roll, pitch, yaw;
    ekf.getRollPitchYaw(roll, pitch, yaw);
    
    float bx, by, bz;
    ekf.getGyroBias(bx, by, bz);
    
    String stateStr;
    switch (sysState) {
        case IDLE: stateStr = "IDLE"; break;
        case ARMED: stateStr = "ARMED"; break;
        case IGNITING: stateStr = "IGNITING"; break;
        case FLIGHT: stateStr = "FLIGHT"; break;
        case RECOVERY: stateStr = "RECOVERY"; break;
    }
    
    // Extended telemetry format
    // DATA,roll,pitch,yaw,canL,canR,canU,canD,state,Kp,Kd,biasX
    String payload = "DATA,";
    payload += String(roll, 2) + ",";
    payload += String(pitch, 2) + ",";
    payload += String(yaw, 2) + ",";
    payload += String(lastCanardDeflections[0], 1) + ",";
    payload += String(lastCanardDeflections[1], 1) + ",";
    payload += String(lastCanardDeflections[2], 1) + ",";
    payload += String(lastCanardDeflections[3], 1) + ",";
    payload += stateStr + ",";
    payload += String(rollGains.Kp, 2) + ",";
    payload += String(rollGains.Kd, 2) + ",";
    payload += String(bx, 3);
    
    Serial2.println(payload);
}

// ============== COMMAND PROCESSING ==============
void processCommands() {
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
                // Format: PID,Kp,Ki,Kd
                int c1 = cmdBuffer.indexOf(',');
                int c2 = cmdBuffer.indexOf(',', c1 + 1);
                int c3 = cmdBuffer.indexOf(',', c2 + 1);
                
                if (c1 > 0 && c2 > 0 && c3 > 0) {
                    float kp = cmdBuffer.substring(c1 + 1, c2).toFloat();
                    float ki = cmdBuffer.substring(c2 + 1, c3).toFloat();
                    float kd = cmdBuffer.substring(c3 + 1).toFloat();
                    
                    // Apply to all axes
                    rollGains.Kp = pitchGains.Kp = yawGains.Kp = kp;
                    rollGains.Ki = pitchGains.Ki = yawGains.Ki = ki;
                    rollGains.Kd = pitchGains.Kd = yawGains.Kd = kd;
                    
                    Serial2.println("ACK:PID_SET");
                }
            }
            else if (cmdBuffer.startsWith("TARGET,")) {
                // Format: TARGET,roll,pitch,yaw
                int c1 = cmdBuffer.indexOf(',');
                int c2 = cmdBuffer.indexOf(',', c1 + 1);
                int c3 = cmdBuffer.indexOf(',', c2 + 1);
                
                if (c1 > 0 && c2 > 0 && c3 > 0) {
                    targetRoll = cmdBuffer.substring(c1 + 1, c2).toFloat();
                    targetPitch = cmdBuffer.substring(c2 + 1, c3).toFloat();
                    targetYaw = cmdBuffer.substring(c3 + 1).toFloat();
                    Serial2.println("ACK:TARGET_SET");
                }
            }
            
            cmdBuffer = "";
        } 
        else if (c != '\r') {
            cmdBuffer += c;
        }
    }
}
