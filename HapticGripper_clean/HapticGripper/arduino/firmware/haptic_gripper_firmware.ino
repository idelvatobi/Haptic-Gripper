// ============================================================
// Haptic Gripper — Combined Script with Logarithmic Force Mapping
// ============================================================
//
// Connections:
// FSR                -> A0
// Potentiometer      -> A1
// Feedback servo     -> D9
// Slave servo        -> D10
// ============================================================

#include <Servo.h>
#include <math.h>

const int FORCE_PIN = A0;
const int POT_PIN   = A1;
const int FEEDBACK_SERVO_PIN = 9;
const int SLAVE_SERVO_PIN    = 10;

Servo feedbackServo;
Servo slaveServo;

const int SAMPLE_RATE = 50;

// Moving Average Filter
const int MA_WINDOW = 5;
int ma_buffer[MA_WINDOW];
int ma_index = 0;
bool ma_ready = false;

void ma_add(int value) {
  ma_buffer[ma_index] = value;
  ma_index = (ma_index + 1) % MA_WINDOW;
  if (ma_index == 0) ma_ready = true;
}

float ma_get() {
  int count = ma_ready ? MA_WINDOW : ma_index;
  if (count == 0) return 0;
  long sum = 0;
  for (int i = 0; i < count; i++) sum += ma_buffer[i];
  return (float)sum / count;
}

// Logarithmic Force Mapping
const int   FSR_MIN           = 0;
const int   FSR_MAX           = 680;
const int   FEEDBACK_MIN_ANGLE = 30;
const int   FEEDBACK_MAX_ANGLE = 150;
const float LOG_GAIN          = 10.0;
const int   FSR_DEADZONE      = 20;
const int   SERVO_STEP        = 3;

int feedbackTargetAngle  = FEEDBACK_MIN_ANGLE;
int feedbackCurrentAngle = FEEDBACK_MIN_ANGLE;

int force_to_log_angle(float fsrValue) {
  if (fsrValue < FSR_DEADZONE) return FEEDBACK_MIN_ANGLE;
  fsrValue = constrain(fsrValue, FSR_MIN, FSR_MAX);
  float normalized = (fsrValue - FSR_MIN) / (float)(FSR_MAX - FSR_MIN);
  float logMapped  = log(1.0 + LOG_GAIN * normalized) / log(1.0 + LOG_GAIN);
  int angle = FEEDBACK_MIN_ANGLE + logMapped * (FEEDBACK_MAX_ANGLE - FEEDBACK_MIN_ANGLE);
  return constrain(angle, FEEDBACK_MIN_ANGLE, FEEDBACK_MAX_ANGLE);
}

void move_feedback_servo_smoothly(int target) {
  if (feedbackCurrentAngle < target) {
    feedbackCurrentAngle += SERVO_STEP;
    if (feedbackCurrentAngle > target) feedbackCurrentAngle = target;
  } else if (feedbackCurrentAngle > target) {
    feedbackCurrentAngle -= SERVO_STEP;
    if (feedbackCurrentAngle < target) feedbackCurrentAngle = target;
  }
  feedbackServo.write(feedbackCurrentAngle);
}

// Potentiometer + Slave Servo
int potValue  = 0;
int slaveAngle = 90;
const int SLAVE_MIN_ANGLE = 20;
const int SLAVE_MAX_ANGLE = 160;

void setup() {
  Serial.begin(9600);
  feedbackServo.attach(FEEDBACK_SERVO_PIN);
  slaveServo.attach(SLAVE_SERVO_PIN);
  feedbackServo.write(feedbackCurrentAngle);
  slaveServo.write(90);
  delay(1000);
  Serial.println("# === HAPTIC GRIPPER: LOG FORCE MAPPING + POT CONTROL ===");
  Serial.println("# FSR -> A0 | Potentiometer -> A1");
  Serial.println("# Feedback Servo -> D9 | Slave Servo -> D10");
  Serial.println("timestamp_ms,fsr_raw,fsr_smoothed,feedback_target,feedback_current,pot_value,slave_angle");
}

void loop() {
  // 1. FSR + Feedback Servo
  int fsrRaw = analogRead(FORCE_PIN);
  ma_add(fsrRaw);
  float fsrSmoothed = ma_get();
  feedbackTargetAngle = force_to_log_angle(fsrSmoothed);
  move_feedback_servo_smoothly(feedbackTargetAngle);

  // 2. Potentiometer + Slave Servo
  potValue  = analogRead(POT_PIN);
  slaveAngle = map(potValue, 0, 1023, SLAVE_MIN_ANGLE, SLAVE_MAX_ANGLE);
  slaveServo.write(slaveAngle);

  // Serial Output
  Serial.print(millis());             Serial.print(",");
  Serial.print(fsrRaw);               Serial.print(",");
  Serial.print(fsrSmoothed, 1);       Serial.print(",");
  Serial.print(feedbackTargetAngle);  Serial.print(",");
  Serial.print(feedbackCurrentAngle); Serial.print(",");
  Serial.print(potValue);             Serial.print(",");
  Serial.println(slaveAngle);

  delay(SAMPLE_RATE);
}
