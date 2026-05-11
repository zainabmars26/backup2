#include <Servo.h>

Servo myservo;
int pos = 90; // Start at center
String currentMode = "scan";
int frame_center_x = 640;
int step = 1; // Direction of scan

void setup() {
  Serial.begin(115200);
  myservo.attach(9);
  myservo.write(pos);
}

void loop() {
  checkSerial(); // Update mode and target data

  if (currentMode == "scan") {
    scanObject();
  } 
  // If mode is "track", the checkSerial function will call track()
}

void checkSerial() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim();

    if (data.startsWith("GO")) {
      currentMode = "scan";
    } 
    else if (data.startsWith("Stop")) {
      currentMode = "track";
      // Expecting format "Stop:XXXX" where XXXX is center X
      if (data.length() > 5) {
        int targetX = data.substring(5).toInt();
        track(targetX);
      }
    }
  }
}

void scanObject() {
  static unsigned long lastMove = 0;
  if (millis() - lastMove > 45) { // Non-blocking delay
    pos += step;
    if (pos >= 170 || pos <= 0) {
      step = -step; // Reverse direction
    }
    myservo.write(pos);
    lastMove = millis();
  }
}

void track(int targetX) {
  int error = targetX - frame_center_x;
  int deadzone = 10; // Ignore small movements to prevent jitter

  if (abs(error) > deadzone) {
    // Map the pixel error to a small degree change
    // Using a smaller scaling factor (e.g., /40) makes tracking smoother
    int moveBuffer = error / 40; 
    pos = pos + moveBuffer;

    // Constrain servo limits
    pos = constrain(pos, 0, 170);
    myservo.write(pos);
  }
}
