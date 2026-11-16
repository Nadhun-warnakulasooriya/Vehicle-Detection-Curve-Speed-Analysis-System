#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h> // අලුත් 1.3" OLED Library එක

// ═══════════════════════════════════════════════════════════
//   WIFI SETTINGS (නිවැරදි කරන ලදී)
// ═══════════════════════════════════════════════════════════
const char* ssid = "Xperia_5612";         
const char* password = "12345678";  

// ═══════════════════════════════════════════════════════════
//   NETWORK SETTINGS
// ═══════════════════════════════════════════════════════════
unsigned int localPort = 4210;
WiFiUDP udp;

// ═══════════════════════════════════════════════════════════
//   PIN CONFIGURATION (LEDs පමණි)
// ═══════════════════════════════════════════════════════════
const int S1_GREEN_LED = 4;
const int S1_YELLOW_LED = 7;
const int S1_RED_LED = 5;

const int S2_GREEN_LED = 15;
const int S2_YELLOW_LED = 18;
const int S2_RED_LED = 16;

// දුර සීමාවන් (මීටර් වලින්)
const float DANGER_DISTANCE = 25.0;  // මීටර් 25ට අඩු නම් රතු
const float CAUTION_DISTANCE = 50.0; // මීටර් 25 ත් 50 ත් අතර නම් කහ

// ═══════════════════════════════════════════════════════════
//   OLED DISPLAY CONFIGURATION (1.3" SH1106)
// ═══════════════════════════════════════════════════════════
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

#define I2C_1_SDA 8
#define I2C_1_SCL 9
#define I2C_2_SDA 10
#define I2C_2_SCL 11

TwoWire I2C_1 = TwoWire(0);
TwoWire I2C_2 = TwoWire(1);

// 1.3" SH1106 Display Objects ලෙස වෙනස් කර ඇත
Adafruit_SH1106G display1(SCREEN_WIDTH, SCREEN_HEIGHT, &I2C_1, -1);
Adafruit_SH1106G display2(SCREEN_WIDTH, SCREEN_HEIGHT, &I2C_2, -1);

void setup() {
  Serial.begin(115200);

  // 1. LED Pins Setup
  pinMode(S1_GREEN_LED, OUTPUT);
  pinMode(S1_YELLOW_LED, OUTPUT);
  pinMode(S1_RED_LED, OUTPUT);
  
  pinMode(S2_GREEN_LED, OUTPUT);
  pinMode(S2_YELLOW_LED, OUTPUT);
  pinMode(S2_RED_LED, OUTPUT);
  
  turnOffAll();

  // 2. I2C Buses ආරම්භ කිරීම
  I2C_1.begin(I2C_1_SDA, I2C_1_SCL, 100000);
  I2C_2.begin(I2C_2_SDA, I2C_2_SCL, 100000);

  // 3. Displays ආරම්භ කිරීම (SH110X සඳහා අදාල වන පරිදි)
  if(!display1.begin(0x3C, true)) {
    Serial.println("Display 1 allocation failed");
  }
  if(!display2.begin(0x3C, true)) {
    Serial.println("Display 2 allocation failed");
  }
 
  showLoadingScreen(display1, "SIDE 1");
  showLoadingScreen(display2, "SIDE 2");

  // 4. Wi-Fi සම්බන්ධ වීම
  Serial.println("\nConnecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.println(WiFi.localIP());

  // 5. UDP ආරම්භ කිරීම
  udp.begin(localPort);
 
  updateDisplay(display1, "SIDE 1", 999.0, 0, "WAITING...");
  updateDisplay(display2, "SIDE 2", 999.0, 0, "WAITING...");
}

void loop() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char incomingPacket[255];
    int len = udp.read(incomingPacket, 255);
    if (len > 0) {
      incomingPacket[len] = '\0';
    }

    StaticJsonDocument<256> doc;
    DeserializationError error = deserializeJson(doc, incomingPacket);

    if (error) {
      Serial.println("JSON parse failed");
      return;
    }

    // දත්ත ලබා ගැනීම
    float s1_dist = doc["side1"]["dist"];
    bool s1_overspeed = doc["side1"]["overspeed"];
    String s1_type = doc["side1"]["type"].as<String>();
   
    float s2_dist = doc["side2"]["dist"];
    bool s2_overspeed = doc["side2"]["overspeed"];
    String s2_type = doc["side2"]["type"].as<String>();

    // ═══════════════════════════════════════════════════════════
    //   SIDE 1 LOGIC (RED, YELLOW, GREEN)
    // ═══════════════════════════════════════════════════════════
    int s1_status = 0; // 0 = Safe, 1 = Caution, 2 = Danger
   
    if (s1_overspeed || (s1_dist > 0 && s1_dist < DANGER_DISTANCE)) {
      s1_status = 2; // Danger (Red)
    } else if (s1_dist >= DANGER_DISTANCE && s1_dist <= CAUTION_DISTANCE) {
      s1_status = 1; // Caution (Yellow)
    }

    setLights(1, s1_status);
    updateDisplay(display1, "SIDE 1", s1_dist, s1_status, s1_type);

    // ═══════════════════════════════════════════════════════════
    //   SIDE 2 LOGIC (RED, YELLOW, GREEN)
    // ═══════════════════════════════════════════════════════════
    int s2_status = 0;
   
    if (s2_overspeed || (s2_dist > 0 && s2_dist < DANGER_DISTANCE)) {
      s2_status = 2; // Danger (Red)
    } else if (s2_dist >= DANGER_DISTANCE && s2_dist <= CAUTION_DISTANCE) {
      s2_status = 1; // Caution (Yellow)
    }

    setLights(2, s2_status);
    updateDisplay(display2, "SIDE 2", s2_dist, s2_status, s2_type);
  }
}

// ═══════════════════════════════════════════════════════════
//   HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════

void setLights(int side, int status) {
  if (side == 1) {
    digitalWrite(S1_GREEN_LED, (status == 0) ? HIGH : LOW);
    digitalWrite(S1_YELLOW_LED, (status == 1) ? HIGH : LOW);
    digitalWrite(S1_RED_LED, (status == 2) ? HIGH : LOW);
  } else {
    digitalWrite(S2_GREEN_LED, (status == 0) ? HIGH : LOW);
    digitalWrite(S2_YELLOW_LED, (status == 1) ? HIGH : LOW);
    digitalWrite(S2_RED_LED, (status == 2) ? HIGH : LOW);
  }
}

void turnOffAll() {
  digitalWrite(S1_GREEN_LED, LOW); digitalWrite(S1_YELLOW_LED, LOW); digitalWrite(S1_RED_LED, LOW);
  digitalWrite(S2_GREEN_LED, LOW); digitalWrite(S2_YELLOW_LED, LOW); digitalWrite(S2_RED_LED, LOW);
}

void showLoadingScreen(Adafruit_SH1106G &display, String title) {
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);
  display.setTextSize(2);
  display.setCursor(20, 15);
  display.print(title);
  display.setTextSize(1);
  display.setCursor(20, 40);
  display.print("Connecting...");
  display.display();
}

void updateDisplay(Adafruit_SH1106G &display, String title, float dist, int status, String vehType) {
  display.clearDisplay();
 
  display.setTextSize(1);
  display.setTextColor(SH110X_BLACK, SH110X_WHITE);
  display.setCursor(0, 0);
  display.print(" " + title + " ");
 
  display.setTextColor(SH110X_WHITE);
  display.setCursor(60, 0);
  if(vehType.length() > 0) display.print(vehType);

  display.setTextSize(2);
  display.setCursor(0, 20);
  if (dist < 999.0) {
    display.print(dist, 1);
    display.print(" m");
  } else {
    display.print("CLEAR");
  }

  display.setCursor(0, 45);
  if (status == 2) {
    display.print("DANGER!");
  } else if (status == 1) {
    display.print("CAUTION");
  } else {
    display.print("SAFE");
  }

  display.display();
}