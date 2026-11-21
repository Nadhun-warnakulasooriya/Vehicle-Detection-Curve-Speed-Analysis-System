#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>

// ═══════════════════════════════════════════════════════════
//   WIFI & NETWORK SETTINGS
// ═══════════════════════════════════════════════════════════
const char* ssid     = "Xperia_5612";
const char* password = "12345678";

unsigned int localPort = 4210;
WiFiUDP udp;

// ═══════════════════════════════════════════════════════════
//   BEHAVIOR SETTINGS
// ═══════════════════════════════════════════════════════════
// Blind bend එකකදී Side 1 වාහනයේ දුර Side 2 Traffic Light එකටත්,
// Side 2 වාහනයේ දුර Side 1 Traffic Light එකටත් දෙන්න ඕනෙ නම් 'true' කරන්න.
#define CROSS_SIDE_LIGHTS false

// දුර සීමාවන් (මීටර් වලින් - Traffic Lights සඳහා පමණි)
const float DANGER_DISTANCE  = 25.0;  // 25m ට අඩු නම් රතු
const float CAUTION_DISTANCE = 50.0;  // 25m - 50m අතර නම් කහ

// PC එකෙන් දත්ත එන එක නැවතුනොත් (තත්පර 3කට වඩා) ආරක්ෂිත තත්ත්වයට reset වීමට
const unsigned long UDP_TIMEOUT_MS = 3000;

// ═══════════════════════════════════════════════════════════
//   PIN CONFIGURATION (TRAFFIC LIGHT LEDs)
// ═══════════════════════════════════════════════════════════
const int S1_GREEN_LED  = 4;
const int S1_YELLOW_LED = 7;
const int S1_RED_LED    = 5;

const int S2_GREEN_LED  = 15;
const int S2_YELLOW_LED = 18;
const int S2_RED_LED    = 16;

// ═══════════════════════════════════════════════════════════
//   OLED DISPLAY CONFIGURATION (1.3" SH1106)
// ═══════════════════════════════════════════════════════════
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64

#define I2C_1_SDA 8
#define I2C_1_SCL 9
#define I2C_2_SDA 10
#define I2C_2_SCL 11

TwoWire I2C_1 = TwoWire(0);
TwoWire I2C_2 = TwoWire(1);

Adafruit_SH1106G display1(SCREEN_WIDTH, SCREEN_HEIGHT, &I2C_1, -1);
Adafruit_SH1106G display2(SCREEN_WIDTH, SCREEN_HEIGHT, &I2C_2, -1);

// ═══════════════════════════════════════════════════════════
//   STATE TRACKING (ANTI-FLICKER & TIMEOUT)
// ═══════════════════════════════════════════════════════════
// Display modes: 0 = CLEAR, 1 = SAFE, 2 = OVERSPEED, 3 = WAITING
struct DisplayState {
  int mode;
  String vehType;
  int speedLimit;
};

DisplayState lastDisp1 = {-1, "", -1};
DisplayState lastDisp2 = {-1, "", -1};

int lastLight1 = -1;
int lastLight2 = -1;

unsigned long lastPacketTime = 0;
bool isTimedOut = true;

// Function Prototypes
void updateDisplayIfChanged(Adafruit_SH1106G &display, const char* title,
                            int mode, String vehType, int speedLimit,
                            DisplayState &prevState);
int getDistanceStatus(float dist);
int getDisplayMode(float dist, bool overspeed, int limit);
void setLights(int side, int status);
void turnOffAll();
void showLoadingScreen(Adafruit_SH1106G &display, const char* title);

// ═══════════════════════════════════════════════════════════
//   SETUP
// ═══════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(1000);

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

  // 3. OLED Displays ආරම්භ කිරීම
  if (!display1.begin(0x3C, true)) {
    Serial.println("Display 1 allocation failed");
  }
  if (!display2.begin(0x3C, true)) {
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
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  // 5. UDP ආරම්භ කිරීම
  udp.begin(localPort);

  // ආරම්භක තිරය පෙන්වීම
  setLights(1, 0);
  setLights(2, 0);
  updateDisplayIfChanged(display1, "SIDE 1", 3, "", 0, lastDisp1);
  updateDisplayIfChanged(display2, "SIDE 2", 3, "", 0, lastDisp2);
}

// ═══════════════════════════════════════════════════════════
//   MAIN LOOP (NON-BLOCKING)
// ═══════════════════════════════════════════════════════════
void loop() {
  int packetSize = udp.parsePacket();

  if (packetSize > 0) {
    char incomingPacket[512];
    int len = udp.read(incomingPacket, sizeof(incomingPacket) - 1);
    if (len > 0) {
      incomingPacket[len] = '\0';
    }

    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, incomingPacket);

    if (error) {
      Serial.print("JSON parse failed: ");
      Serial.println(error.c_str());
      return;
    }

    lastPacketTime = millis();
    isTimedOut = false;

    // 1. JSON දත්ත කියවා ගැනීම
    float  s1_dist      = doc["side1"]["dist"]      | 999.0f;
    bool   s1_overspeed = doc["side1"]["overspeed"] | false;
    int    s1_limit     = doc["side1"]["limit"]     | 0;
    String s1_type      = String(doc["side1"]["type"] | "");

    float  s2_dist      = doc["side2"]["dist"]      | 999.0f;
    bool   s2_overspeed = doc["side2"]["overspeed"] | false;
    int    s2_limit     = doc["side2"]["limit"]     | 0;
    String s2_type      = String(doc["side2"]["type"] | "");

    // 2. TRAFFIC LIGHTS -> DISTANCE WARNING පමණයි
    int s1_light = getDistanceStatus(s1_dist);
    int s2_light = getDistanceStatus(s2_dist);

    #if CROSS_SIDE_LIGHTS
      setLights(2, s1_light); // Side 1 දුර -> Side 2 Light එකට
      setLights(1, s2_light); // Side 2 දුර -> Side 1 Light එකට
    #else
      setLights(1, s1_light);
      setLights(2, s2_light);
    #endif

    // 3. OLED DISPLAYS -> SPEED WARNING පමණයි
    int s1_mode = getDisplayMode(s1_dist, s1_overspeed, s1_limit);
    int s2_mode = getDisplayMode(s2_dist, s2_overspeed, s2_limit);

    updateDisplayIfChanged(display1, "SIDE 1", s1_mode, s1_type, s1_limit, lastDisp1);
    updateDisplayIfChanged(display2, "SIDE 2", s2_mode, s2_type, s2_limit, lastDisp2);
  }

  // PC එකෙන් දත්ත එන එක නැවතුනොත් (Timeout) ආරක්ෂිත තත්ත්වයට යාම
  if (!isTimedOut && (millis() - lastPacketTime > UDP_TIMEOUT_MS)) {
    isTimedOut = true;
    setLights(1, 0);
    setLights(2, 0);
    updateDisplayIfChanged(display1, "SIDE 1", 3, "", 0, lastDisp1);
    updateDisplayIfChanged(display2, "SIDE 2", 3, "", 0, lastDisp2);
  }
}

// ═══════════════════════════════════════════════════════════
//   HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════

// දුර අනුව Traffic Light තත්ත්වය (0 = Green, 1 = Yellow, 2 = Red)
int getDistanceStatus(float dist) {
  if (dist > 0 && dist < DANGER_DISTANCE) {
    return 2; // Red (Danger)
  } else if (dist >= DANGER_DISTANCE && dist <= CAUTION_DISTANCE) {
    return 1; // Yellow (Caution)
  }
  return 0;   // Green (Safe / Clear)
}

// Display එකේ පෙන්විය යුතු තත්ත්වය (0 = CLEAR, 1 = SAFE, 2 = OVERSPEED)
int getDisplayMode(float dist, bool overspeed, int limit) {
  if (overspeed && limit > 0) {
    return 2; // Overspeed warning
  } else if (dist < 999.0) {
    return 1; // Vehicle detected, safe speed
  }
  return 0;   // No vehicles
}

void setLights(int side, int status) {
  if (side == 1) {
    if (lastLight1 == status) return;
    lastLight1 = status;
    digitalWrite(S1_GREEN_LED,  (status == 0) ? HIGH : LOW);
    digitalWrite(S1_YELLOW_LED, (status == 1) ? HIGH : LOW);
    digitalWrite(S1_RED_LED,    (status == 2) ? HIGH : LOW);
  } else {
    if (lastLight2 == status) return;
    lastLight2 = status;
    digitalWrite(S2_GREEN_LED,  (status == 0) ? HIGH : LOW);
    digitalWrite(S2_YELLOW_LED, (status == 1) ? HIGH : LOW);
    digitalWrite(S2_RED_LED,    (status == 2) ? HIGH : LOW);
  }
}

void turnOffAll() {
  digitalWrite(S1_GREEN_LED, LOW);  digitalWrite(S1_YELLOW_LED, LOW);  digitalWrite(S1_RED_LED, LOW);
  digitalWrite(S2_GREEN_LED, LOW);  digitalWrite(S2_YELLOW_LED, LOW);  digitalWrite(S2_RED_LED, LOW);
  lastLight1 = -1;
  lastLight2 = -1;
}

void showLoadingScreen(Adafruit_SH1106G &display, const char* title) {
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);
  display.setTextSize(2);
  display.setCursor(28, 15);
  display.print(title);
  display.setTextSize(1);
  display.setCursor(22, 42);
  display.print("Connecting...");
  display.display();
}

// අගයන් වෙනස් වුණොත් පමණක් OLED එක නැවත අඳියි (Flicker වළක්වයි)
void updateDisplayIfChanged(Adafruit_SH1106G &display, const char* title,
                            int mode, String vehType, int speedLimit,
                            DisplayState &prevState) {
  // 'Tree weeler' නම නිවැරදි කර Capital කිරීම
  if (vehType == "Tree weeler") {
    vehType = "THREE WHEELER";
  }
  vehType.toUpperCase();

  // Overspeed නැති අවස්ථාවල vehicle type සහ limit නොසලකා සංසන්දනය කිරීම
  if (mode != 2) {
    vehType = "";
    speedLimit = 0;
  }

  // පෙර පෙන්වූ දත්තම නම් නැවත draw නොකර පිටවීම
  if (prevState.mode == mode &&
      prevState.vehType == vehType &&
      prevState.speedLimit == speedLimit) {
    return;
  }

  prevState.mode       = mode;
  prevState.vehType    = vehType;
  prevState.speedLimit = speedLimit;

  display.clearDisplay();

  // ඉහළ Title Bar එක
  display.setTextSize(1);
  display.setTextColor(SH110X_BLACK, SH110X_WHITE);
  display.setCursor(0, 0);
  display.print(" ");
  display.print(title);
  display.print(" ");

  display.setTextColor(SH110X_WHITE);

  if (mode == 2) {
    // ── OVERSPEED WARNING ──
    // පේළිය 1: Vehicle Type
    display.setTextSize(1);
    display.setCursor(0, 13);
    display.print(vehType);

    // පේළිය 2: SLOW DOWN
    display.setTextSize(2);
    display.setCursor(0, 25);
    display.print("SLOW DOWN");

    // පේළිය 3: TO X km/h
    display.setTextSize(2);
    display.setCursor(0, 46);
    display.print("TO ");
    display.print(speedLimit);
    display.print("km/h");
  }
  else if (mode == 1) {
    // ── වාහනයක් ඇත, වේගය ආරක්ෂිතයි ──
    display.setTextSize(2);
    display.setCursor(40, 28);
    display.print("SAFE");
  }
  else if (mode == 0) {
    // ── වාහන කිසිවක් නැත ──
    display.setTextSize(2);
    display.setCursor(34, 28);
    display.print("CLEAR");
  }
  else {
    // ── PC එකෙන් දත්ත ලැබෙන තුරු ──
    display.setTextSize(2);
    display.setCursor(4, 28);
    display.print("WAITING...");
  }

  display.display();
}
