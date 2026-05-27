#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ═══════════════════════════════════════════════════════════
//  NETWORK & PIN CONFIGURATION (Dakku Hillside Node)
// ═══════════════════════════════════════════════════════════
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const int udpPort = 4210;

const int LED_GREEN  = 12;
const int LED_YELLOW = 13;
const int LED_RED    = 14;

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

WiFiUDP udp;
unsigned long lastPacketTime = 0;
const unsigned long TIMEOUT_MS = 2000; // Fail-safe timeout if PC disconnects
bool displayActive = false;

void setup() {
  Serial.begin(115200);

  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;); // Don't proceed, loop forever
  }
  
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Dakku Node Init...");
  display.display();

  connectWiFi();
  udp.begin(udpPort);
}

void loop() {
  // 1. Handle Wi-Fi Reconnection Gracefully
  if (WiFi.status() != WL_CONNECTED) {
    display.clearDisplay();
    display.setCursor(0,0);
    display.println("WiFi Lost! Reconnecting...");
    display.display();
    connectWiFi();
  }

  // 2. Parse Incoming UDP Packets
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char packetBuffer[255];
    int len = udp.read(packetBuffer, 255);
    if (len > 0) packetBuffer[len] = '\0';

    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, packetBuffer);

    if (!error) {
      lastPacketTime = millis();
      float dist = doc["dist"];
      bool overspeed = doc["overspeed"];
      int limit = doc["limit"];
      const char* vType = doc["type"];

      updateLEDs(dist);
      updateOLED(overspeed, limit, vType);
    }
  }

  // 3. Fail-safe: Clear outputs if PC stops sending data
  if (millis() - lastPacketTime > TIMEOUT_MS) {
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_RED, LOW);
    if (displayActive) {
      display.clearDisplay();
      display.display();
      displayActive = false;
    }
  }
}

// ═══════════════════════════════════════════════════════════
//  HARDWARE CONTROL FUNCTIONS
// ═══════════════════════════════════════════════════════════

void connectWiFi() {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
  display.clearDisplay();
  display.display();
}

void updateLEDs(float dist) {
  // Distance thresholds logic
  if (dist == 999.0) { // No vehicles detected
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_RED, LOW);
  } else if (dist < 10.0) {
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_RED, HIGH);
  } else if (dist < 25.0) {
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_YELLOW, HIGH);
    digitalWrite(LED_RED, LOW);
  } else {
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_YELLOW, LOW);
    digitalWrite(LED_RED, LOW);
  }
}

void updateOLED(bool overspeed, int limit, const char* vType) {
  if (overspeed) {
    display.clearDisplay();
    
    // 1. Draw "Warning Sign" (Triangle with Exclamation)
    display.drawTriangle(10, 50, 30, 10, 50, 50, SSD1306_WHITE);
    display.setCursor(28, 26);
    display.setTextSize(2);
    display.print("!");

    // 2. "SLOW DOWN" Text
    display.setCursor(60, 16);
    display.setTextSize(1);
    display.println("SLOW");
    display.setCursor(60, 28);
    display.println("DOWN!");

    // 3. Max Speed for specific vehicle type
    display.setCursor(0, 56);
    display.setTextSize(1);
    display.print("Max ");
    display.print(vType);
    display.print(": ");
    display.print(limit);
    display.print("km/h");

    display.display();
    displayActive = true;
  } else if (displayActive) {
    // Clear OLED if speed is safe
    display.clearDisplay();
    display.display();
    displayActive = false;
  }
}