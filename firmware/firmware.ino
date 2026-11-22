/*
  TIDAL — 3-category waste display  (STM32 Nucleo-F072RB, Arduino IDE + STM32duino)
  ---------------------------------------------------------------------------------
  Pairs with:  python vision/live_tags.py

  The laptop sends one word per line (resent every 0.2 s as a heartbeat):
    PLASTIC -> LCD "Plastic&Aluminum", GREEN LED on
    PAPER   -> LCD "Paper&Cardboard",  WHITE LED on
    TRASH   -> LCD "Trash",            RED LED on
    IDLE    -> LCD "Show an item",     all LEDs off
  No message for 1.5 s -> LCD "Laptop offline", all three LEDs blink.

  ---------------- WIRING (Nucleo headers, 3.3 V logic) ----------------
  LEDs (each through its own resistor to GND, long leg toward the pin):
    D4 -> 220 R -> GREEN LED -> GND        (Plastic&Aluminum)
    D5 -> 220 R -> RED LED   -> GND        (Trash)
    D6 -> 100 R -> WHITE LED -> GND        (Paper & cardboard)   * see note
    * White LEDs need ~3 V, and the pin only gives 3.3 V, so 220 R leaves almost no
      current (very dim). Use 100 R (or 68 R). Same if your green LED is dim.

  LCD1602, parallel (16 pins, set USE_I2C_LCD 0):
    1 VSS -> GND          2 VDD -> 5V           3 V0  -> middle of 10k pot (pot ends to 5V and GND)
    4 RS  -> D12          5 RW  -> GND          6 E   -> D11
    11 D4 -> D10          12 D5 -> D9           13 D6 -> D8           14 D7 -> D7
    15 A  -> 5V through 220 R (backlight)       16 K  -> GND
    (LCD pins 7-10 unused.) RW to GND = the LCD only listens, it never sends 5 V back.

  LCD1602 with I2C backpack (4 pins, set USE_I2C_LCD 1, install "LiquidCrystal I2C"):
    GND -> GND   VCC -> 5V   SDA -> D14 (SDA)   SCL -> D15 (SCL)
    Address is usually 0x27 (else 0x3F).

  Don't use D0/D1 (USB serial to the laptop) or D13 (on-board LED).
  If the LCD shows only blocks or nothing, turn the contrast pot.
*/

#define USE_I2C_LCD 0 // 0 = 16-pin parallel LCD, 1 = I2C backpack

#if USE_I2C_LCD
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
LiquidCrystal_I2C lcd(0x27, 16, 2);
#else
#include <LiquidCrystal.h>
//               RS   E    D4   D5  D6  D7
LiquidCrystal lcd(D12, D11, D10, D9, D8, D7);
#endif

const int PIN_GREEN = D4; // Plastic & Aluminum
const int PIN_RED = D5;   // Trash
const int PIN_WHITE = D6; // Paper & cardboard

const unsigned long TIMEOUT_MS = 1500;
unsigned long lastMsg = 0;
String line = "";
String shown = ""; // what's on the LCD now, so we only redraw on change

void setLeds(bool g, bool w, bool r)
{
  digitalWrite(PIN_GREEN, g);
  digitalWrite(PIN_WHITE, w);
  digitalWrite(PIN_RED, r);
}

void showLcd(const String &top, const String &bottom)
{
  String key = top + "|" + bottom;
  if (key == shown)
    return; // no flicker: only redraw when it changes
  shown = key;
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(top);
  lcd.setCursor(0, 1);
  lcd.print(bottom);
}

void handle(const String &cmd)
{
  lastMsg = millis();
  if (cmd == "PLASTIC")
  {
    setLeds(true, false, false);
    showLcd("Plastic&Aluminum", "-> GREEN bin");
  }
  else if (cmd == "PAPER")
  {
    setLeds(false, true, false);
    showLcd("Paper&Cardboard", "-> WHITE bin");
  }
  else if (cmd == "TRASH")
  {
    setLeds(false, false, true);
    showLcd("Trash", "-> RED bin");
  }
  else if (cmd == "IDLE")
  {
    setLeds(false, false, false);
    showLcd("Show an item", "to the camera");
  }
}

void setup()
{
  Serial.begin(115200);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_WHITE, OUTPUT);
  pinMode(PIN_RED, OUTPUT);

#if USE_I2C_LCD
  lcd.init();
  lcd.backlight();
#else
  lcd.begin(16, 2);
#endif

  // Self-test: each LED in turn, so you can check the wiring
  showLcd("TIDAL Recycling", "LED test...");
  setLeds(true, false, false);
  delay(300);
  setLeds(false, true, false);
  delay(300);
  setLeds(false, false, true);
  delay(300);
  setLeds(false, false, false);
  showLcd("Waiting for", "laptop...");
}

void loop()
{
  // Non-blocking line reader: collect characters until '\n'
  while (Serial.available())
  {
    char c = Serial.read();
    if (c == '\n')
    {
      line.trim();
      if (line.length())
        handle(line);
      line = "";
    }
    else if (c != '\r' && line.length() < 20)
      line += c;
  }

  // Fail-safe: laptop stopped talking -> say so and blink everything
  if (lastMsg == 0 || millis() - lastMsg > TIMEOUT_MS)
  {
    bool on = (millis() / 300) % 2;
    setLeds(on, on, on);
    showLcd("Laptop offline", "Run live_tags.py");
  }
}