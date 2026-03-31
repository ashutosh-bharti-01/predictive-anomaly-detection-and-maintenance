#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT11

const char* ssid = "wifi_name";
const char* password = "wifi_pass";

//  PC IP 
const char* serverName = "http://192.168.1.4:8000/sensor";

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n Wifi Connected!");
  dht.begin();
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    float temp = dht.readTemperature();
    float hum = dht.readHumidity();

    if (isnan(temp) || isnan(hum)) {
      Serial.println("Sensor error");
      return;
    }
 
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    String json = "{";
    json += "\"temperature\":" + String(temp) + ",";
    json += "\"humidity\":" + String(hum);
    json += "}";

    int httpResponseCode = http.POST(json);
    int rnd = random(0, 100);
    Serial.print("random==>");
    Serial.println(rnd);

    if (rnd > 85) {
    temp += random(5, 10);  
    }

    Serial.println("sending data...");
    String strTemp = String(temp)+"C";
    String strHum = String(hum)+"%";
    Serial.println(strTemp);
    Serial.println(strHum);


    Serial.print("Response code: ");
    Serial.println(httpResponseCode);

    http.end();
  } else {
    Serial.print('not connected...retring connection...');
  }

  delay(3000);
}
