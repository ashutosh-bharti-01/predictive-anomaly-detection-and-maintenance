import pandas as pd
import numpy as np
from datetime import datetime, timedelta

rows = []
time = datetime.now()

# Stable baseline
temp = 65
vibration = 0.25
pressure = 101
humidity = 45
rpm = 1500
voltage = 230
current = 8

for i in range(2000):
    temp += np.random.normal(0, 0.2)
    vibration += np.random.normal(0, 0.01)
    pressure += np.random.normal(0, 0.1)
    humidity += np.random.normal(0, 0.2)
    rpm += np.random.normal(0, 2)
    voltage += np.random.normal(0, 0.5)
    current += np.random.normal(0, 0.05)

    rows.append({
        "timestamp": time + timedelta(seconds=i * 10),
        "temperature": round(temp, 2),
        "vibration": round(max(0, vibration), 3),
        "pressure": round(pressure, 2),
        "humidity": round(humidity, 2),
        "rpm": round(rpm, 0),
        "voltage": round(voltage, 2),
        "current": round(current, 2),
        "anomaly": 0
    })

df = pd.DataFrame(rows)
df.to_csv("../../sensor_data.csv", index=False)

print("✅ sensor_data.csv generated")