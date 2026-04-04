import pandas as pd
import numpy as np
from datetime import datetime, timedelta

rows = []
time = datetime.now()

# 🔹 Base NORMAL values
temp = 70
vibration = 0.3
pressure = 102
humidity = 45
rpm = 1500
voltage = 230
current = 8

# 🔹 Thresholds (aligned with prediction logic)
FAILURE_TEMP = 120
FAILURE_VIB = 5
FAILURE_PRESSURE = 150

for i in range(300):

    # =========================
    # 🟢 PHASE 1: NORMAL (0–100)
    # =========================
    if i < 100:
        temp += np.random.normal(0, 0.2)
        vibration += np.random.normal(0, 0.01)
        pressure += np.random.normal(0, 0.1)

        anomaly = 0

    # =========================
    # 🟡 PHASE 2: DEGRADATION (100–200)
    # =========================
    elif i < 200:
        temp += np.random.normal(0.3, 0.4)
        vibration += np.random.normal(0.1, 0.05)
        pressure += np.random.normal(0.5, 0.3)
        rpm += np.random.normal(10, 20)

        anomaly = 0  # still not failure

    # =========================
    # 🔴 PHASE 3: FAILURE ZONE (200+)
    # =========================
    else:
        temp += np.random.uniform(2, 5)
        vibration += np.random.uniform(0.5, 2)
        pressure += np.random.uniform(5, 10)
        rpm += np.random.uniform(50, 200)

        # 🔥 Force threshold crossing
        if temp < FAILURE_TEMP:
            temp += np.random.uniform(5, 10)

        anomaly = 1

    rows.append({
        "timestamp": time + timedelta(seconds=i * 10),
        "temperature": round(temp, 2),
        "vibration": round(vibration, 3),
        "pressure": round(pressure, 2),
        "humidity": round(humidity + np.random.normal(0, 1), 2),
        "rpm": round(rpm, 0),
        "voltage": round(voltage + np.random.normal(0, 2), 2),
        "current": round(current + np.random.normal(0, 0.5), 2),
        "anomaly": anomaly
    })

df = pd.DataFrame(rows)
df.to_csv("../../anomaly_data.csv", index=False)

print("🔥 anomaly_data.csv generated with failure patterns")