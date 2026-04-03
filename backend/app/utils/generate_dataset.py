import pandas as pd
import numpy as np
from datetime import datetime, timedelta

rows = []
time = datetime.now()

temp = 60
vibration = 0.2
pressure = 101
rpm = 1500
voltage = 230
current = 8

for i in range(300):

    # 🟢 Phase 1: Normal (0–150)
    if i < 150:
        temp += np.random.normal(0, 0.2)
        vibration += np.random.normal(0, 0.01)
        pressure += np.random.normal(0, 0.1)
        rpm += np.random.normal(0, 5)

    # 🟡 Phase 2: Degradation (150–230)
    elif i < 230:
        temp += np.random.normal(0.4, 0.2)
        vibration += np.random.normal(0.05, 0.02)
        pressure += np.random.normal(0.5, 0.2)
        rpm += np.random.normal(10, 5)

    # 🔴 Phase 3: Failure zone (230–300)
    else:
        temp += np.random.uniform(1.5, 3)
        vibration += np.random.uniform(0.2, 0.5)
        pressure += np.random.uniform(2, 5)
        rpm += np.random.uniform(20, 50)

    # Derived signals
    voltage += np.random.normal(0.3, 0.5)
    current -= np.random.normal(0.05, 0.1)

    # Clamp values (avoid unrealistic)
    vibration = max(0, vibration)

    # Label anomaly (for validation only)
    anomaly = 1 if i > 230 else 0

    rows.append({
        "timestamp": time + timedelta(seconds=i * 10),
        "temperature": round(temp, 2),
        "vibration": round(vibration, 3),
        "pressure": round(pressure, 2),
        "rpm": round(rpm, 2),
        "voltage": round(voltage, 2),
        "current": round(current, 2),
        "anomaly": anomaly
    })

df = pd.DataFrame(rows)
df.to_csv("sensor_data.csv", index=False)

print("🔥 Realistic failure dataset generated!")