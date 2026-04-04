import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

router = APIRouter()

CSV_NAME = "sensor_data.csv"


def _clamp(value, low, high):
    return max(low, min(high, value))


@router.post("/generate")
def generate_dataset(
    rows: int = Query(500, ge=10, le=20000),
    interval_seconds: int = Query(10, ge=1, le=3600),
    anomaly_rate: float = Query(0.06, ge=0.0, le=0.5),
    seed: Optional[int] = Query(None),
):
    rng = random.Random(seed) if seed is not None else random

    start_time = datetime.now(timezone.utc) - timedelta(seconds=interval_seconds * (rows - 1))

    csv_path = Path(__file__).resolve().parents[3] / CSV_NAME
    rows_out = []

    for i in range(rows):
        ts = start_time + timedelta(seconds=i * interval_seconds)

        temp = 72 + 4 * math.sin(i / 60) + rng.gauss(0, 0.3)
        pressure = 101 + 2 * math.sin(i / 80) + rng.gauss(0, 0.2)
        vibration = 0.35 + 0.05 * math.sin(i / 30) + rng.gauss(0, 0.01)
        humidity = 45 + 5 * math.sin(i / 90) + rng.gauss(0, 0.5)
        flow_rate = 22 + 1.5 * math.sin(i / 70) + rng.gauss(0, 0.2)
        voltage = 230 + 1.5 * math.sin(i / 100) + rng.gauss(0, 0.3)
        current = 7.5 + 0.6 * math.sin(i / 65) + rng.gauss(0, 0.1)
        rpm = 1500 + 60 * math.sin(i / 50) + rng.gauss(0, 5)

        anomaly = 0
        if rng.random() < anomaly_rate:
            anomaly = 1
            temp += rng.uniform(6, 15)
            pressure += rng.uniform(5, 15)
            vibration += rng.uniform(0.4, 1.2)
            flow_rate -= rng.uniform(3, 7)
            voltage -= rng.uniform(10, 25)
            current += rng.uniform(2, 5)
            rpm -= rng.uniform(200, 500)

        rows_out.append({
            "timestamp": ts.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "temperature": round(_clamp(temp, 40, 120), 2),
            "pressure": round(_clamp(pressure, 85, 130), 2),
            "vibration": round(_clamp(vibration, 0.05, 2.5), 3),
            "humidity": round(_clamp(humidity, 15, 95), 2),
            "flow_rate": round(_clamp(flow_rate, 5, 35), 2),
            "voltage": round(_clamp(voltage, 180, 260), 2),
            "current": round(_clamp(current, 2, 20), 2),
            "rpm": int(_clamp(rpm, 600, 2200)),
            "anomaly": anomaly,
        })

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    return {
        "rows": rows,
        "path": str(csv_path),
        "preview": rows_out[-10:],
    }
