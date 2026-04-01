import csv
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/history")
def get_history():
    csv_path = Path(__file__).resolve().parents[3] / "sensor_data.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="sensor_data.csv not found")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    def coerce_row(row):
        numeric_fields = {
            "temperature": float,
            "pressure": float,
            "vibration": float,
            "humidity": float,
            "flow_rate": float,
            "voltage": float,
            "current": float,
            "rpm": float,
        }
        result = {"timestamp": row.get("timestamp", "")}
        for key, caster in numeric_fields.items():
            value = row.get(key)
            result[key] = caster(value) if value else None
        result["anomaly"] = int(row["anomaly"]) if row.get("anomaly") else None
        return result

    return [coerce_row(row) for row in rows[-200:]]
