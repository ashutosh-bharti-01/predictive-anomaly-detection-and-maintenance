from fastapi import APIRouter
from app.db.mongo import collection
import app.services.ml_service as ml_service
import pandas as pd

router = APIRouter()


@router.get("/train-model")
def train_model_route(source: str = "db"):
    try:
        if source == "csv":
            df = pd.read_csv("sensor_data.csv")
            print("📁 Training from CSV")
        else:
            data = list(collection.find({}, {"_id": 0}))
            print("🗄️ Training from MongoDB")

            if not data:
                return {"error": "No data in DB"}

            df = pd.DataFrame(data)

        FEATURE_COLUMNS = [
            "temperature",
            "vibration",
            "pressure",
            "humidity",
            "rpm",
            "voltage",
            "current",
        ]

        for col in FEATURE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = 0  # fallback if missing

        df = df.dropna(subset=["temperature", "vibration", "pressure"])

        print("✅ Cleaned rows:", len(df))

        if len(df) < 20:
            return {"error": "Not enough data to train"}

        ml_service.train_model(df)

        import os
        model_exists = os.path.exists(ml_service.MODEL_PATH)

        return {
            "status": "success",
            "rows_used": len(df),
            "features": FEATURE_COLUMNS,
            "model_saved": model_exists,
            "model_path": ml_service.MODEL_PATH
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }