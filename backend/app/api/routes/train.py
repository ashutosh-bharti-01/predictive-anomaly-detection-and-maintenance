from fastapi import APIRouter, UploadFile, File
from app.db.mongo import collection
import app.services.ml_service as ml_service
import pandas as pd
from pathlib import Path
import io
import os

router = APIRouter()

CSV_PATH = Path(__file__).resolve().parents[3] / "sensor_data.csv"


@router.post("/train-model")
async def train_model_route(file: UploadFile = File(None)):

    try:
        df = None
        source_used = None

        # =========================
        # Uploaded CSV (TOP PRIORITY)
        # =========================
        if file is not None:
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode()))
            source_used = "uploaded_csv"
            print("📤 Training from UPLOADED CSV")

        # =========================
        # Local CSV
        # =========================
        elif CSV_PATH.exists():
            df = pd.read_csv(CSV_PATH)
            source_used = "local_csv"
            print("📁 Training from LOCAL CSV")

        # =========================
        # MongoDB fallback
        # =========================
        else:
            data = list(collection.find({}, {"_id": 0}))

            if not data:
                return {"error": "No data in CSV or DB"}

            df = pd.DataFrame(data)
            source_used = "mongodb"
            print("🗄️ Training from MongoDB")

        # =========================
        # 🔹 CLEAN DATA
        # =========================
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
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce")

        df = df.dropna(subset=["temperature", "vibration", "pressure"])

        print("✅ Cleaned rows:", len(df))

        if len(df) < 30:
            return {"error": "Not enough data to train"}

        # =========================
        # CHECK NORMAL DATA AVAILABILITY
        # =========================
        if "anomaly" in df.columns:
            normal_count = len(df[df["anomaly"] == 0])
        else:
            normal_count = len(df)

        if normal_count < 20:
            return {
                "error": "Not enough NORMAL data to train",
                "normal_rows": normal_count
            }

        # =========================
        # TRAIN MODEL
        # =========================
        ml_service.train_model(df)

        model_exists = os.path.exists(ml_service.MODEL_PATH)

        return {
            "status": "success",
            "source": source_used,
            "rows_used": len(df),
            "normal_rows_used": normal_count,
            "features": FEATURE_COLUMNS,
            "model_saved": model_exists,
            "model_path": ml_service.MODEL_PATH
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }