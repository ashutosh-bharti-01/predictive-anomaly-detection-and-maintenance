from fastapi import APIRouter, UploadFile, File
from app.db.mongo import data_collection as collection
import app.services.ml_service as ml_service
import app.services.prediction_service as prediction_service
from app.services.ai_service import generate_explanation
import pandas as pd
from pathlib import Path
import io

router = APIRouter()

CSV_PATH = Path(__file__).resolve().parents[3] / "sensor_data.csv"

# 🔥 Load model once
ml_service.load_model()

global_index = 0

@router.get("/next")
def next_data():
    global global_index
    try:
        if not CSV_PATH.exists():
            return {"error": "CSV file not found"}
        
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return {"error": "CSV is empty"}
            
        for col in ["temperature", "vibration", "pressure", "humidity", "rpm", "voltage", "current"]:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce")
            
        df = df.dropna(subset=["temperature", "vibration", "pressure"])
        
        if global_index >= len(df):
            global_index = 0
            
        row = df.iloc[global_index]
        global_index += 1
        
        ml_service.ensure_model(df)
        pred, score = ml_service.detect_anomaly(row)
        severity = "critical" if pred == -1 else "normal"
        prediction = prediction_service.predict_future(df, ml_service.model)
        explanation = generate_explanation(row, severity, prediction)
        
        result = row.to_dict()
        result["anomaly"] = pred
        result["score"] = score
        result["severity"] = severity
        result["prediction"] = prediction
        result["explanation"] = explanation
        return result
    except Exception as e:
        return {"error": str(e)}


@router.post("/predict")
async def predict_data(file: UploadFile = File(None)):

    df = None
    source_used = None

    # =========================
    # Uploaded CSV
    # =========================
    if file is not None and file.filename:
        try:
            contents = await file.read()
            print("📄 Uploaded file size:", len(contents))
            if not contents:
                return {"error": "Uploaded file is empty"}

            decoded = contents.decode("utf-8").strip()

            if not decoded:
                return {"error": "Uploaded file has no content"}

            df = pd.read_csv(io.StringIO(decoded))

            if df.empty or len(df.columns) == 0:
                return {"error": "CSV has no valid columns/data"}

        except Exception as e:
            return {"error": f"Invalid CSV file: {str(e)}"}
        source_used = "uploaded_csv"
        print("📤 Using UPLOADED CSV")

    # =========================
    # Local CSV
    # =========================
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        source_used = "local_csv"
        print("📁 Using LOCAL CSV")

    # =========================
    # MongoDB fallback
    # =========================
    else:
        data = list(
            collection.find({}, {"_id": 0})
            .sort("timestamp", 1)
        )

        if not data:
            return {"error": "No data in CSV or DB"}

        df = pd.DataFrame(data)
        source_used = "mongodb"
        print("🗄 Using DB")

    # =========================
    # 🔹 CLEAN DATA
    # =========================
    for col in [
        "temperature", "vibration", "pressure",
        "humidity", "rpm", "voltage", "current"
    ]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce")

    df = df.dropna(subset=["temperature", "vibration", "pressure"])

    print("Rows used:", len(df))

    if len(df) == 0:
        return {"error": "No valid data after cleaning"}

    # =========================
    # USE LATEST ROW
    # =========================
    row = df.iloc[-1]

    # =========================
    # MODEL (NO RETRAIN LOOP)
    # =========================
    ml_service.ensure_model(df)

    # =========================
    # ANOMALY DETECTION
    # =========================
    pred, score = ml_service.detect_anomaly(row)

    severity = "critical" if pred == -1 else "normal"

    # =========================
    # PREDICTION (ARIMA + ML)
    # =========================
    prediction = prediction_service.predict_future(
        df,
        ml_service.model
    )

    # =========================
    # EXPLANATION
    # =========================
    explanation = generate_explanation(row, severity, prediction)

    return {
        "source": source_used,
        **row.to_dict(),
        "anomaly": pred,
        "score": score,
        "severity": severity,
        "prediction": prediction,
        "explanation": explanation
    }
