
from fastapi import APIRouter
from app.db.mongo import collection
import app.services.ml_service as ml_service
import app.services.prediction_service as prediction_service
from app.services.ai_service import generate_explanation
import pandas as pd
from app.utils.stats import update_stats, compute_z

RETRAIN_INTERVAL = 50

router = APIRouter()

index = 0

@router.get("/next")
def next_data():
    global index

    data = list(
        collection.find({}, {"_id": 0})
        .sort("timestamp", 1)
        .limit(200)
    )

    if not data:
        return {"error": "No data in DB"}

    df = pd.DataFrame(data)

    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["vibration"] = pd.to_numeric(df["vibration"], errors="coerce")
    df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")

    df = df.dropna(subset=["temperature", "vibration", "pressure"])
    print("Raw rows:", len(data))
    print("After df:", len(df))
    if index >= len(df):
        index = 0

    row = df.iloc[-1]
    ml_service.load_model()

    if not ml_service.is_trained:
        print("Before training, is_trained =", ml_service.is_trained)
        print("⚠️ Model not found, training now...")
        ml_service.train_model(df)
        print("After training, is_trained =", ml_service.is_trained)


    if index % 50 == 0:
        update_stats(df)

    z = compute_z(row["temperature"])

    pred, score = ml_service.detect_anomaly(row)

    severity = "critical" if pred == -1 else "normal"

    prediction = prediction_service.predict_future(df, ml_service.model)

    explanation = generate_explanation(row, severity, prediction)

    # collection.update_one(
    #     {"timestamp": row["timestamp"]},
    #     {"$set": {
    #         "anomaly": pred,
    #         "score": score,
    #         "severity": severity,
    #         "prediction": prediction,
    #         "explanation": explanation
    #     }}
    # )

    return {
        **row.to_dict(),
        "anomaly": pred,
        "score": score,
        "severity": severity,
        "prediction": prediction,
        "explanation": explanation
    }