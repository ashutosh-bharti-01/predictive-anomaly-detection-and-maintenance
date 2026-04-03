import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
last_trained_timestamp = None
RETRAIN_THRESHOLD = 50
FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "humidity",
    "rpm",
    "voltage",
    "current",
]
model = IsolationForest(contamination=0.08, random_state=42)

is_trained = False


def train_model(df):
    global is_trained, last_trained_timestamp

    print("🚀 training model with rows:", len(df))

    if len(df) < 20:
        print("not enough data")
        return

    df = df.tail(500)

    X = df[FEATURE_COLUMNS].fillna(0)

    model.fit(X)

    print("💾 saving model...")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    is_trained = True
    last_trained_timestamp = df["timestamp"].max()
    print("✅ model trained successfully. is_trained =", is_trained, " last_trained_timestamp =", last_trained_timestamp)


def detect_anomaly(row):
    if not is_trained:
        return 1, 0.0

    try:
        X = [[
            float(row.get(col, 0)) if row.get(col) is not None else 0.0
            for col in FEATURE_COLUMNS
        ]]

        pred = model.predict(X)[0]
        score = model.decision_function(X)[0]

        return int(pred), float(score)

    except Exception as e:
        print("Detect anomaly error:", e)
        return 1, 0.0


def load_model():
    global model, is_trained

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        is_trained = True
        is_trained = True
        print("✅ model loaded from disk")
    else:
        print("⚠️ no saved model found")

def ensure_model(df):
    global is_trained

    if is_trained:
        return

    load_model()

    if is_trained:
        return

    print("⚠️ model not found. Training now...")
    train_model(df)


def should_retrain(df):
    global last_trained_timestamp

    if df.empty:
        return False

    latest_ts = df["timestamp"].max()

    # First time → train
    if last_trained_timestamp is None:
        return True

    # Count new rows since last training
    new_rows = df[df["timestamp"] > last_trained_timestamp]

    if len(new_rows) >= RETRAIN_THRESHOLD:
        print(f"🔥 New data detected: {len(new_rows)} rows → retrain")
        return True

    print(f"⏭️ Skipping retrain. New rows: {len(new_rows)}")
    return False