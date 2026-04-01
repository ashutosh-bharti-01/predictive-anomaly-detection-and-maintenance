import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")

model = IsolationForest(contamination=0.08, random_state=42)

is_trained = False


def train_model(df):
    global is_trained

    print("🚀 training model with rows:", len(df))

    if len(df) < 20:
        print("not enough data")
        return

    X = df[["temperature", "vibration", "pressure"]]

    model.fit(X)

    print("💾 saving model...")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    is_trained = True
    print("✅ model trained successfully. is_trained =", is_trained)


def detect_anomaly(row):
    if not is_trained:
        return 1, 0.0

    X = [[row["temperature"], row["vibration"], row["pressure"]]]

    pred = model.predict(X)[0]
    score = model.decision_function(X)[0]

    return int(pred), float(score)


def load_model():
    global model, is_trained

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
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

    # Otherwise train
    print("⚠️ model not found. Training now...")
    train_model(df)