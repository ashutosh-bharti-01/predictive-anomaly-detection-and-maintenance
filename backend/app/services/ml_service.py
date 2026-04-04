import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
scaler = StandardScaler()

FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "humidity",
    "rpm",
    "voltage",
    "current",
]

# ✅ Lower contamination (more realistic)
model = IsolationForest(contamination=0.005, random_state=42)

is_trained = False


# =========================
# 🔥 TRAIN MODEL
# =========================
def train_model(df):
    global is_trained

    print("🚀 Training model...")

    if len(df) < 50:
        print("❌ Not enough data")
        return

    if "anomaly" in df.columns:
        df = df[df["anomaly"] == 0]


    if len(df) < 30:
        print("❌ Not enough NORMAL data")
        return

    X = df[FEATURE_COLUMNS].fillna(0)
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((model, scaler), MODEL_PATH)

    is_trained = True

    print(f"✅ Model trained on {len(df)} normal rows")


# =========================
# 🔍 LOAD MODEL
# =========================
def load_model():
    global model, scaler, is_trained

    if os.path.exists(MODEL_PATH):
        loaded = joblib.load(MODEL_PATH)

        if isinstance(loaded, tuple):
            model, scaler = loaded
        else:
            model = loaded
            print("⚠️ Old model detected (no scaler). Retrain recommended.")

        is_trained = True
        print("✅ Model loaded from disk")

    else:
        print("⚠️ No saved model found")
# =========================
# 🔍 ENSURE MODEL (ONE TIME)
# =========================
def ensure_model(df):
    global is_trained

    if is_trained:
        return

    load_model()

    if is_trained:
        return

    print("⚠️ Model not found → training...")
    train_model(df)


# =========================
# ANOMALY DETECTION
# =========================
def detect_anomaly(row):
    if not is_trained:
        return 1, 0.0

    try:
        X = [[
            float(row.get(col, 0)) if row.get(col) is not None else 0.0
            for col in FEATURE_COLUMNS
        ]]
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        score = model.decision_function(X_scaled)[0]

        return int(pred), float(score)

    except Exception as e:
        print("Detect anomaly error:", e)
        return 1, 0.0