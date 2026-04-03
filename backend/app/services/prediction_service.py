import numpy as np
from sklearn.exceptions import NotFittedError

WINDOW_SIZE = 10
TIME_INTERVAL_MIN = 5
FAILURE_TEMP = 120

FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "humidity",
    "rpm",
    "voltage",
    "current",
]


def compute_slope(series):
    try:
        series = series.astype(float).dropna()

        if len(series) < 3:
            return 0.1

        if series.nunique() <= 2:
            return 0.1

        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]

        return float(slope if slope != 0 else 0.3)

    except Exception as e:
        print("Slope error:", e)
        return 0.1


def compute_feature_slopes(df):
    slopes = {}

    for col in FEATURE_COLUMNS:
        try:
            series = df[col].astype(float).dropna()

            if len(series) < 3:
                slopes[col] = 0.0
            else:
                x = np.arange(len(series))
                slopes[col] = np.polyfit(x, series, 1)[0]
        except:
            slopes[col] = 0.0

    return slopes


def predict_future(df, model, steps=30):
    if len(df) < WINDOW_SIZE:
        last_temp = df["temperature"].iloc[-1] if len(df) > 0 else 0

        return {
            "risk": "low",
            "slope": 0.0,
            "next_temp": round(float(last_temp), 2),
            "forecast": [],
            "failure_in_minutes": None,
            "failure_in_hours": None
        }

    recent = df.tail(WINDOW_SIZE)

    # 🔹 Base state
    base_row = recent.iloc[-1].to_dict()

    # 🔹 Compute slopes for ALL features
    feature_slopes = compute_feature_slopes(recent)

    forecast = []
    time_axis = []

    failure_time = None
    failure_reason = None

    try:
        for i in range(steps):
            future_row = {}

            # 🔥 Generate full future state
            for col in FEATURE_COLUMNS:
                base_val = float(base_row.get(col, 0))
                slope_val = feature_slopes.get(col, 0)

                future_val = base_val + slope_val * (i + 1)
                future_row[col] = future_val

            forecast.append({
                col: round(float(future_row[col]), 2)
                for col in FEATURE_COLUMNS
            })

            minutes = (i + 1) * TIME_INTERVAL_MIN
            time_axis.append(minutes)

            # 🔥 Full feature vector for ML
            future_vector = [
                float(future_row[col]) for col in FEATURE_COLUMNS
            ]

            pred = model.predict([future_vector])[0]
            score = model.decision_function([future_vector])[0]

            # 🔥 HYBRID FAILURE LOGIC
            if failure_time is None:

                if pred == -1:
                    failure_time = minutes
                    failure_reason = "ml_anomaly"

                elif future_row["temperature"] >= FAILURE_TEMP:
                    failure_time = minutes
                    failure_reason = "threshold_breach"

        failure_hours = round(failure_time / 60, 2) if failure_time else None

        # 🔥 Risk classification
        if failure_time is None:
            risk = "low"
        elif failure_hours < 2:
            risk = "critical"
        elif failure_hours < 6:
            risk = "high"
        else:
            risk = "medium"

        return {
            "forecast": forecast,
            "failure_in_minutes": failure_time,
            "failure_in_hours": failure_hours,
            "failure_reason": failure_reason,
            "risk": risk,
            "slope": float(feature_slopes.get("temperature", 0))
        }

    except NotFittedError:
        return {
            "risk": "low",
            "error": "model_not_trained",
            "forecast": [],
            "failure_in_minutes": None,
            "failure_in_hours": None
        }