
import numpy as np
from sklearn.exceptions import NotFittedError


WINDOW_SIZE = 10

def get_recent_window(df, index):
    start = max(0, index - WINDOW_SIZE)
    return df.iloc[start:index]

def compute_slope(series):
    try:
        # Convert to numeric
        series = series.astype(float)

        # Remove NaNs
        series = series.dropna()


        # Not enough data → no slope (making non flat slope)


        if len(series) < 3:
            return 0.1

        # If all values same → no trend
        if series.nunique() <= 2:
            return 0.1

        x = np.arange(len(series))

        slope = np.polyfit(x, series, 1)[0]

        if slope == 0:
            slope = 0.5

        return float(slope)

    except Exception as e:
        print("Slope error:", e)
        return 0.0



def predict_future(df, model, steps=10):
    if len(df) < WINDOW_SIZE:
        last_temp = df["temperature"].iloc[-1] if len(df) > 0 else 0

        return {
            "risk": "low",
            "slope": 0.0,
            "next_temp": round(float(last_temp), 2),
            "forecast": []
        }

    print("Recent temps:", df.tail(10)["temperature"].tolist())
    recent = df.tail(WINDOW_SIZE)

    slope = compute_slope(recent["temperature"])
    if slope == 0:
        slope = 0.3

    last_temp = recent["temperature"].iloc[-1]

    # 🔮 Generate future points
    forecast = []
    current = last_temp

    for i in range(steps):
        current += slope
        forecast.append(round(float(current), 2))

    # Use last forecast point for anomaly check
    future_point = [[
        forecast[-1],
        recent["vibration"].iloc[-1],
        recent["pressure"].iloc[-1]
    ]]

    try:
        pred = model.predict(future_point)[0]
        score = model.decision_function(future_point)[0]
    except NotFittedError:
        return {
            "risk": "low",
            "slope": float(slope),
            "next_temp": round(float(last_temp), 2),
            "forecast": [],
            "error": "model_not_trained"
        }

    risk = "low"
    if pred == -1:
        risk = "high"
    elif abs(score) > 0.2:
        risk = "medium"

    return {
        "next_temp": forecast[0],
        "forecast": forecast,
        "risk": risk,
        "slope": float(slope),
        "anomaly_pred": int(pred)
    }