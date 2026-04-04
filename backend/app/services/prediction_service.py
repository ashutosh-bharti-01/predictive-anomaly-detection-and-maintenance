import numpy as np
from sklearn.exceptions import NotFittedError
from statsmodels.tsa.arima.model import ARIMA

WINDOW_SIZE = 10
TIME_INTERVAL_MIN = 5

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
    FAILURE_TEMP = df["temperature"].quantile(0.98)
    if len(df) < WINDOW_SIZE:
        last_temp = df["temperature"].iloc[-1] if len(df) > 0 else 0

        return {
            "forecast": [],
            "failure_in_minutes": None,
            "failure_in_hours": None,
            "failure_reason": None,
            "risk": "low",
            "next_temp": round(float(last_temp), 2)
        }

    recent = df.tail(WINDOW_SIZE)
    base_row = recent.iloc[-1].to_dict()

    # 🔥 ARIMA forecast
    temp_series = recent["temperature"]
    temp_forecast = forecast_temperature_arima(temp_series, steps)

    forecast = []
    time_axis = []

    failure_time = None
    failure_reason = None

    for i in range(steps):
        future_row = {}

        for col in FEATURE_COLUMNS:

            if col == "temperature":
                future_val = temp_forecast[i]
            else:
                base_val = float(base_row.get(col, 0))
                slope_val = 0
                future_val = base_val + slope_val * (i + 1)

            future_row[col] = future_val

        forecast.append({
            col: round(float(future_row[col]), 2)
            for col in FEATURE_COLUMNS
        })

        minutes = (i + 1) * TIME_INTERVAL_MIN
        time_axis.append(minutes)

        if failure_time is None:

            if future_row["temperature"] >= FAILURE_TEMP:
                failure_time = minutes
                failure_reason = "threshold_breach"

    failure_hours = round(failure_time / 60, 2) if failure_time else None

    # Risk logic
    if failure_time is None:
        risk = "low"
    elif failure_hours < 12:
        risk = "critical"
    elif failure_hours < 48:
        risk = "high"
    else:
        risk = "medium"

    return {
        "forecast": forecast,
        "failure_in_minutes": failure_time,
        "failure_in_hours": failure_hours,
        "failure_reason": failure_reason,
        "risk": risk,
        "next_temp": round(float(temp_forecast[0]), 2)
    }
def forecast_temperature_arima(series, steps):
    try:
        series = series.astype(float).dropna()

        if len(series) < 20:
            raise Exception("Not enough data - arima")

        # (p,d,q) = (2,1,2) → safe default
        model = ARIMA(series, order=(2, 1, 2))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=steps)

        return forecast.tolist()

    except Exception as e:
        print("arima fallback:", e)

        # fallback → simple slope
        last = series.iloc[-1]
        return [last for _ in range(steps)]