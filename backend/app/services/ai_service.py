import requests
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def generate_explanation(data, severity, prediction):
    try:
        if not OPENROUTER_API_KEY:
            return fallback_explanation(data, severity, prediction)

        prompt = f"""
You are an industrial predictive maintenance expert.

Analyze the machine sensor data and explain:

1. What is happening
2. Why it is happening
3. What action should be taken

DATA:
Temperature: {data.get('temperature')}
Vibration: {data.get('vibration')}
Pressure: {data.get('pressure')}
Humidity: {data.get('humidity')}
RPM: {data.get('rpm')}
Voltage: {data.get('voltage')}
Current: {data.get('current')}

ML OUTPUT:
Severity: {severity}
Risk: {prediction.get('risk')}
Failure in minutes: {prediction.get('failure_in_minutes')}
Failure in hours: {prediction.get('failure_in_hours')}

Keep explanation concise and practical.
"""

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistralai/mistral-7b-instruct",  # cheap + good
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)

        if response.status_code != 200:
            print("OpenRouter error:", response.text)
            return fallback_explanation(data, severity, prediction)

        result = response.json()

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        print("AI error:", e)
        return fallback_explanation(data, severity, prediction)


# =========================
# 🔁 FALLBACK (IMPORTANT)
# =========================
def fallback_explanation(data, severity, prediction):
    temp = data.get("temperature", 0)
    vib = data.get("vibration", 0)

    if severity == "critical":
        return (
            f"⚠️ Critical condition detected. "
            f"Temperature ({temp}°C) and vibration ({vib}) indicate abnormal machine behavior. "
            f"Immediate inspection recommended to prevent failure."
        )

    if prediction.get("failure_in_minutes"):
        return (
            f"⚠️ Potential failure expected in {prediction['failure_in_minutes']} minutes. "
            f"Monitor system closely and prepare maintenance."
        )

    return (
        f"✅ System operating within normal parameters. "
        f"Temperature and vibration levels are stable."
    )