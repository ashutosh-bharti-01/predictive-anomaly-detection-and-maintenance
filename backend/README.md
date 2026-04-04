
This is a FastAPI backend for:
- Uploading and storing IoT sensor CSV data in MongoDB
- Running anomaly detection (IsolationForest)
- Forecasting temperature trend (ARIMA) and estimating risk
- (Optional) Generating an AI explanation using the OpenAI API

The app uses `sensor_data.csv` as a default local dataset, with MongoDB as a fallback data source.


Prerequisites
- Python 3.10+ (recommended)
- MongoDB running locally on `mongodb://localhost:27017/` (default config in `app/db/mongo.py`)

Required (for API access):
- `ALLOWED_EMAILS` (comma-separated allowlist; only these emails can access the APIs)
- `AUTH_PASSWORD` (shared password for allowed emails)
- `JWT_SECRET_KEY` (server-side secret for signing tokens)

Optional (for AI explanation):
- OpenRouter API key in environment variable `OPENROUTER_API_KEY`


Install
PowerShell (Windows):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


Run
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8081 --reload
```

API docs:
- Swagger UI: `http://localhost:8081/docs`
- ReDoc: `http://localhost:8081/redoc`


Data Format
CSV columns used by the model:
- `temperature`
- `vibration`
- `pressure`
- `humidity`
- `rpm`
- `voltage`
- `current`

Common column:
- `timestamp` (optional, but recommended for history)

Optional label column:
- `anomaly` (0/1). If present during training, rows with `anomaly == 0` are used as "normal" data.


Routes
Base URL: `http://localhost:8081`

1) POST `/auth/login`
- Exchanges email + password for a bearer token.
- Body: JSON `{ "email": "...", "password": "..." }`

Example:
```bash
curl -X POST http://localhost:8081/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"you@example.com\",\"password\":\"your-password\"}"
```

Use the returned token for all other routes:
```bash
curl http://localhost:8081/history \
  -H "Authorization: Bearer <TOKEN>"
```

2) POST `/upload`
- Upload a CSV and store rows in MongoDB (`iot_db.sensor_data`).
- Body: multipart form-data with field `file` (required).

Example:
```bash
curl -F "file=@sensor_data.csv" http://localhost:8081/upload \
  -H "Authorization: Bearer <TOKEN>"
```

3) GET `/history`
- Returns the last ~200 rows for charting.
- Data source priority:
  1. Local `sensor_data.csv` if present
  2. MongoDB fallback

Example:
```bash
curl http://localhost:8081/history \
  -H "Authorization: Bearer <TOKEN>"
```

4) POST `/generate`
- Generates a mixed dataset and overwrites local `sensor_data.csv`.
- Query params:
  - `rows` (default 500, min 10, max 20000)
  - `interval_seconds` (default 10, min 1, max 3600)
  - `anomaly_rate` (default 0.06, min 0.0, max 0.5)
  - `seed` (optional int)

Example:
```bash
curl -X POST "http://localhost:8081/generate?rows=500&interval_seconds=10&anomaly_rate=0.05&seed=42" \
  -H "Authorization: Bearer <TOKEN>"
```

5) POST `/generate-normal`
- Generates a normal-only dataset (`anomaly = 0` for all rows) and overwrites local `sensor_data.csv`.
- Query params:
  - `rows` (default 500, min 10, max 20000)
  - `interval_seconds` (default 10, min 1, max 3600)
  - `seed` (optional int)

Example:
```bash
curl -X POST "http://localhost:8081/generate-normal?rows=500&interval_seconds=10&seed=42" \
  -H "Authorization: Bearer <TOKEN>"
```

6) POST `/generate-anomaly`
- Generates a failure-pattern dataset and writes to local `anomaly_data.csv`.
- Query params:
  - `rows` (default 300, min 50, max 5000)
  - `interval_seconds` (default 10, min 1, max 3600)
  - `seed` (optional int)

Example:
```bash
curl -X POST "http://localhost:8081/generate-anomaly?rows=300&interval_seconds=10&seed=42" \
  -H "Authorization: Bearer <TOKEN>"
```

7) POST `/train-model`
- Trains and saves the IsolationForest model to `models/isolation_forest.pkl`.
- Data source priority:
  1. Uploaded CSV (multipart `file`, optional)
  2. Local `sensor_data.csv` if present
  3. MongoDB fallback

Example (train from uploaded CSV):
```bash
curl -X POST -F "file=@sensor_data.csv" http://localhost:8081/train-model \
  -H "Authorization: Bearer <TOKEN>"
```

8) GET `/model-status`
- Returns whether the model file exists and when it was last modified.

Example:
```bash
curl http://localhost:8081/model-status \
  -H "Authorization: Bearer <TOKEN>"
```

9) POST `/predict`
- Predicts anomaly + forecast/risk for the latest row.
- Data source priority:
  1. Uploaded CSV (multipart `file`, optional)
  2. Local `sensor_data.csv` if present
  3. MongoDB fallback

Returns (high-level):
- Latest row values
- `anomaly` (-1 for anomaly, 1 for normal) and `score`
- `prediction`: ARIMA-based temperature forecast and `risk`
- `explanation`: optional AI explanation (currently stubbed)

Example:
```bash
curl -X POST -F "file=@sensor_data.csv" http://localhost:8081/predict \
  -H "Authorization: Bearer <TOKEN>"
```


Notes
- MongoDB connection is currently hardcoded to `localhost:27017`. Change `app/db/mongo.py` if you need a different URI.
- If you want AI explanations, wire `app/services/ai_service.py` to use `OpenAI(api_key=os.getenv("OPENAI_API_KEY"))` and set `OPENAI_API_KEY` in your environment.
- `app/services/OPENAI.py` contains an example OpenAI call; do not commit API keys to source control.
