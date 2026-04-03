# Churn Prediction Service

A production-ready REST API that trains machine learning models to predict customer churn and serves real-time predictions. Built with FastAPI, scikit-learn, and Docker.

---

## What it does

The service ingests a customer dataset, trains a classification model (Logistic Regression or Random Forest), and exposes endpoints to:

- predict whether individual customers will churn
- inspect model status, metrics, and training history
- retrain the model with different hyperparameters without restarting the service

---

## Project structure

```
churn-service/
├── data/
│   └── churn_dataset.csv          # Training data
├── models/                        # Persisted artifacts (git-ignored)
│   ├── churn_model.joblib         # Trained sklearn Pipeline
│   ├── churn_model_metadata.json  # Last training run metadata
│   └── training_history.json      # Append-only training history
├── src/
│   ├── utils/
│   │   ├── dataset.py             # CSV loading and validation
│   │   ├── preprocessing.py       # Feature engineering and splitting
│   │   ├── logreg.py              # ChurnPreprocessor (custom sklearn transformer)
│   │   ├── transformer_universal.py # ColumnTransformer-based pipeline builder
│   │   ├── model_factory.py       # Model selection and hyperparameter defaults
│   │   ├── model_manipulation.py  # joblib save / load for model and metadata
│   │   ├── history_recorder.py    # Append-only JSON training history
│   │   └── log_control.py         # Centralised logging configuration
│   ├── schemas.py                 # Pydantic request / response models
│   ├── error_handlers.py          # Global exception handlers and error format
│   ├── model_store.py             # In-memory model state (loaded once on import)
│   └── main.py                    # FastAPI app and all endpoints
├── tests/
│   ├── conftest.py                # Shared fixtures and test isolation
│   ├── test_dataset.py
│   ├── test_preprocessing.py
│   ├── test_logreg.py
│   ├── test_model_manipulation.py
│   ├── test_history_recorder.py
│   └── test_api.py                # Integration tests with TestClient
├── Dockerfile
├── .dockerignore
├── pytest.ini
└── requirements.txt
```

---

## Dataset format

The service expects a CSV file at `data/churn_dataset.csv` with the following columns:

| Column | Type | Description |
|---|---|---|
| `monthly_fee` | float | Monthly subscription fee |
| `usage_hours` | float | Average hours of product usage per month |
| `support_requests` | int | Number of support tickets opened |
| `account_age_months` | int | How long the customer has been subscribed |
| `failed_payments` | int | Number of failed payment attempts |
| `region` | str | Customer's geographic region |
| `device_type` | str | Primary device used (e.g. mobile, desktop) |
| `payment_method` | str | Payment method (e.g. card, bank transfer) |
| `autopay_enabled` | int | Whether autopay is active: `1` = yes, `0` = no |
| `churn` | int | Target label: `1` = churned, `0` = retained |

Missing values in features are imputed automatically (median for numeric, mode for categorical). Rows with a missing `churn` label are dropped.

---

## Running locally

**Requirements:** Python 3.12+

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd churn-service

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the service
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at `http://localhost:8000`.  
Interactive documentation: `http://localhost:8000/docs`

---

## Running with Docker

```bash
# Build the image
docker build -t churn-service .

# Run — mount models/ as a volume so trained models survive restarts
docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-service
```

The service is available at `http://localhost:8000`.  
To run in the background, add `-d` to the `docker run` command.

---

## Running tests

```bash
# All tests
pytest

# Unit tests only (no HTTP)
pytest tests/ -k "not TestApi"

# Single file
pytest tests/test_preprocessing.py -v

# Stop on first failure
pytest -x
```

Tests use synthetic data and isolated `tmp_path` directories — they never touch `data/`, `models/`, or `training_history.json`.

---

## API reference

### `GET /health`

Returns service liveness and readiness.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_ready": true,
  "model_type": "logreg",
  "dataset_ready": true,
  "dataset_detail": null
}
```

`status` is `"ok"` only when both the dataset and a trained model are available. Monitoring probes should check this field.

---

### `POST /model/train`

Trains a model and saves it to disk. Accepts an optional training configuration.

**Train with default Logistic Regression:**
```bash
curl -X POST http://localhost:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "hyperparameters": {}}'
```

**Train with stronger regularisation:**
```bash
curl -X POST http://localhost:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "hyperparameters": {"C": 0.1, "max_iter": 500}}'
```

**Train a Random Forest:**
```bash
curl -X POST http://localhost:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 200, "max_depth": 10}}'
```

**Response:**
```json
{
  "message": "Model trained and saved successfully.",
  "model_type": "logreg",
  "hyperparameters": {"C": 1.0, "max_iter": 1000, "random_state": 42, "class_weight": "balanced"},
  "accuracy": 0.765,
  "f1_score": 0.482,
  "roc_auc": 0.831
}
```

Supported `model_type` values: `"logreg"`, `"random_forest"`.  
Hyperparameter keys not supplied fall back to the model's defaults.

---

### `POST /predict`

Predicts churn for one or more clients. Requires a trained model.

**Single client:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "clients": [{
      "monthly_fee": 95.0,
      "usage_hours": 3.0,
      "support_requests": 8,
      "account_age_months": 6,
      "failed_payments": 3,
      "region": "south",
      "device_type": "mobile",
      "payment_method": "card",
      "autopay_enabled": 0
    }]
  }'
```

**Batch (multiple clients in one request):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "clients": [
      {"monthly_fee": 95.0, "usage_hours": 3.0, "support_requests": 8,
       "account_age_months": 6, "failed_payments": 3, "region": "south",
       "device_type": "mobile", "payment_method": "card", "autopay_enabled": 0},
      {"monthly_fee": 30.0, "usage_hours": 40.0, "support_requests": 0,
       "account_age_months": 48, "failed_payments": 0, "region": "north",
       "device_type": "desktop", "payment_method": "bank", "autopay_enabled": 1}
    ]
  }'
```

**Response:**
```json
[
  {"churn_class": 1, "probability_churn": 0.83, "probability_no_churn": 0.17},
  {"churn_class": 0, "probability_churn": 0.12, "probability_no_churn": 0.88}
]
```

- `churn_class` — `1` = predicted to churn, `0` = predicted to stay
- `probability_churn` — model confidence for churn
- `probability_no_churn` — model confidence for retention

If no model has been trained yet, the endpoint returns `409` with `"code": "model_not_ready"`.

---

### Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/model/status` | Model availability, type, hyperparameters, and last metrics |
| `GET` | `/model/metrics` | Training history with optional `?model_type=` and `?last_n=` filters |
| `GET` | `/model/schema` | Expected feature names and types for `/predict` |
| `GET` | `/dataset/preview` | First N rows of the dataset (`?n=10`) |
| `GET` | `/dataset/info` | Row count, column list, and churn class distribution |
| `GET` | `/dataset/split-info` | Train / test split sizes and class distribution per split |
| `GET` | `/docs` | Interactive Swagger UI with request examples |

---

## Error format

All errors return the same JSON envelope regardless of origin:

```json
{
  "code": "validation_error",
  "message": "1 validation error(s) in the request body.",
  "details": [
    {
      "field": "body → clients → 0 → monthly_fee",
      "issue": "missing",
      "message": "Field required"
    }
  ]
}
```

Common codes: `validation_error`, `model_not_ready`, `not_found`, `data_error`, `internal_server_error`.