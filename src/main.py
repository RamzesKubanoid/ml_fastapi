"""ML churn service"""
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from sklearn.metrics import accuracy_score, f1_score

from src.schemas import (
    PredictionRequestChurn,
    PredictionResponseChurn,
)
from src.utils.dataset import load_churn_dataset, dataset_info
from src.utils.preprocessing import prepare_data, get_split_info, \
    load_raw_splits
from src.utils.logreg import train_churn_model
from src.utils.model_manipulation import save_churn_model, save_model_metadata
from src import model_store


app = FastAPI()


@app.get("/health")
def health_check():
    """
    Server health check
    """
    return {
        "message": "ml churn service is running"
    }


# ── Prediction ───────────────────────────────────────────────────────────────

# OpenAPI examples shown in /docs under "Try it out"
_PREDICT_EXAMPLES = {
    "single_likely_churn": {
        "summary": "Single client — likely to churn",
        "description": (
            "High monthly fee, many support requests, failed payments, "
            "and no autopay. Model should return a high churn probability."
        ),
        "value": {
            "clients": [
                {
                    "monthly_fee": 95.0,
                    "usage_hours": 3.0,
                    "support_requests": 8,
                    "account_age_months": 6,
                    "failed_payments": 3,
                    "region": "africa",
                    "device_type": "mobile",
                    "payment_method": "card",
                    "autopay_enabled": 0,
                }
            ]
        },
    },
    "single_likely_stay": {
        "summary": "Single client — likely to stay",
        "description": (
            "Low fee, high usage, long tenure, autopay enabled, "
            "no failed payments. Model should return a low churn probability."
        ),
        "value": {
            "clients": [
                {
                    "monthly_fee": 30.0,
                    "usage_hours": 40.0,
                    "support_requests": 0,
                    "account_age_months": 48,
                    "failed_payments": 0,
                    "region": "america",
                    "device_type": "tablet",
                    "payment_method": "crypto",
                    "autopay_enabled": 1,
                }
            ]
        },
    },
    "batch_multiple_clients": {
        "summary": "Batch — three clients at once",
        "description": "Demonstrates batch prediction in one request.",
        "value": {
            "clients": [
                {
                    "monthly_fee": 95.0,
                    "usage_hours": 3.0,
                    "support_requests": 8,
                    "account_age_months": 6,
                    "failed_payments": 3,
                    "region": "africa",
                    "device_type": "mobile",
                    "payment_method": "card",
                    "autopay_enabled": 0,
                },
                {
                    "monthly_fee": 30.0,
                    "usage_hours": 40.0,
                    "support_requests": 0,
                    "account_age_months": 48,
                    "failed_payments": 0,
                    "region": "america",
                    "device_type": "tablet",
                    "payment_method": "crypto",
                    "autopay_enabled": 1,
                },
                {
                    "monthly_fee": 60.0,
                    "usage_hours": 18.0,
                    "support_requests": 2,
                    "account_age_months": 20,
                    "failed_payments": 1,
                    "region": "europe",
                    "device_type": "desktop",
                    "payment_method": "paypal",
                    "autopay_enabled": 0,
                },
            ]
        },
    },
}


@app.post(
    "/predict",
    response_model=list[PredictionResponseChurn],
    summary="Predict churn for one or more clients",
    responses={
        200: {
            "description": "Prediction results — one entry per client.",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "churn_class": 1,
                            "probability_churn": 0.83,
                            "probability_no_churn": 0.17,
                        }
                    ]
                }
            },
        },
        409: {"description": "Model has not been trained yet."},
        422: {"description": "Invalid input data."},
    },
)
def predict(
    request: PredictionRequestChurn = Body(openapi_examples=_PREDICT_EXAMPLES),
) -> list[PredictionResponseChurn]:
    """
    Accepts **one or more** `FeatureVectorChurn` objects and returns a
    predicted churn class and class probabilities for each client.

    - `churn_class` — `1` if the client is predicted to churn, `0` otherwise.
    - `probability_churn` — model confidence the client **will** churn.
    - `probability_no_churn` — model confidence the client **will stay**.

    Pass a list with a single item to predict for one client.
    """
    if model_store.model is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No trained model is available. "
                "Please call POST /model/train first."
            ),
        )

    X = pd.DataFrame([client.model_dump() for client in request.clients])

    classes = model_store.model.predict(X)
    probabilities = model_store.model.predict_proba(X)   # shape (n, 2)

    return [
        PredictionResponseChurn(
            churn_class=int(cls),
            probability_churn=round(float(proba[1]), 4),
            probability_no_churn=round(float(proba[0]), 4),
        )
        for cls, proba in zip(classes, probabilities)
    ]


# ── Dataset ──────────────────────────────────────────────────────────────────


@app.get("/dataset/preview")
def preview_dataset(
    n: int = Query(default=10, ge=1, description="Number of rows to return"),
):
    """
    Returns the first N rows of the churn dataset as JSON.

    - **n**: how many rows to return (default: 10, must be ≥ 1)
    """
    try:
        df = load_churn_dataset()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return df.head(n).to_dict(orient="records")


@app.get("/dataset/info")
def get_dataset_info():
    """
    Returns info about dataset
    """
    result = dataset_info()
    return result


@app.get("/dataset/split-info")
def split_info():
    """
    Returns train/test split sizes and churn class distribution for both splits
    """
    try:
        _, _, y_train, y_test = prepare_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return get_split_info(y_train, y_test)


# ── Model ────────────────────────────────────────────────────────────────────


@app.post("/model/train")
def train_model():
    """
    Trains a LogisticRegression model on churn_dataset.csv and returns
    accuracy and F1 metrics evaluated on the held-out test set.
    """
    try:
        X_train, X_test, y_train, y_test = load_raw_splits()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    model = train_churn_model(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)

    save_churn_model(model)
    metadata = save_model_metadata(accuracy, f1)

    model_store.update(model, metadata)

    return {
        "message":  "Model trained and saved successfully.",
        "accuracy": accuracy,
        "f1_score": f1,
    }


@app.get("/model/status")
def model_status():
    """
    Returns whether a trained model is available, when it was last trained,
    and the metrics it achieved on the test set.
    """
    if model_store.model is None:
        return {
            "trained":    False,
            "trained_at": None,
            "metrics":    None,
        }

    meta = model_store.metadata or {}
    return {
        "trained":    True,
        "trained_at": meta.get("trained_at"),
        "metrics":    meta.get("metrics"),
    }
