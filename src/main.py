"""ML churn service"""
from typing import get_type_hints
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from sklearn.metrics import accuracy_score, f1_score

from src.schemas import (
    PredictionRequestChurn,
    PredictionResponseChurn,
    TrainingConfigChurn,
    FeatureVectorChurn
)
from src.utils.dataset import load_churn_dataset, dataset_info
from src.utils.preprocessing import prepare_data, get_split_info, \
    load_raw_splits, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.utils.model_factory import build_churn_pipeline, \
    resolve_hyperparameters
from src.utils.model_manipulation import save_churn_model, save_model_metadata
from src import model_store
from src.utils.history_recorder import (
    append_training_record,
    build_training_record,
    compute_roc_auc,
    load_history,
)
from src.error_handlers import (
    ErrorResponseChurn,
    register_error_handlers,
)
from src.utils.log_control import get_logger


log = get_logger(__name__)

app = FastAPI()
register_error_handlers(app)


@app.get("/health")
def health_check():
    """
    Returns service liveness plus readiness signals:
    - **model_ready**   — whether a trained model is in memory.
    - **dataset_ready** — whether the dataset file is accessible.
    """
    # ── Dataset probe ─────────────────────────────────────────
    dataset_ready = False
    dataset_detail = None
    try:
        load_churn_dataset()
        dataset_ready = True
    except (FileNotFoundError, ValueError) as exc:
        dataset_detail = str(exc)
        log.warning("Health: dataset not accessible: %s", exc)

    # ── Model probe ───────────────────────────────────────────
    model_ready = model_store.model is not None
    model_type = None
    if model_ready and model_store.metadata:
        model_type = model_store.metadata.get("model_type")

    status = "ok" if (dataset_ready and model_ready) else "degraded"
    log.info("Health check: status=%s model=%s dataset=%s",
             status, model_ready, dataset_ready)

    return {
        "status":         status,
        "model_ready":    model_ready,
        "model_type":     model_type,
        "dataset_ready":  dataset_ready,
        "dataset_detail": dataset_detail,
    }


# ── Prediction ───────────────────────────────────────────────────────────────

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
        409: {
            "description": "Model has not been trained yet.",
            "model": ErrorResponseChurn,
            "content": {
                "application/json": {
                    "example": {
                        "code":    "model_not_ready",
                        "message": (
                            "No trained model is available. "
                            "Please call POST /model/train first."
                        ),
                        "details": None,
                    }
                }
            },
        },
        422: {
            "description": (
                "Validation error — wrong types, missing or "
                "extra fields."
            ),
            "model": ErrorResponseChurn,
            "content": {
                "application/json": {
                    "examples": {
                        "missing_field": {
                            "summary": "Required field missing",
                            "value": {
                                "code":    "validation_error",
                                "message": (
                                    "1 validation error(s) in "
                                    "the request body."
                                ),
                                "details": [
                                    {
                                        "field": (
                                            "body → clients → 0 → "
                                            "monthly_fee"
                                        ),
                                        "issue": "missing",
                                        "message": "Field required",
                                    }
                                ],
                            },
                        },
                        "wrong_type": {
                            "summary": "Wrong value type",
                            "value": {
                                "code":    "validation_error",
                                "message": (
                                    "1 validation error(s) in "
                                    "the request body."
                                ),
                                "details": [
                                    {
                                        "field": (
                                            "body → clients → 0 → "
                                            "support_requests"
                                        ),
                                        "issue": "int_parsing_error",
                                        "message": (
                                            "Input should be a "
                                            "valid integer"
                                        ),
                                    }
                                ],
                            },
                        },
                        "empty_clients": {
                            "summary": "Empty clients list",
                            "value": {
                                "code":    "validation_error",
                                "message": (
                                    "1 validation error(s) in "
                                    "the request body."
                                ),
                                "details": [
                                    {
                                        "field": "body → clients",
                                        "issue": "too_short",
                                        "message": (
                                            "List should have at "
                                            "least 1 item"
                                        ),
                                    }
                                ],
                            },
                        },
                    }
                }
            },
        },
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

    n_clients = len(request.clients)
    log.info("Prediction request: %d client(s)", n_clients)

    X = pd.DataFrame([client.model_dump() for client in request.clients])

    classes = model_store.model.predict(X)
    probabilities = model_store.model.predict_proba(X)   # shape (n, 2)

    results = [
        PredictionResponseChurn(
            churn_class=int(cls),
            probability_churn=round(float(proba[1]), 4),
            probability_no_churn=round(float(proba[0]), 4),
        )
        for cls, proba in zip(classes, probabilities)
    ]
    churn_count = sum(r.churn_class for r in results)
    log.info("Prediction done: %d/%d predicted to churn",
             churn_count, n_clients)
    return results


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
    return dataset_info()


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

_TRAIN_EXAMPLES = {
    "logreg_defaults": {
        "summary": "Logistic Regression — default hyperparameters",
        "value": {
            "model_type": "logreg",
            "hyperparameters": {},
        },
    },
    "logreg_custom": {
        "summary": "Logistic Regression — stronger regularisation",
        "value": {
            "model_type": "logreg",
            "hyperparameters": {"C": 0.1, "max_iter": 500},
        },
    },
    "random_forest_defaults": {
        "summary": "Random Forest — default hyperparameters",
        "value": {
            "model_type": "random_forest",
            "hyperparameters": {},
        },
    },
    "random_forest_custom": {
        "summary": "Random Forest — deeper trees, more estimators",
        "value": {
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 200, "max_depth": 10},
        },
    },
}


@app.post(
    "/model/train",
    responses={
        422: {
            "description": "Dataset problem or invalid hyperparameters.",
            "model": ErrorResponseChurn,
            "content": {
                "application/json": {
                    "examples": {
                        "empty_dataset": {
                            "summary": "Dataset is empty",
                            "value": {
                                "code":    "data_error",
                                "message": (
                                    "Dataset is empty after dropping "
                                    "rows with missing target."
                                ),
                                "details": None,
                            },
                        },
                        "dataset_not_found": {
                            "summary": "Dataset file not found",
                            "value": {
                                "code":    "not_found",
                                "message": (
                                    "Dataset file not found: "
                                    "data/churn_dataset.csv"
                                ),
                                "details": None,
                            },
                        },
                        "bad_hyperparameters": {
                            "summary": "Invalid hyperparameter",
                            "value": {
                                "code":    "unprocessable_entity",
                                "message": (
                                    "Invalid hyperparameters for "
                                    "'logreg': __init__() got an "
                                    "unexpected keyword argument "
                                    "'unknown_param'"
                                ),
                                "details": None,
                            },
                        },
                    }
                }
            },
        },
    },
)
def train_model(
    config: TrainingConfigChurn = Body(openapi_examples=_TRAIN_EXAMPLES),
):
    """
    Trains a churn prediction model according to the supplied configuration.

    - `model_type` — `"logreg"` (LogisticRegression) or `"random_forest"`
        (RandomForestClassifier).
    - `hyperparameters` — optional overrides; any key not supplied falls
        back to the model default.

    After training the model and its metadata are saved to disk so they survive
    a service restart. The in-memory store is updated immediately so
    `/predict` and `/model/status` reflect the new model without a restart.
    """
    log.info("Training started: model_type=%s hyperparameters=%s",
             config.model_type, config.hyperparameters)
    try:
        X_train, X_test, y_train, y_test = load_raw_splits()
    except (FileNotFoundError, ValueError) as e:
        log.error("Training failed — dataset error: %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Merge caller overrides with per-model defaults
    resolved_params = resolve_hyperparameters(
        config.model_type, config.hyperparameters
    )

    try:
        pipeline = build_churn_pipeline(config.model_type, resolved_params)
        pipeline.fit(X_train, y_train)
    except (TypeError, ValueError) as e:
        # Catches unknown or incompatible hyperparameter keys
        log.error("Training failed — bad hyperparameters for %s: %s",
                  config.model_type, e)
        raise HTTPException(
            status_code=422,
            detail=f"Invalid hyperparameters for '{config.model_type}': {e}",
        ) from e

    y_pred = pipeline.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    f1 = round(f1_score(y_test, y_pred), 4)

    save_churn_model(pipeline)
    metadata = save_model_metadata(
        accuracy=accuracy,
        f1=f1,
        model_type=config.model_type,
        hyperparameters=resolved_params,
    )
    model_store.update(pipeline, metadata)

    roc_auc = compute_roc_auc(pipeline, X_test, y_test)
    record = build_training_record(
        model_type=config.model_type,
        hyperparameters=resolved_params,
        accuracy=accuracy,
        f1=f1,
        roc_auc=roc_auc,
    )
    append_training_record(record)

    log.info(
        "Training complete: model=%s accuracy=%.4f f1=%.4f roc_auc=%.4f",
        config.model_type, accuracy, f1, roc_auc,
    )
    return {
        "message": "Model trained and saved successfully.",
        "model_type": config.model_type,
        "hyperparameters": resolved_params,
        "accuracy": accuracy,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }


@app.get("/model/status")
def model_status():
    """
    Returns whether a trained model is available, when it was last trained,
    which model type was used, its hyperparameters, and the test-set metrics.
    """
    if model_store.model is None:
        return {
            "trained":         False,
            "trained_at":      None,
            "model_type":      None,
            "hyperparameters": None,
            "metrics":         None,
        }

    meta = model_store.metadata or {}
    return {
        "trained":         True,
        "trained_at":      meta.get("trained_at"),
        "model_type":      meta.get("model_type"),
        "hyperparameters": meta.get("hyperparameters"),
        "metrics":         meta.get("metrics"),
    }


# ── Model schema ─────────────────────────────────────────────────────────────

@app.get("/model/schema")
def model_schema():
    """
    Returns the list of features the model expects, their value types, and
    which group each belongs to (numeric vs categorical).

    Clients can use this response to validate and construct prediction requests
    correctly before calling POST /predict.

    Response structure:
    - `features`       — ordered list of feature descriptors.
    - `input_model`    — name of the Pydantic schema that /predict accepts.
    - `note`           — reminder that feature names
        and types must match exactly.
    """

    hints: dict[str, type] = get_type_hints(FeatureVectorChurn)
    type_names = {int: "int", float: "float", str: "str"}

    numeric_features = [
        {
            "name":        name,
            "type":        type_names.get(hints[name], str(hints[name])),
            "group":       "numeric",
            "description": "Scaled with StandardScaler during training.",
        }
        for name in NUMERIC_FEATURES
    ]

    categorical_features = [
        {
            "name":        name,
            "type":        type_names.get(hints[name], str(hints[name])),
            "group":       "categorical",
            "description": "One-hot encoded during training.",
        }
        for name in CATEGORICAL_FEATURES
    ]

    return {
        "input_model": "PredictionRequestChurn",
        "note": (
            "Feature names and types must exactly match this schema. "
            "Unknown categorical values are handled gracefully (OHE encodes "
            "them as all-zeros); unknown field names are rejected by the "
            "PredictionRequestChurn validator."
        ),
        "features": numeric_features + categorical_features,
    }


# ── Model metrics ────────────────────────────────────────────────────────────

@app.get("/model/metrics")
def model_metrics(
    model_type: str | None = Query(
        default=None,
        description="Filter by model type: 'logreg' or 'random_forest'.",
    ),
    last_n: int = Query(
        default=10,
        ge=1,
        le=100,
        description="How many of the most recent records to return.",
    ),
):
    """
    Returns training history from training_history.json.

    - **model_type** — optional filter; omit to see all model types.
    - **last_n** — number of most recent runs to return (default 10, max 100).

    Each record contains:
    - `trained_at`      — UTC timestamp of the training run.
    - `model_type`      — which classifier was used.
    - `hyperparameters` — full resolved hyperparameter dict.
    - `metrics`         — accuracy, f1_score, roc_auc on the test set.

    Use this endpoint to compare runs across different model types and
    hyperparameter settings.
    """
    records = load_history(model_type=model_type, last_n=last_n)

    latest = records[0] if records else None

    return {
        "total_returned": len(records),
        "filter": {
            "model_type": model_type,
            "last_n":     last_n,
        },
        "latest_run": latest,
        "history":    records,
    }
