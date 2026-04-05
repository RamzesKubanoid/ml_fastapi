"""Model endpoints"""
from typing import get_type_hints

from fastapi import APIRouter, Body, HTTPException, Query
from sklearn.metrics import accuracy_score, f1_score

from src.schemas import TrainingConfigChurn, FeatureVectorChurn
from src.error_handlers import ErrorResponseChurn
from src.ml.preprocessing import (
    load_raw_splits,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.ml.model_factory import build_churn_pipeline, \
    resolve_hyperparameters
from src.core.model_manipulation import save_churn_model, save_model_metadata
from src.core.history_recorder import (
    append_training_record,
    build_training_record,
    compute_roc_auc,
    load_history,
)
from src.core.log_control import get_logger
from src.core import model_store

log = get_logger(__name__)
router = APIRouter()

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


@router.post(
    "/train",
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
                                "code": "data_error",
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
                                "code": "not_found",
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
                                "code": "unprocessable_entity",
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

    - `model_type` — `"logreg"` (LogisticRegression)
        or `"random_forest"` (RandomForestClassifier).
    - `hyperparameters` — optional overrides; any key not supplied
        falls back to the model default.

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

    resolved_params = resolve_hyperparameters(
        config.model_type, config.hyperparameters)

    try:
        pipeline = build_churn_pipeline(config.model_type, resolved_params)
        pipeline.fit(X_train, y_train)
    except (TypeError, ValueError) as e:
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

    log.info("Training complete: model=%s accuracy=%.4f f1=%.4f roc_auc=%.4f",
             config.model_type, accuracy, f1, roc_auc)

    return {
        "message":         "Model trained and saved successfully.",
        "model_type":      config.model_type,
        "hyperparameters": resolved_params,
        "accuracy":        accuracy,
        "f1_score":        f1,
        "roc_auc":         roc_auc,
    }


@router.get("/status")
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


@router.get("/schema")
def model_schema():
    """
    Returns the list of features the model expects, their value types, and
    which group each belongs to (numeric vs categorical).

    Clients can use this response to validate and construct prediction requests
    correctly before calling POST /predict.

    Response structure:
    - `features`    — ordered list of feature descriptors.
    - `input_model` — name of the Pydantic schema that /predict accepts.
    - `note`        — reminder that feature names and types must match exactly.
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


@router.get("/metrics")
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
    - **last_n**     — number of most recent runs
        to return (default 10, max 100).

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
        "filter": {"model_type": model_type, "last_n": last_n},
        "latest_run": latest,
        "history":    records,
    }
