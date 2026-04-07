"""Prediction related endpoint"""
from typing import Union
import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from src.schemas import FeatureVectorChurn, PredictionResponseChurn
from src.error_handlers import ErrorResponseChurn
from src.core.log_control import get_logger
from src.core import model_store

log = get_logger(__name__)
router = APIRouter()

_PREDICT_EXAMPLES = {
    "single_object_churn": {
        "summary": "Single object — likely to churn",
        "description": (
            "Send a single FeatureVectorChurn object directly. "
            "High fee, many sup requests, no autopay → high churn probability."
        ),
        "value": {
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
    },
    "single_object_stay": {
        "summary": "Single object — likely to stay",
        "description": (
            "Send a single FeatureVectorChurn object directly. "
            "Low fee, high usage, autopay on → low churn probability."
        ),
        "value": {
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
    },
    "batch_list": {
        "summary": "Batch — list of three clients",
        "description": "Send a JSON array to predict for many clients at once",
        "value": [
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
        ],
    },
}


@router.post(
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
    request: Union[FeatureVectorChurn, list[FeatureVectorChurn]] = Body(
        openapi_examples=_PREDICT_EXAMPLES,
    ),
) -> list[PredictionResponseChurn]:
    """
    Accepts a **single** `FeatureVectorChurn` object or a **list** of them
    and returns a predicted churn class and class probabilities per client.

    - `churn_class` — `1` = predicted to churn, `0` = predicted to stay.
    - `probability_churn` — model confidence the client **will** churn.
    - `probability_no_churn` — model confidence the client **will stay**.

    Single object:
    ```json
    {"monthly_fee": 50.0, "usage_hours": 20.0, ...}
    ```

    List of objects:
    ```json
    [{"monthly_fee": 50.0, ...}, {"monthly_fee": 80.0, ...}]
    ```
    """
    if model_store.model is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No trained model is available. "
                "Please call POST /model/train first."
            ),
        )

    clients = request if isinstance(request, list) else [request]
    n_clients = len(clients)

    if len(clients) == 0:
        raise HTTPException(
            status_code=422,
            detail="At least one client must be provided for prediction.",
        )

    log.info("Prediction request: %d client(s)", n_clients)

    X = pd.DataFrame([client.model_dump() for client in clients])

    classes = model_store.model.predict(X)
    probabilities = model_store.model.predict_proba(X)

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
             churn_count,
             n_clients)
    return results
