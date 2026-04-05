"""Prediction related endpoint"""
import pandas as pd
from fastapi import APIRouter, Body, HTTPException

from src.schemas import PredictionRequestChurn, PredictionResponseChurn
from src.error_handlers import ErrorResponseChurn
from src.core.log_control import get_logger
from src.core import model_store

log = get_logger(__name__)
router = APIRouter()

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
