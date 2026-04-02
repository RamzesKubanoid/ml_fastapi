"""ML churn service models"""
from typing import Any, Literal

from pydantic import BaseModel, Field


class FeatureVectorChurn(BaseModel):
    """
    Base churn dataset model
    """
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "monthly_fee": 65.0,
                    "usage_hours": 12.5,
                    "support_requests": 3,
                    "account_age_months": 24,
                    "failed_payments": 1,
                    "region": "africa",
                    "device_type": "mobile",
                    "payment_method": "card",
                    "autopay_enabled": 1,
                }
            ]
        }
    }


class DataSetRowChurn(FeatureVectorChurn):
    """
    Base churn dataset model with prediction
    """
    churn: int


# ── Prediction request ───────────────────────────────────────────────────────

class PredictionRequestChurn(BaseModel):
    """
    Wraps one or more FeatureVectorChurn objects for a prediction request.
    Pass a list with a single item to predict for one client.
    """
    clients: list[FeatureVectorChurn] = Field(
        min_length=1,
        description="One or more client feature vectors to predict churn for.",
        examples=[
            [
                {
                    "monthly_fee": 65.0,
                    "usage_hours": 12.5,
                    "support_requests": 3,
                    "account_age_months": 24,
                    "failed_payments": 1,
                    "region": "North",
                    "device_type": "Mobile",
                    "payment_method": "Credit Card",
                    "autopay_enabled": 1,
                }
            ]
        ],
    )


# ── Prediction response ──────────────────────────────────────────────────────

class PredictionResponseChurn(BaseModel):
    """
    Structured prediction result for a single client.
    """
    churn_class: int = Field(
        description="Predicted churn label: 1 = will churn, 0 = will stay.",
        examples=[1],
    )
    probability_churn: float = Field(
        description="Model confidence that the client will churn (class 1).",
        examples=[0.83],
    )
    probability_no_churn: float = Field(
        description="Model confidence that the client will stay (class 0).",
        examples=[0.17],
    )


# ── Training config ──────────────────────────────────────────────────────────

class TrainingConfigChurn(BaseModel):
    """
    Controls which model is trained and with what hyperparameters.

    Supported model types and their key hyperparameters:

    **logreg** (LogisticRegression):
    - `C` (float, default 1.0) — inverse regularisation strength;
        smaller = stronger regularisation
    - `max_iter` (int, default 1000) — maximum solver iterations
    - `class_weight` (str, default "balanced") — handles class imbalance

    **random_forest** (RandomForestClassifier):
    - `n_estimators` (int, default 100) — number of trees
    - `max_depth` (int | null, default null) — maximum tree depth;
        null = unlimited
    - `class_weight` (str, default "balanced") — handles class imbalance
    - `random_state` (int, default 42) — reproducibility seed
    """
    model_type: Literal["logreg", "random_forest"] = Field(
        default="logreg",
        description="Which classifier to train: 'logreg' or 'random_forest'.",
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Hyperparameter overrides for the selected model. "
            "Any key not provided falls back to the model's default value. "
            "Unknown keys are rejected by the classifier at fit time."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_type": "logreg",
                    "hyperparameters": {"C": 0.5, "max_iter": 500},
                },
                {
                    "model_type": "random_forest",
                    "hyperparameters": {"n_estimators": 200, "max_depth": 10},
                },
            ]
        }
    }
