"""
builds churn pipelines by model type.

Delegates pipeline construction to transformer_universal.py (ColumnTransformer
approach) and owns the per-model default hyperparameter registry.
"""
from typing import Any

from src.utils.transformer_universal import build_universal_pipeline


# ── Per-model default hyperparameters ────────────────────────────────────────
# Stored here so that resolve_hyperparameters() always returns a complete dict.
# /model/status therefore always shows the full effective configuration even
# when the caller passed an empty hyperparameters dict.

_DEFAULTS: dict[str, dict[str, Any]] = {
    "logreg": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": "balanced",
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42,
        "class_weight": "balanced",
    },
}


def resolve_hyperparameters(
    model_type: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Merges caller-supplied overrides with the model's default hyperparameters.

    Defaults are applied first; overrides win for any key present in both.
    The merged dict is what gets stored in metadata and passed to the
    classifier constructor, so /model/status always shows the full picture.

    Args:
        model_type: One of the supported model type strings.
        overrides:  Caller-supplied hyperparameter overrides (may be empty).

    Returns:
        Complete hyperparameter dict ready to pass to the classifier.
    """
    return {**_DEFAULTS[model_type], **overrides}


def build_churn_pipeline(
    model_type: str,
    hyperparameters: dict[str, Any],
):
    """
    Constructs a full sklearn Pipeline using ColumnTransformer-based
    preprocessing (StandardScaler + OneHotEncoder) followed by the chosen
    classifier.

    The entire pipeline is a single sklearn object — preprocessor and model
    are saved and loaded atomically by joblib, eliminating any risk of a
    model being paired with a mismatched transformer after a restart.

    Args:
        model_type:      "logreg" or "random_forest".
        hyperparameters: Fully-resolved dict (defaults already merged in via
                         resolve_hyperparameters).

    Returns:
        Unfitted sklearn Pipeline.
    """
    return build_universal_pipeline(model_type, hyperparameters)
