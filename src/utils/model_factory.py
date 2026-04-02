"""
model_factory.py — builds sklearn classifiers and full pipelines by name.

Keeps model construction separate from training logic, making it easy to
add new model types in one place without touching any other file.
"""
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils.logreg import ChurnPreprocessor


# ── Per-model default hyperparameters ────────────────────────────────────────
# Defined here so they are stored in metadata even when the caller passes an
# empty hyperparameters dict, making /model/status always informative.

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


def build_classifier(
    model_type: str,
    hyperparameters: dict[str, Any],
):
    """
    Instantiates an unfitted sklearn classifier from a type string and a
    fully-resolved hyperparameter dict.

    Args:
        model_type:      "logreg" or "random_forest".
        hyperparameters: Complete hyperparameter dict
                         (use resolve_hyperparameters
                         to merge defaults with caller overrides first).

    Returns:
        Unfitted sklearn classifier instance.

    Raises:
        ValueError: If model_type is not a supported type string.
    """
    if model_type == "logreg":
        return LogisticRegression(**hyperparameters)

    if model_type == "random_forest":
        return RandomForestClassifier(**hyperparameters)

    supported = list(_DEFAULTS.keys())
    raise ValueError(
        f"Unsupported model_type '{model_type}'. Supported: {supported}"
    )


def build_churn_pipeline(
    model_type: str,
    hyperparameters: dict[str, Any],
) -> Pipeline:
    """
    Composes a full sklearn Pipeline:
        ChurnPreprocessor (from logreg.py) → classifier.

    Reuses the existing ChurnPreprocessor so all preprocessing logic stays in
    one place and the pipeline is compatible with the rest of the codebase.

    Args:
        model_type:      "logreg" or "random_forest".
        hyperparameters: Fully-resolved hyperparameters (defaults + overrides).

    Returns:
        Unfitted sklearn Pipeline.
    """
    classifier = build_classifier(model_type, hyperparameters)
    return Pipeline(
        steps=[
            ("preprocessor", ChurnPreprocessor()),
            ("classifier", classifier),
        ]
    )
