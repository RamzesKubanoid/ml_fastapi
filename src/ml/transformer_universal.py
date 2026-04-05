"""
ColumnTransformer-based preprocessing pipeline.

An alternative to ChurnPreprocessor (logreg.py) that uses sklearn's native
ColumnTransformer to combine StandardScaler and OneHotEncoder in a single
sklearn object.

"""
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


# ── Preprocessor ─────────────────────────────────────────────────────────────

def build_column_transformer() -> ColumnTransformer:
    """
    Builds a ColumnTransformer with two parallel branches:

      - "numeric"      → StandardScaler  applied to NUMERIC_FEATURES
      - "categorical"  → OneHotEncoder   applied to CATEGORICAL_FEATURES

    remainder="drop" ensures no unexpected columns (e.g. stray CSV columns)
    can pass through silently.

    handle_unknown="ignore" on the encoder means that unseen category values
    at prediction time produce an all-zeros OHE row instead of crashing —
    the safest default for production.

    Returns:
        Unfitted ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # keeps orig column names, no prefix
    )


# ── Pipeline factories ───────────────────────────────────────────────────────

def build_universal_pipeline(
    model_type: str,
    hyperparameters: dict[str, Any],
) -> Pipeline:
    """
    Builds a full sklearn Pipeline:
        ColumnTransformer (scale + encode) → classifier.

    The whole pipeline — transformer and model — is a single sklearn object
    that joblib saves and loads atomically. This means:
      - No risk of loading a model paired with a mismatched scaler.
      - predict() on raw DataFrames works out of the box: the transformer
        runs first, then the classifier.

    Args:
        model_type:      "logreg" or "random_forest".
        hyperparameters: Fully-resolved hyperparameter dict
                         (use resolve_hyperparameters()
                         to merge defaults first).

    Returns:
        Unfitted sklearn Pipeline with steps:
            "preprocessor" → ColumnTransformer
            "classifier"   → LogisticRegression | RandomForestClassifier

    Raises:
        ValueError: If model_type is not a supported string.
    """
    classifier = _build_classifier(model_type, hyperparameters)
    return Pipeline(
        steps=[
            ("preprocessor", build_column_transformer()),
            ("classifier",   classifier),
        ]
    )


# ── Internal classifier factory ──────────────────────────────────────────────

def _build_classifier(
    model_type: str,
    hyperparameters: dict[str, Any],
):
    """
    Instantiates an unfitted classifier from a type string.

    Args:
        model_type:      "logreg" or "random_forest".
        hyperparameters: Complete hyperparameter dict.

    Returns:
        Unfitted sklearn classifier.

    Raises:
        ValueError: If model_type is not recognised.
    """
    if model_type == "logreg":
        return LogisticRegression(**hyperparameters)

    if model_type == "random_forest":
        return RandomForestClassifier(**hyperparameters)

    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        f"Supported: ['logreg', 'random_forest']"
    )
