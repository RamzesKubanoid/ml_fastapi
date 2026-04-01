import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    _build_preprocessor,
    load_raw_splits,
)


# ── Custom transformer ───────────────────────────────────────────────────────

class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that wraps _build_preprocessor from
    utils/preprocessing.py, keeping all original preprocessing logic intact.

    Inheriting BaseEstimator gives get_params()/set_params() for free —
    required for GridSearchCV and Pipeline cloning.
    Inheriting TransformerMixin gives fit_transform() for free —
    called internally by Pipeline.fit().

    Attributes:
        scaler_:  Fitted StandardScaler (set during fit).
        encoder_: Fitted OneHotEncoder  (set during fit).
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnPreprocessor":
        """
        Delegates to _build_preprocessor to fit the StandardScaler on numeric
        features and OneHotEncoder on categorical features.

        Args:
            X: Raw training feature matrix containing all columns defined in
               NUMERIC_FEATURES and CATEGORICAL_FEATURES.
            y: Ignored. Present only for sklearn API compatibility.

        Returns:
            self
        """
        self.scaler_, self.encoder_ = _build_preprocessor(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the fitted scaler and encoder to X, replicating the transform
        logic from _apply_preprocessing, and returns a named DataFrame.

        Args:
            X: Raw feature matrix (train or test split).

        Returns:
            DataFrame with scaled numeric columns followed by OHE categorical
            columns, identical in structure to _apply_preprocessing output.
        """
        ohe_columns = self.encoder_.get_feature_names_out(
            CATEGORICAL_FEATURES
        ).tolist()

        scaled = pd.DataFrame(
            self.scaler_.transform(X[NUMERIC_FEATURES]),
            columns=NUMERIC_FEATURES,
            index=X.index,
        )
        encoded = pd.DataFrame(
            self.encoder_.transform(X[CATEGORICAL_FEATURES]),
            columns=ohe_columns,
            index=X.index,
        )
        return pd.concat([scaled, encoded], axis=1)


# ── Pipeline factory ─────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Constructs an unfitted sklearn Pipeline with two named steps:
      1. "preprocessor" — ChurnPreprocessor (wraps existing preprocessing).
      2. "classifier"   — LogisticRegression.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", ChurnPreprocessor()),
            ("classifier", LogisticRegression(
                max_iter=1000,  # avoids ConvergenceWarning on real data
                random_state=42,  # reproducible results
                class_weight="balanced",  # handles churn class imbalance
            )),
        ]
    )


# ── Training ─────────────────────────────────────────────────────────────────

def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Builds and fits the churn prediction pipeline on the training data.

    pipeline.fit() internally calls ChurnPreprocessor.fit_transform(X_train)
    first, then passes the result to LogisticRegression.fit() — all in one
    call. Scaler and encoder are fitted on training data only, preventing any
    data leakage from the test set.

    Args:
        X_train: Raw (unscaled, un-encoded) training feature matrix, as
                 returned by load_raw_splits().
        y_train: Binary churn labels (0 = retained, 1 = churned).

    Returns:
        Fitted sklearn Pipeline ready for prediction on raw feature input.
    """
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_raw_splits()

    model = train_churn_model(X_train, y_train)

    print(f"Train accuracy : {model.score(X_train, y_train):.4f}")
    print(f"Test  accuracy : {model.score(X_test,  y_test):.4f}")
