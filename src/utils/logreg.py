"""
logreg.py — custom sklearn-compatible preprocessing transformer.

ChurnPreprocessor wraps the fitting and transformation logic from
preprocessing.py into a standard sklearn Transformer so it can be
composed inside any sklearn Pipeline.

The production training pipeline (model_factory.py + transformer_universal.py)
uses ColumnTransformer instead, which saves and loads the entire preprocessing
and model chain as a single atomic joblib object. ChurnPreprocessor is kept
here for testing, experimentation, and as a reference implementation.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _build_preprocessor,
)


class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer wrapping _build_preprocessor.

    Inheriting BaseEstimator provides get_params() / set_params() — required
    for GridSearchCV and Pipeline cloning.
    Inheriting TransformerMixin provides fit_transform() — called internally
    by Pipeline.fit().

    Attributes:
        scaler_:  Fitted StandardScaler (set during fit).
        encoder_: Fitted OneHotEncoder  (set during fit).
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnPreprocessor":
        """
        Fits StandardScaler on numeric features and OneHotEncoder on
        categorical features using only the provided data.

        Args:
            X: Feature matrix with all columns in NUMERIC_FEATURES and
               CATEGORICAL_FEATURES.
            y: Ignored — present for sklearn API compatibility.

        Returns:
            self
        """
        self.scaler_, self.encoder_ = _build_preprocessor(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the fitted scaler and encoder to X.

        Args:
            X: Raw feature matrix (train or test split).

        Returns:
            DataFrame with scaled numeric columns followed by OHE categorical
            columns. Column names are preserved for numeric features and
            auto-generated (e.g. region_north) for OHE columns.
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
