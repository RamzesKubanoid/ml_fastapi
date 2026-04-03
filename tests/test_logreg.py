"""
test_logreg.py — unit tests for src/utils/logreg.py and
src/utils/model_factory.py + transformer_universal.py

Tests cover: ChurnPreprocessor sklearn API, pipeline construction,
model training, prediction output shapes and types.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.utils.logreg import ChurnPreprocessor, build_pipeline, \
    train_churn_model
from src.utils.model_factory import build_churn_pipeline, \
    resolve_hyperparameters
from src.utils.preprocessing import NUMERIC_FEATURES


# ── ChurnPreprocessor ────────────────────────────────────────────────────────

class TestChurnPreprocessor:
    def test_fit_sets_scaler_and_encoder(self, raw_splits):
        X_train, *_ = raw_splits
        cp = ChurnPreprocessor()
        cp.fit(X_train)
        assert hasattr(cp, "scaler_")
        assert hasattr(cp, "encoder_")

    def test_fit_returns_self(self, raw_splits):
        X_train, *_ = raw_splits
        cp = ChurnPreprocessor()
        result = cp.fit(X_train)
        assert result is cp

    def test_transform_returns_dataframe(self, raw_splits):
        X_train, X_test, *_ = raw_splits
        cp = ChurnPreprocessor().fit(X_train)
        out = cp.transform(X_test)
        assert isinstance(out, pd.DataFrame)

    def test_transform_has_no_nulls(self, raw_splits):
        X_train, X_test, *_ = raw_splits
        cp = ChurnPreprocessor().fit(X_train)
        out = cp.transform(X_test)
        assert out.isna().sum().sum() == 0

    def test_numeric_columns_present_after_transform(self, raw_splits):
        X_train, X_test, *_ = raw_splits
        cp = ChurnPreprocessor().fit(X_train)
        out = cp.transform(X_test)
        for col in NUMERIC_FEATURES:
            assert col in out.columns

    def test_fit_transform_matches_fit_then_transform(self, raw_splits):
        X_train, *_ = raw_splits
        cp1 = ChurnPreprocessor()
        out1 = cp1.fit_transform(X_train)   # TransformerMixin
        cp2 = ChurnPreprocessor()
        out2 = cp2.fit(X_train).transform(X_train)
        pd.testing.assert_frame_equal(out1, out2)


# ── build_pipeline (logreg.py) ───────────────────────────────────────────────

class TestBuildPipeline:
    def test_returns_pipeline(self):
        p = build_pipeline()
        assert isinstance(p, Pipeline)

    def test_has_preprocessor_and_classifier_steps(self):
        p = build_pipeline()
        step_names = [name for name, _ in p.steps]
        assert "preprocessor" in step_names
        assert "classifier" in step_names


# ── train_churn_model ────────────────────────────────────────────────────────

class TestTrainChurnModel:
    def test_returns_fitted_pipeline(self, raw_splits):
        X_train, _, y_train, _ = raw_splits
        model = train_churn_model(X_train, y_train)
        assert isinstance(model, Pipeline)

    def test_predict_output_shape(self, raw_splits):
        X_train, X_test, y_train, _ = raw_splits
        model = train_churn_model(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_predict_only_binary_classes(self, raw_splits):
        X_train, X_test, y_train, _ = raw_splits
        model = train_churn_model(X_train, y_train)
        preds = model.predict(X_test)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, raw_splits):
        X_train, X_test, y_train, _ = raw_splits
        model = train_churn_model(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)

    def test_predict_proba_rows_sum_to_one(self, raw_splits):
        X_train, X_test, y_train, _ = raw_splits
        model = train_churn_model(X_train, y_train)
        proba = model.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ── resolve_hyperparameters ──────────────────────────────────────────────────

class TestResolveHyperparameters:
    def test_empty_overrides_returns_defaults(self):
        params = resolve_hyperparameters("logreg", {})
        assert "C" in params
        assert "max_iter" in params

    def test_override_wins_over_default(self):
        params = resolve_hyperparameters("logreg", {"C": 0.01})
        assert params["C"] == 0.01

    def test_non_overridden_defaults_preserved(self):
        params = resolve_hyperparameters("logreg", {"C": 0.01})
        assert "max_iter" in params   # default still present

    def test_random_forest_defaults(self):
        params = resolve_hyperparameters("random_forest", {})
        assert "n_estimators" in params


# ── build_churn_pipeline (model_factory / transformer_universal) ─────────────

class TestBuildChurnPipeline:
    def test_logreg_pipeline_fits_and_predicts(self, raw_splits):
        X_train, X_test, y_train, _ = raw_splits
        params = resolve_hyperparameters("logreg", {})
        p = build_churn_pipeline("logreg", params)
        p.fit(X_train, y_train)
        assert p.predict(X_test).shape == (len(X_test),)

    def test_random_forest_pipeline_fits_and_predicts(self, raw_splits):
        X_train, X_test, y_train, _ = raw_splits
        params = resolve_hyperparameters("random_forest", {"n_estimators": 10})
        p = build_churn_pipeline("random_forest", params)
        p.fit(X_train, y_train)
        assert p.predict(X_test).shape == (len(X_test),)

    def test_unsupported_model_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            build_churn_pipeline("xgboost", {})

    def test_bad_hyperparameter_raises(self, raw_splits):
        # LogisticRegression(**params) is called inside build_churn_pipeline
        # (in _build_classifier), so TypeError is raised at construction time,
        # not at .fit() time.
        X_train, _, y_train, _ = raw_splits
        params = resolve_hyperparameters("logreg", {"unknown_param": 999})
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            build_churn_pipeline("logreg", params)
