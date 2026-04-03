"""
test_model_manipulation.py — unit tests for src/utils/model_manipulation.py

Tests cover: save/load round-trip, missing file errors, metadata persistence.
"""
import json
import numpy as np
import pytest

from src.utils.model_manipulation import (
    load_churn_model,
    load_model_metadata,
    save_churn_model,
    save_model_metadata,
)


# ── save_churn_model / load_churn_model ──────────────────────────────────────

class TestModelPersistence:
    def test_save_creates_file(self, tmp_path, trained_pipeline):
        path = tmp_path / "model.joblib"
        save_churn_model(trained_pipeline, path)
        assert path.exists()

    def test_load_returns_pipeline(self, tmp_path, trained_pipeline):
        path = tmp_path / "model.joblib"
        save_churn_model(trained_pipeline, path)
        loaded = load_churn_model(path)
        from sklearn.pipeline import Pipeline
        assert isinstance(loaded, Pipeline)

    def test_loaded_pipeline_predicts_same(self,
                                           tmp_path,
                                           trained_pipeline,
                                           raw_splits):
        _, X_test, *_ = raw_splits
        path = tmp_path / "model.joblib"
        save_churn_model(trained_pipeline, path)
        loaded = load_churn_model(path)

        np.testing.assert_array_equal(
            trained_pipeline.predict(X_test),
            loaded.predict(X_test),
        )

    def test_save_creates_parent_directory(self, tmp_path, trained_pipeline):
        nested = tmp_path / "a" / "b" / "model.joblib"
        save_churn_model(trained_pipeline, nested)
        assert nested.exists()

    def test_load_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No saved model"):
            load_churn_model(tmp_path / "ghost.joblib")


# ── save_model_metadata / load_model_metadata ────────────────────────────────

class TestMetadataPersistence:
    def test_save_creates_json_file(self, tmp_path):
        path = tmp_path / "meta.json"
        save_model_metadata(0.80, 0.55, "logreg", {"C": 1.0}, path)
        assert path.exists()

    def test_saved_metadata_is_valid_json(self, tmp_path):
        path = tmp_path / "meta.json"
        save_model_metadata(0.80, 0.55, "logreg", {"C": 1.0}, path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_metadata_contains_required_keys(self, tmp_path):
        path = tmp_path / "meta.json"
        save_model_metadata(0.80, 0.55, "logreg", {"C": 1.0}, path)
        with open(path) as f:
            data = json.load(f)
        assert {"trained_at",
                "model_type",
                "hyperparameters",
                "metrics"} <= set(data.keys())

    def test_metrics_values_correct(self, tmp_path):
        path = tmp_path / "meta.json"
        save_model_metadata(0.80, 0.55, "logreg", {"C": 1.0}, path)
        meta = load_model_metadata(path)
        assert meta["metrics"]["accuracy"] == 0.80
        assert meta["metrics"]["f1_score"] == 0.55

    def test_model_type_stored(self, tmp_path):
        path = tmp_path / "meta.json"
        save_model_metadata(
            0.80, 0.55, "random_forest", {"n_estimators": 100}, path)
        meta = load_model_metadata(path)
        assert meta["model_type"] == "random_forest"

    def test_load_returns_none_if_missing(self, tmp_path):
        result = load_model_metadata(tmp_path / "no_meta.json")
        assert result is None

    def test_save_returns_metadata_dict(self, tmp_path):
        path = tmp_path / "meta.json"
        result = save_model_metadata(0.75, 0.50, "logreg", {}, path)
        assert isinstance(result, dict)
        assert result["metrics"]["accuracy"] == 0.75
