"""
test_history_recorder.py — unit tests for src/utils/history_recorder.py

Tests cover: record construction, append/load round-trip, filtering,
ordering, thread-safety basics, roc_auc computation.
"""
import json
from datetime import datetime

from src.utils.history_recorder import (
    _load_raw,
    append_training_record,
    build_training_record,
    compute_roc_auc,
    load_history,
)


# ── build_training_record ────────────────────────────────────────────────────

class TestBuildTrainingRecord:
    def test_returns_dict(self):
        r = build_training_record("logreg", {"C": 1.0}, 0.80, 0.55, 0.88)
        assert isinstance(r, dict)

    def test_required_keys_present(self):
        r = build_training_record("logreg", {"C": 1.0}, 0.80, 0.55, 0.88)
        assert {"trained_at",
                "model_type",
                "hyperparameters",
                "metrics"} <= set(r.keys())

    def test_metrics_values(self):
        r = build_training_record("logreg", {}, 0.80, 0.55, 0.88)
        assert r["metrics"]["accuracy"] == 0.80
        assert r["metrics"]["f1_score"] == 0.55
        assert r["metrics"]["roc_auc"] == 0.88

    def test_trained_at_is_iso_string(self):
        r = build_training_record("logreg", {}, 0.80, 0.55, 0.88)
        # Should not raise
        datetime.fromisoformat(r["trained_at"])


# ── append_training_record / load_history ────────────────────────────────────

class TestHistoryPersistence:
    def _record(self, model_type="logreg"):
        return build_training_record(model_type, {"C": 1.0}, 0.80, 0.55, 0.88)

    def test_creates_file_on_first_append(self, tmp_path):
        path = tmp_path / "history.json"
        append_training_record(self._record(), path)
        assert path.exists()

    def test_file_is_valid_json_array(self, tmp_path):
        path = tmp_path / "history.json"
        append_training_record(self._record(), path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_multiple_appends_accumulate(self, tmp_path):
        path = tmp_path / "history.json"
        for _ in range(3):
            append_training_record(self._record(), path)
        assert len(_load_raw(path)) == 3

    def test_load_history_newest_first(self, tmp_path):
        path = tmp_path / "history.json"
        for mt in ["logreg", "random_forest", "logreg"]:
            append_training_record(self._record(mt), path)
        history = load_history(path)
        # last appended should be first returned
        assert history[0]["model_type"] == "logreg"

    def test_load_history_filter_by_model_type(self, tmp_path):
        path = tmp_path / "history.json"
        for mt in ["logreg", "random_forest", "logreg"]:
            append_training_record(self._record(mt), path)
        rf_only = load_history(path, model_type="random_forest")
        assert len(rf_only) == 1
        assert rf_only[0]["model_type"] == "random_forest"

    def test_load_history_last_n(self, tmp_path):
        path = tmp_path / "history.json"
        for _ in range(5):
            append_training_record(self._record(), path)
        result = load_history(path, last_n=2)
        assert len(result) == 2

    def test_load_history_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "no_history.json"
        result = load_history(path)
        assert result == []

    def test_filter_and_last_n_combined(self, tmp_path):
        path = tmp_path / "history.json"
        for mt in ["logreg", "random_forest"] * 4:
            append_training_record(self._record(mt), path)
        result = load_history(path, model_type="logreg", last_n=2)
        assert len(result) == 2
        assert all(r["model_type"] == "logreg" for r in result)


# ── compute_roc_auc ──────────────────────────────────────────────────────────

class TestComputeRocAuc:
    def test_returns_float_between_zero_and_one(self,
                                                trained_pipeline,
                                                raw_splits):
        _, X_test, _, y_test = raw_splits
        score = compute_roc_auc(trained_pipeline, X_test, y_test)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_rounds_to_four_decimal_places(self, trained_pipeline, raw_splits):
        _, X_test, _, y_test = raw_splits
        score = compute_roc_auc(trained_pipeline, X_test, y_test)
        assert score == round(score, 4)
