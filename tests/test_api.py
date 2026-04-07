"""
test_api.py — integration tests for the churn FastAPI service.

Uses TestClient (synchronous, no real HTTP) with the synthetic CSV fixture
from conftest.py. Each test class covers one logical scenario.

The `client` fixture in conftest.py:
  - resets model_store.model to None (untrained state)
  - patches DEFAULT_DATASET_PATH to the synthetic CSV
  - redirects model / metadata / history files to tmp_path
"""
import pytest


# ── /health ──────────────────────────────────────────────────────────────────

class TestHealth:
    """testing service availability"""
    def test_returns_200(self, client):
        """testing service availability"""
        assert client.get("/health").status_code == 200

    def test_body_contains_message(self, client):
        """testing service availability"""
        body = client.get("/health").json()
        assert "model_ready" in body


# ── /dataset/preview ─────────────────────────────────────────────────────────

class TestDatasetPreview:
    """testing /dataset/preview"""
    def test_returns_200(self, client):
        """testing /dataset/preview health"""
        assert client.get("/dataset/preview").status_code == 200

    def test_default_returns_ten_rows(self, client):
        """testing /dataset/preview return 10 rows"""
        rows = client.get("/dataset/preview").json()
        assert len(rows) == 10

    def test_n_param_limits_rows(self, client):
        """testing /dataset/preview return n rows"""
        rows = client.get("/dataset/preview?n=3").json()
        assert len(rows) == 3

    def test_n_zero_returns_422(self, client):
        """testing /dataset/preview return 422 on 0 row"""
        r = client.get("/dataset/preview?n=0")
        assert r.status_code == 422

    def test_rows_contain_expected_keys(self, client):
        """testing row contains expected key"""
        rows = client.get("/dataset/preview?n=1").json()
        assert "monthly_fee" in rows[0]
        assert "churn" in rows[0]


# ── /dataset/info ────────────────────────────────────────────────────────────

class TestDatasetInfo:
    """testing /dataset/info"""
    def test_returns_200(self, client):
        """testing /dataset/info health"""
        assert client.get("/dataset/info").status_code == 200

    def test_body_shape(self, client):
        """testing /dataset/preview"""
        body = client.get("/dataset/info").json()
        assert {"num_rows", "num_columns", "feature_names",
                "churn_distribution"} <= set(body.keys())


# ── /dataset/split-info ──────────────────────────────────────────────────────

class TestDatasetSplitInfo:
    """testing /dataset/split-info"""
    def test_returns_200(self, client):
        """testing /dataset/split-info health"""
        assert client.get("/dataset/split-info").status_code == 200

    def test_contains_train_and_test_sizes(self, client):
        """testing /dataset/preview"""
        body = client.get("/dataset/split-info").json()
        assert "train_size" in body
        assert "test_size" in body


# ── /model/status (untrained) ────────────────────────────────────────────────

class TestModelStatusUntrained:
    """testing /dataset/status"""
    def test_returns_200(self, client):
        """testing /dataset/status health"""
        assert client.get("/model/status").status_code == 200

    def test_trained_is_false(self, client):
        body = client.get("/model/status").json()
        assert body["trained"] is False

    def test_metrics_are_null(self, client):
        body = client.get("/model/status").json()
        assert body["metrics"] is None


# ── /model/train ─────────────────────────────────────────────────────────────

class TestModelTrain:
    def test_logreg_returns_200(self, client):
        r = client.post("/model/train",
                        json={"model_type": "logreg", "hyperparameters": {}})
        assert r.status_code == 200

    def test_response_contains_metrics(self, client):
        body = client.post(
            "/model/train",
            json={"model_type": "logreg", "hyperparameters": {}}
        ).json()
        assert "accuracy" in body
        assert "f1_score" in body
        assert "roc_auc" in body

    def test_metrics_are_floats_between_zero_and_one(self, client):
        body = client.post(
            "/model/train",
            json={"model_type": "logreg", "hyperparameters": {}}
        ).json()
        for key in ("accuracy", "f1_score", "roc_auc"):
            assert 0.0 <= body[key] <= 1.0

    def test_random_forest_returns_200(self, client):
        r = client.post(
            "/model/train",
            json={"model_type": "random_forest",
                  "hyperparameters": {"n_estimators": 10}}
        )
        assert r.status_code == 200

    def test_invalid_model_type_returns_422(self, client):
        r = client.post(
            "/model/train",
            json={"model_type": "xgboost", "hyperparameters": {}}
        )
        assert r.status_code == 422

    def test_bad_hyperparameter_returns_422(self, client):
        r = client.post(
            "/model/train",
            json={"model_type": "logreg",
                  "hyperparameters": {"nonexistent_param": 999}}
        )
        assert r.status_code == 422

    def test_error_body_matches_common_format(self, client):
        body = client.post(
            "/model/train",
            json={"model_type": "xgboost", "hyperparameters": {}}
        ).json()
        assert {"code", "message"} <= set(body.keys())


# ── /model/status (after training) ───────────────────────────────────────────

class TestModelStatusAfterTraining:
    def test_trained_becomes_true(self, client):
        client.post("/model/train",
                    json={"model_type": "logreg", "hyperparameters": {}})
        body = client.get("/model/status").json()
        assert body["trained"] is True

    def test_model_type_is_stored(self, client):
        client.post("/model/train",
                    json={"model_type": "logreg", "hyperparameters": {}})
        body = client.get("/model/status").json()
        assert body["model_type"] == "logreg"

    def test_metrics_are_present(self, client):
        client.post("/model/train",
                    json={"model_type": "logreg", "hyperparameters": {}})
        body = client.get("/model/status").json()
        assert body["metrics"] is not None
        assert "accuracy" in body["metrics"]


# ── /predict (untrained) ─────────────────────────────────────────────────────

_VALID_CLIENT = {
    "monthly_fee": 50.0,
    "usage_hours": 20.0,
    "support_requests": 2,
    "account_age_months": 12,
    "failed_payments": 1,
    "region": "north",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 0,
}


class TestPredictUntrained:
    def test_returns_409_when_no_model(self, client):
        r = client.post("/predict", json=_VALID_CLIENT)
        assert r.status_code == 409

    def test_error_body_has_model_not_ready_code(self, client):
        body = client.post("/predict", json=_VALID_CLIENT).json()
        assert body["code"] == "model_not_ready"


# ── /predict (after training) ────────────────────────────────────────────────

class TestPredictAfterTraining:
    @pytest.fixture(autouse=True)
    def train_first(self, client):
        client.post("/model/train",
                    json={"model_type": "logreg", "hyperparameters": {}})

    def test_single_client_returns_one_result(self, client):
        body = client.post("/predict", json=_VALID_CLIENT).json()
        assert len(body) == 1

    def test_batch_returns_correct_count(self, client):
        body = client.post("/predict",
                           json=[_VALID_CLIENT,
                                 _VALID_CLIENT,
                                 _VALID_CLIENT]).json()
        assert len(body) == 3

    def test_response_has_required_fields(self, client):
        result = client.post("/predict", json=_VALID_CLIENT).json()[0]
        assert {"churn_class",
                "probability_churn",
                "probability_no_churn"} <= set(result.keys())

    def test_churn_class_is_binary(self, client):
        result = client.post("/predict", json=_VALID_CLIENT).json()[0]
        assert result["churn_class"] in {0, 1}

    def test_probabilities_sum_to_one(self, client):
        result = client.post("/predict", json=_VALID_CLIENT).json()[0]
        total = result["probability_churn"] + result["probability_no_churn"]
        assert abs(total - 1.0) < 1e-4

    def test_missing_required_field_returns_422(self, client):
        bad_client = {
            k: v for k, v in _VALID_CLIENT.items() if k != "monthly_fee"}
        r = client.post("/predict", json=bad_client)
        assert r.status_code == 422

    def test_wrong_type_returns_422(self, client):
        bad_client = {**_VALID_CLIENT, "support_requests": "not-an-int"}
        r = client.post("/predict", json=bad_client)
        assert r.status_code == 422

    def test_empty_clients_list_returns_422(self, client):
        r = client.post("/predict", json=[])
        assert r.status_code == 422

    def test_error_body_format_on_validation_failure(self, client):
        bad_client = {
            k: v for k, v in _VALID_CLIENT.items() if k != "monthly_fee"}
        body = client.post("/predict", json=bad_client).json()
        assert {"code", "message", "details"} <= set(body.keys())
        assert body["code"] == "validation_error"


# ── /model/metrics ───────────────────────────────────────────────────────────

class TestModelMetrics:
    def test_returns_200_with_no_history(self, client):
        assert client.get("/model/metrics").status_code == 200

    def test_empty_history_returns_empty_list(self, client):
        body = client.get("/model/metrics").json()
        assert body["history"] == []
        assert body["latest_run"] is None

    def test_after_training_history_has_one_record(self, client):
        client.post("/model/train",
                    json={"model_type": "logreg", "hyperparameters": {}})
        body = client.get("/model/metrics").json()
        assert body["total_returned"] == 1

    def test_filter_by_model_type(self, client):
        client.post("/model/train",
                    json={"model_type": "logreg", "hyperparameters": {}})
        client.post("/model/train",
                    json={"model_type": "random_forest",
                          "hyperparameters": {"n_estimators": 10}})
        body = client.get("/model/metrics?model_type=logreg").json()
        assert all(r["model_type"] == "logreg" for r in body["history"])

    def test_last_n_limits_results(self, client):
        for _ in range(3):
            client.post("/model/train",
                        json={"model_type": "logreg", "hyperparameters": {}})
        body = client.get("/model/metrics?last_n=2").json()
        assert body["total_returned"] == 2

    def test_latest_run_matches_first_history_entry(self, client):
        client.post("/model/train",
                    json={"model_type": "logreg", "hyperparameters": {}})
        body = client.get("/model/metrics").json()
        assert body["latest_run"] == body["history"][0]


# ── /model/schema ────────────────────────────────────────────────────────────

class TestModelSchema:
    def test_returns_200(self, client):
        assert client.get("/model/schema").status_code == 200

    def test_body_has_features_list(self, client):
        body = client.get("/model/schema").json()
        assert "features" in body
        assert isinstance(body["features"], list)

    def test_all_features_have_required_keys(self, client):
        features = client.get("/model/schema").json()["features"]
        for f in features:
            assert {"name", "type", "group"} <= set(f.keys())

    def test_groups_are_numeric_or_categorical(self, client):
        features = client.get("/model/schema").json()["features"]
        for f in features:
            assert f["group"] in {"numeric", "categorical"}

    def test_churn_not_in_features(self, client):
        features = client.get("/model/schema").json()["features"]
        names = [f["name"] for f in features]
        assert "churn" not in names


# ── Full end-to-end scenario ─────────────────────────────────────────────────

class TestEndToEndScenario:
    """
    Mirrors the real usage flow:
        train → status → predict → metrics
    """
    def test_full_flow(self, client):
        # 1. Train
        train_r = client.post(
            "/model/train",
            json={"model_type": "logreg", "hyperparameters": {}}
        )
        assert train_r.status_code == 200
        assert train_r.json()["accuracy"] > 0

        # 2. Status shows trained
        status = client.get("/model/status").json()
        assert status["trained"] is True
        assert status["model_type"] == "logreg"

        # 3. Predict returns a result
        predict_r = client.post("/predict", json=_VALID_CLIENT)
        assert predict_r.status_code == 200
        result = predict_r.json()[0]
        assert result["churn_class"] in {0, 1}

        # 4. Metrics has one record
        metrics = client.get("/model/metrics").json()
        assert metrics["total_returned"] == 1
        assert metrics["latest_run"]["model_type"] == "logreg"
