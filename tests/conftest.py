"""
conftest.py — shared pytest fixtures for the churn service test suite.

All fixtures that are used across more than one test module live here.
pytest discovers this file automatically; no imports needed in test files.
"""
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
import src.core.model_store as model_store
from src.ml import dataset as ds_mod
import src.core.model_manipulation as mm_mod
import src.core.history_recorder as hr_mod
from src.main import app
import src.api.api_model as api_model_mod
import src.api.api_dataset as api_dataset_mod
import src.api.health as api_health_mod
from src.ml.preprocessing import load_raw_splits
from src.ml.model_factory import build_churn_pipeline, \
    resolve_hyperparameters

# ── Synthetic dataset ────────────────────────────────────────────────────────
# 40 rows — small enough to be fast, large enough for stratified 80/20 split
# (at least 2 samples per class in the test fold).

_N = 40
_CHURN_LABELS = [0] * 30 + [1] * 10   # 75 / 25 split — realistic imbalance

SYNTHETIC_ROWS = [
    {
        "monthly_fee":       20.0 + i * 2,
        "usage_hours":       50.0 - i * 1.1,
        "support_requests":  i % 5,
        "account_age_months": 60 - i,
        "failed_payments":   i % 3,
        "region":            ["north", "south", "east", "west"][i % 4],
        "device_type":       ["mobile", "desktop", "tablet"][i % 3],
        "payment_method":    ["card", "bank", "paypal"][i % 3],
        "autopay_enabled":   i % 2,
        "churn":             _CHURN_LABELS[i],
    }
    for i in range(_N)
]


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """Raw DataFrame that mirrors the churn CSV schema."""
    return pd.DataFrame(SYNTHETIC_ROWS)


@pytest.fixture()
def synthetic_csv(tmp_path: Path,
                  synthetic_df: pd.DataFrame) -> Path:
    """Writes synthetic data to a temporary CSV and returns its Path."""
    csv = tmp_path / "churn_dataset.csv"
    synthetic_df.to_csv(csv, index=False)
    return csv


@pytest.fixture()
def raw_splits(synthetic_csv: Path):
    """Returns (X_train, X_test, y_train, y_test) from the synthetic CSV."""
    return load_raw_splits(synthetic_csv)


@pytest.fixture()
def trained_pipeline(raw_splits):
    """Returns a pipeline fitted on synthetic training data."""
    X_train, _, y_train, _ = raw_splits
    params = resolve_hyperparameters("logreg", {})
    pipeline = build_churn_pipeline("logreg", params)
    pipeline.fit(X_train, y_train)
    return pipeline


# ── TestClient with isolated state ───────────────────────────────────────────

@pytest.fixture()
def client(tmp_path: Path,
           synthetic_csv: Path,
           monkeypatch: pytest.MonkeyPatch):
    """
    FastAPI TestClient with:
      - model_store reset to None so each test starts untrained
      - DEFAULT_DATASET_PATH patched to the synthetic CSV
      - model / history files redirected to tmp_path so tests never
        touch the real models/ directory
    """
    # ── Reset in-memory model state ───────────────────────────────────────
    monkeypatch.setattr(model_store, "model",    None)
    monkeypatch.setattr(model_store, "metadata", None)

    # ── Redirect dataset default path ─────────────────────────────────────
    monkeypatch.setattr(ds_mod, "DEFAULT_DATASET_PATH", synthetic_csv)

    # ── Model + metadata persistence ──────────────────────────────────────

    _model_path = tmp_path / "churn_model.joblib"
    _metadata_path = tmp_path / "metadata.json"
    _history_path = tmp_path / "history.json"

    # Store originals so wrappers can delegate to them
    mm_mod._real_save_model = mm_mod.save_churn_model
    mm_mod._real_load_model = mm_mod.load_churn_model
    mm_mod._real_save_metadata = mm_mod.save_model_metadata
    mm_mod._real_load_metadata = mm_mod.load_model_metadata
    hr_mod._real_append = hr_mod.append_training_record
    hr_mod._real_load = hr_mod.load_history

    def _save_model(pipeline, path=_model_path):
        mm_mod._real_save_model(pipeline, path)

    def _load_model(path=_model_path):
        return mm_mod._real_load_model(path)

    def _save_metadata(accuracy, f1, roc_auc, model_type, hyperparameters,
                       path=_metadata_path):
        return mm_mod._real_save_metadata(
            accuracy, f1, roc_auc, model_type, hyperparameters, path
        )

    def _load_metadata(path=_metadata_path):
        return mm_mod._real_load_metadata(path)

    def _append(record, path=_history_path):
        hr_mod._real_append(record, path)

    def _load_history(path=_history_path, model_type=None, last_n=None):
        return hr_mod._real_load(
            path=path, model_type=model_type, last_n=last_n)

    # save_* are imported into src.api.api_model
    monkeypatch.setattr(mm_mod, "save_churn_model", _save_model)
    monkeypatch.setattr(api_model_mod, "save_churn_model", _save_model)

    monkeypatch.setattr(mm_mod, "save_model_metadata", _save_metadata)
    monkeypatch.setattr(api_model_mod, "save_model_metadata", _save_metadata)

    # load_* are imported into model_store only
    monkeypatch.setattr(mm_mod, "load_churn_model", _load_model)
    monkeypatch.setattr(model_store, "load_churn_model", _load_model)

    monkeypatch.setattr(mm_mod, "load_model_metadata", _load_metadata)
    monkeypatch.setattr(model_store, "load_model_metadata", _load_metadata)

    # history functions are imported into src.api.api_model
    monkeypatch.setattr(hr_mod, "append_training_record", _append)
    monkeypatch.setattr(api_model_mod, "append_training_record", _append)

    monkeypatch.setattr(hr_mod, "load_history", _load_history)
    monkeypatch.setattr(api_model_mod, "load_history", _load_history)

    # load_churn_dataset is imported into dataset and health routers
    monkeypatch.setattr(api_dataset_mod,
                        "load_churn_dataset",
                        lambda: ds_mod.load_churn_dataset(synthetic_csv))
    monkeypatch.setattr(api_health_mod,
                        "load_churn_dataset",
                        lambda: ds_mod.load_churn_dataset(synthetic_csv))

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
