"""
history_recorder.py — append-only training history stored as a JSON file.

Each call to /model/train appends one record to models/training_history.json.
The file is a JSON array so it can be inspected with any text editor or
parsed by any tool without a database driver.

Design decisions:
  - Append-only: records are never mutated or deleted, preserving a full audit
    trail of every training run.
  - File-per-service: one flat JSON array is sufficient for a single-instance
    service.
  - All I/O goes through a file lock to prevent corruption if two training
    requests run concurrently.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import roc_auc_score


# ── Paths & constants ────────────────────────────────────────────────────────

DEFAULT_HISTORY_PATH = Path("models/training_history.json")

# _lock serialises concurrent writes within the same process.
_lock = threading.Lock()


# ── Record builder ───────────────────────────────────────────────────────────

def build_training_record(
    model_type: str,
    hyperparameters: dict[str, Any],
    accuracy: float,
    f1: float,
    roc_auc: float,
) -> dict[str, Any]:
    """
    Constructs a single training history record.

    Args:
        model_type:      Identifier string, e.g. "logreg" or "random_forest".
        hyperparameters: Full resolved hyperparameter dict used during training
        accuracy:        Accuracy score on the test set.
        f1:              F1 score on the test set (positive class).
        roc_auc:         ROC-AUC score on the test set.

    Returns:
        Dict with keys: trained_at, model_type, hyperparameters, metrics.
    """
    return {
        "trained_at":      datetime.now(timezone.utc).isoformat(),
        "model_type":      model_type,
        "hyperparameters": hyperparameters,
        "metrics": {
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc":  roc_auc,
        },
    }


def compute_roc_auc(model, X_test, y_test) -> float:
    """
    Computes ROC-AUC using predicted probabilities for the positive class.

    Args:
        model:  Fitted sklearn Pipeline with predict_proba support.
        X_test: Raw test feature matrix.
        y_test: True binary labels.

    Returns:
        ROC-AUC score rounded to 4 decimal places.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    return round(float(roc_auc_score(y_test, y_proba)), 4)


# ── Persistence ──────────────────────────────────────────────────────────────

def append_training_record(
    record: dict[str, Any],
    path: Path = DEFAULT_HISTORY_PATH,
) -> None:
    """
    Appends a training record to the history JSON file.

    The file contains a JSON array. If it does not exist yet it is created.
    A threading lock prevents concurrent writes from corrupting the file.

    Args:
        record: Dict produced by build_training_record().
        path:   Path to the history JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with _lock:
        history = _load_raw(path)
        history.append(record)
        with open(path, "w") as f:
            json.dump(history, f, indent=2)


def load_history(
    path: Path = DEFAULT_HISTORY_PATH,
    model_type: str | None = None,
    last_n: int | None = None,
) -> list[dict[str, Any]]:
    """
    Loads training history from the JSON file with optional filtering.

    Args:
        path:       Path to the history JSON file.
        model_type: If provided, returns only records for that model type.
        last_n:     If provided, returns only the N most recent records.

    Returns:
        List of history records, newest-first.
    """
    history = _load_raw(path)

    if model_type:
        history = [r for r in history if r.get("model_type") == model_type]

    # Reverse so newest comes first — more useful for API consumers
    history = list(reversed(history))

    if last_n is not None:
        history = history[:last_n]

    return history


def _load_raw(path: Path) -> list[dict[str, Any]]:
    """Reads the JSON array from disk, or returns [] if the file is absent."""
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)
