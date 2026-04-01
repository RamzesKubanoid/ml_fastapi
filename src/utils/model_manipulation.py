import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


# ── Paths ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = Path("models/churn_model.joblib")
DEFAULT_METADATA_PATH = Path("models/churn_model_metadata.json")


# ── Model ────────────────────────────────────────────────────────────────────

def save_churn_model(
    pipeline: Pipeline,
    path: Path = DEFAULT_MODEL_PATH,
) -> None:
    """
    Serialises a fitted sklearn Pipeline to disk using joblib.

    joblib is preferred over pickle for sklearn objects because it handles
    large numpy arrays more efficiently via memory-mapped files.

    Args:
        pipeline: Fitted sklearn Pipeline to persist.
        path:     Destination file path (default: models/churn_model.joblib).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[model] Saved to {path}")


def load_churn_model(
    path: Path = DEFAULT_MODEL_PATH,
) -> Pipeline:
    """
    Deserialises a fitted sklearn Pipeline from disk.

    Args:
        path: Path to the .joblib file (default: models/churn_model.joblib).

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        FileNotFoundError: If no model file exists at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at: {path}")

    pipeline = joblib.load(path)
    print(f"[model] Loaded from {path}")
    return pipeline


# ── Metadata ─────────────────────────────────────────────────────────────────

def save_model_metadata(
    accuracy: float,
    f1: float,
    path: Path = DEFAULT_METADATA_PATH,
) -> dict:
    """
    Saves model training metadata (timestamp + metrics) as a JSON file
    alongside the model artifact.

    Args:
        accuracy: Accuracy on the test set.
        f1:       F1 score on the test set.
        path:     Destination JSON path 
            (default: models/churn_model_metadata.json).

    Returns:
        The metadata dict that was written to disk.
    """
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "accuracy": accuracy,
            "f1_score": f1,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[model] Metadata saved to {path}")
    return metadata


def load_model_metadata(
    path: Path = DEFAULT_METADATA_PATH,
) -> dict | None:
    """
    Loads model metadata from disk if it exists.

    Args:
        path: Path to the metadata JSON file.

    Returns:
        Metadata dict, or None if the file does not exist.
    """
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)
