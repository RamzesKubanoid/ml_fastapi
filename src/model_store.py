"""
model_store.py — single source of truth for the in-memory model state.

Python's module system executes this file exactly once per process, no matter
how many times it is imported. That makes module-level variables a safe and
idiomatic alternative to app.state when a lifespan handler is not used.

On first import the module attempts to load a previously saved model and its
metadata from disk. If no file exists yet it starts with empty state, which
the /model/status endpoint surfaces as "not trained".
"""
from sklearn.pipeline import Pipeline

from src.utils.model_manipulation import (
    load_churn_model,
    load_model_metadata,
)
from src.utils.log_control import get_logger

log = get_logger(__name__)

# ── State ────────────────────────────────────────────────────────────────────

model:    Pipeline | None = None
metadata: dict | None = None

# ── Load from disk on import ─────────────────────────────────────────────────

try:
    model = load_churn_model()
    metadata = load_model_metadata()
    log.info("Pre-trained model loaded from disk.")
except FileNotFoundError:
    log.warning("No saved model found — service starts untrained.")


# ── Mutators ─────────────────────────────────────────────────────────────────

def update(new_model: Pipeline, new_metadata: dict) -> None:
    """
    Updates the in-memory state after a successful training run.

    Called by the /model/train endpoint so that /model/status reflects the
    new model instantly without requiring a restart.

    Args:
        new_model:    Freshly fitted sklearn Pipeline.
        new_metadata: Metadata dict produced by save_model_metadata().
    """
    global model, metadata
    model = new_model
    metadata = new_metadata
