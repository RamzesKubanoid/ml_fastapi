"""Health service check endpoint"""
from fastapi import APIRouter

from src.ml.dataset import load_churn_dataset
from src.ml.row_handler import _handle_missing, validate_df_rows
from src.core.log_control import get_logger
from src.core import model_store

log = get_logger(__name__)
router = APIRouter()


@router.get("/health")
def health_check():
    """
    Returns service liveness plus readiness signals:
    - **model_ready**   — whether a trained model is in memory.
    - **dataset_ready** — whether the dataset file is accessible.
    """
    dataset_ready = False
    dataset_detail = None
    try:
        df = load_churn_dataset()
        df = _handle_missing(df)
        validate_df_rows(df)
        dataset_ready = True
    except (FileNotFoundError, ValueError) as exc:
        dataset_detail = str(exc)
        log.warning("Health: dataset not accessible: %s", exc)
    except Exception as exc:
        dataset_detail = f"{type(exc).__name__}: {exc}"
        log.warning("Health: dataset validation failed: %s", exc)

    model_ready = model_store.model is not None
    model_type = None
    if model_ready and model_store.metadata:
        model_type = model_store.metadata.get("model_type")

    status = "ok" if (dataset_ready and model_ready) else "degraded"
    log.info("Health check: status=%s model=%s dataset=%s",
             status, model_ready, dataset_ready)

    return {
        "status":         status,
        "model_ready":    model_ready,
        "model_type":     model_type,
        "dataset_ready":  dataset_ready,
        "dataset_detail": dataset_detail,
    }
