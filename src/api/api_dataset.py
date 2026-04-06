"""Dataset related endpoints"""
import json
from fastapi import APIRouter, HTTPException, Query

from src.ml.dataset import load_churn_dataset, dataset_info
from src.ml.preprocessing import prepare_data, get_split_info
from src.core.log_control import get_logger

log = get_logger(__name__)
router = APIRouter()


@router.get("/preview")
def preview_dataset(
    n: int = Query(default=10, ge=1, description="Number of rows to return"),
):
    """
    Returns the first N rows of the churn dataset as JSON.

    - **n**: how many rows to return (default: 10, must be ≥ 1)
    """
    try:
        df = load_churn_dataset()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    # NaN replaced with None so missing float values serialise to JSON null
    # rather than raising "Out of range float values are not JSON compliant".
    return json.loads(df.head(n).to_json(orient="records"))


@router.get("/info")
def get_dataset_info():
    """
    Returns info about dataset
    """
    try:
        return dataset_info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/split-info")
def split_info():
    """
    Returns train/test split sizes and churn class distribution for both splits
    """
    try:
        _, _, y_train, y_test = prepare_data()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return get_split_info(y_train, y_test)
