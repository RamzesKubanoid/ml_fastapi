"""Functions to work with churn dataset"""
from pathlib import Path
import pandas as pd
from src.schemas import DataSetRowChurn
from src.core.log_control import get_logger

log = get_logger(__name__)


DEFAULT_DATASET_PATH = Path(
    __file__).resolve().parents[2] / "data" / "churn_dataset.csv"


def load_churn_dataset(csv_path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """
    Reads a CSV file, validates and transforms each row into a DataSetRowChurn
    object, and returns all records as a pandas DataFrame.

    Args:
        csv_path: Path to the CSV file. Defaults to data/churn_dataset.csv
                  relative to the project root.

    Returns:
        A pandas DataFrame where each row corresponds to a validated
        DataSetRowChurn instance.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
        ValidationError: If a row doesnt conform to the DataSetRowChurn schema.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    raw_df = pd.read_csv(csv_path)

    if raw_df.empty:
        raise ValueError("Dataset file is empty.")

    log.info("Dataset loaded: %d rows from %s", len(raw_df), csv_path)

    validated_rows: list[DataSetRowChurn] = [
        DataSetRowChurn(**row)
        for row in raw_df.to_dict(orient="records")
    ]

    return pd.DataFrame(
        [row.model_dump() for row in validated_rows]
    )


def dataset_info(csv_path: str = str(DEFAULT_DATASET_PATH)) -> dict:
    """
    Loads the churn dataset and returns summary information about it.

    Args:
        csv_path: Path to the CSV file as a string. Defaults to
                  data/churn_dataset.csv relative to the project root.

    Returns:
        A dict containing:
            - num_rows:             total number of rows
            - num_columns:          total number of columns
            - feature_names:        list of all column names except 'churn'
            - churn_distribution:   dict mapping each churn class (0/1) to
                                    its absolute count and relative share
    """
    df = load_churn_dataset(Path(csv_path))

    churn_counts = df["churn"].value_counts().to_dict()
    total = len(df)

    churn_distribution = {
        int(cls): {
            "count": int(count),
            "share": round(count / total, 4),
        }
        for cls, count in churn_counts.items()
    }

    return {
        "num_rows": total,
        "num_columns": len(df.columns),
        "feature_names": [col for col in df.columns if col != "churn"],
        "churn_distribution": churn_distribution,
    }
