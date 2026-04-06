"""Dataset validation and processing functions"""
import pandas as pd
from src.schemas import DataSetRowChurn
from src.core.log_control import get_logger

log = get_logger(__name__)

# ── Column definitions ───────────────────────────────────────────────────────

NUMERIC_FEATURES: list[str] = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
]

CATEGORICAL_FEATURES: list[str] = [
    "region",
    "device_type",
    "payment_method",
    "autopay_enabled",
]

TARGET: str = "churn"


def validate_df_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reads churn dataframe rows, validates and transforms each row
    into a DataSetRowChurn object, and returns
    all records as a pandas DataFrame.
    Args:
        raw_df: pandas Dataframe containing churn data
    Returns:
        A pandas DataFrame where each row corresponds to a validated
        DataSetRowChurn instance.
    """
    validated_rows: list[DataSetRowChurn] = [
        DataSetRowChurn(**row)
        for row in raw_df.to_dict(orient="records")
    ]

    log.info("Dataset validated: %d rows", len(validated_rows))
    return pd.DataFrame(
        [row.model_dump() for row in validated_rows]
    )


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows where the target is missing, then handles missing values in
    features:
      - Numeric columns  → filled with the column median.
      - Categorical cols → filled with the column mode (most frequent value).

    Args:
        df: Raw DataFrame straight from load_churn_dataset.

    Returns:
        Cleaned DataFrame with no missing values.

    Raises:
        ValueError: If any expected column is absent from the DataFrame.
    """
    expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required column(s): {missing_cols}. "
            f"Expected: {expected}. "
            f"Found: {list(df.columns)}."
        )

    rows_before = len(df)

    # Target missing → cannot impute, must drop
    df = df.dropna(subset=[TARGET])

    if df.empty:
        raise ValueError(
            "Dataset is empty after dropping rows with missing target.")

    dropped = rows_before - len(df)
    if dropped:
        print(f"[missing] Dropped {dropped} row(s) with missing target.")

    # Numeric features: impute with median
    missing_numeric = [c for c in NUMERIC_FEATURES if df[c].isna().any()]
    if missing_numeric:
        df[missing_numeric] = df[missing_numeric].fillna(
            df[missing_numeric].median())
        print(
            f"[missing] Imputed median for numeric columns: {missing_numeric}")

    # Categorical features: impute with mode
    missing_categorical = [
        c for c in CATEGORICAL_FEATURES if df[c].isna().any()
        ]
    for col in missing_categorical:
        df[col] = df[col].fillna(df[col].mode()[0])
    if missing_categorical:
        print(
            f"[missing] Imputed mode for categoric cols: {missing_categorical}"
            )

    return df
