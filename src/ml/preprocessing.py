from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.ml.dataset import load_churn_dataset, \
    DEFAULT_DATASET_PATH
from src.ml.row_handler import validate_df_rows, _handle_missing, \
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET


# ── Data preparation ─────────────────────────────────────────────────────────

def prepare_data(
    csv_path: Path = DEFAULT_DATASET_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Full preprocessing pipeline:
        splitted dataset
        → scale numeric → encode categorical → report

    Args:
        csv_path: Path to the churn CSV file.

    Returns:
        X_train, X_test, y_train, y_test — ready for model training.
    """

    X_train, X_test, y_train, y_test = load_raw_splits(csv_path)
    X_train, X_test = _apply_preprocessing(X_train, X_test)

    print_class_distribution(y_train, y_test)

    return X_train, X_test, y_train, y_test


def load_raw_splits(
    csv_path: Path = DEFAULT_DATASET_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    load → clean → split, without scaling or encoding.

    Returns raw feature splits so that an external sklearn Pipeline can own
    the transformation step, avoiding duplicate preprocessing.

    Args:
        csv_path: Path to the churn CSV file.

    Returns:
        X_train, X_test, y_train, y_test — unscaled and un-encoded.
    """
    df = load_churn_dataset(csv_path)
    df = _handle_missing(df)
    df = validate_df_rows(df)
    X, y = _split_features_target(df)
    return _split_train_test(X, y)


# ── Steps ────────────────────────────────────────────────────────────────────

def _split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates the feature matrix X from the target vector y.

    X contains only the explicitly defined NUMERIC_FEATURES and
    CATEGORICAL_FEATURES columns — no stray columns can leak in.

    Args:
        df: Cleaned DataFrame.

    Returns:
        X: DataFrame with shape (n_samples, n_features).
        y: Series of integer churn labels.
    """
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[all_features].copy()
    y = df[TARGET].astype(int)
    return X, y


def _split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits X and y into training and test sets.

    Uses stratify=y so that the class ratio of churn is preserved in both
    splits, which is especially important for imbalanced datasets.

    Args:
        X:            Feature matrix.
        y:            Target vector.
        test_size:    Fraction of data reserved for testing (default 0.2).
        random_state: Seed for reproducibility (default 42).

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def _build_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[StandardScaler, OneHotEncoder]:
    """
    Fits a StandardScaler on numeric features and a OneHotEncoder on
    categorical features using only the training data.

    Fitting on training data exclusively prevents data leakage — test-set
    statistics must never influence the transformers.

    Args:
        X_train: Training feature matrix (pre-split).

    Returns:
        scaler:  StandardScaler fitted on NUMERIC_FEATURES of X_train.
        encoder: OneHotEncoder fitted on CATEGORICAL_FEATURES of X_train.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[NUMERIC_FEATURES])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train[CATEGORICAL_FEATURES])

    return scaler, encoder


def _apply_preprocessing(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits transformers on X_train and applies them to both splits, then
    recombines scaled numeric and encoded categorical columns into a single
    DataFrame preserving interpretable column names.

    Args:
        X_train: Raw training feature matrix.
        X_test:  Raw test feature matrix.

    Returns:
        X_train_processed, X_test_processed — both as DataFrames with named
        columns: original numeric names + OHE-generated category names.
    """
    scaler, encoder = _build_preprocessor(X_train)

    ohe_columns: list[str] = encoder.get_feature_names_out(
        CATEGORICAL_FEATURES).tolist()

    def _transform(X: pd.DataFrame) -> pd.DataFrame:
        scaled = pd.DataFrame(
            scaler.transform(X[NUMERIC_FEATURES]),
            columns=NUMERIC_FEATURES,
            index=X.index,
        )
        encoded = pd.DataFrame(
            encoder.transform(X[CATEGORICAL_FEATURES]),
            columns=ohe_columns,
            index=X.index,
        )
        return pd.concat([scaled, encoded], axis=1)

    return _transform(X_train), _transform(X_test)


# ── Reporting ────────────────────────────────────────────────────────────────

def _class_distribution(y: pd.Series) -> dict:
    """Returns count and share per churn class for a single split."""
    counts = y.value_counts().sort_index()
    total = len(y)
    return {
        int(cls): {
            "count": int(count),
            "share": round(count / total, 4),
        }
        for cls, count in counts.items()
    }


def get_split_info(y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Returns train/test sizes and per-class churn distribution as a dict,
    ready to be serialised to JSON.

    Args:
        y_train: Target labels for the training set.
        y_test:  Target labels for the test set.

    Returns:
        {
            "train_size": int,
            "test_size":  int,
            "churn_distribution": {
                "train": {0: {"count": int, "share": float}, 1: {...}},
                "test":  {0: {"count": int, "share": float}, 1: {...}},
            }
        }
    """
    return {
        "train_size": len(y_train),
        "test_size": len(y_test),
        "churn_distribution": {
            "train": _class_distribution(y_train),
            "test": _class_distribution(y_test),
        },
    }


def print_class_distribution(y_train: pd.Series, y_test: pd.Series) -> None:
    """
    Prints a side-by-side class distribution table for train and test splits.

    Args:
        y_train: Target labels for the training set.
        y_test:  Target labels for the test set.
    """
    info = get_split_info(y_train, y_test)
    dist = info["churn_distribution"]

    print("\n── Class distribution ───────────────────────────────")
    print(f"{'churn':<8} "
          f"{'train_count':>12} {'train_share %':>14} "
          f"{'test_count':>11} {'test_share %':>13}")
    for cls in sorted(dist["train"]):
        tr = dist["train"][cls]
        te = dist["test"][cls]
        print(f"{cls:<8} "
              f"{tr['count']:>12} {tr['share'] * 100:>13.2f}% "
              f"{te['count']:>11} {te['share'] * 100:>12.2f}%")
    print(f"\n  Train size : {info['train_size']:>6}")
    print(f"  Test size  : {info['test_size']:>6}")
    print("─────────────────────────────────────────────────────\n")
