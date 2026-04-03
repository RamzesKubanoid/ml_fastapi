"""
test_preprocessing.py — unit tests for src/utils/preprocessing.py

Tests cover: missing value handling, feature/target split, train/test split,
preprocessing application, class distribution reporting.
"""
# import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    _apply_preprocessing,
    _build_preprocessor,
    _handle_missing,
    _split_features_target,
    _split_train_test,
    get_split_info,
    load_raw_splits,
    prepare_data,
)
# from tests.conftest import SYNTHETIC_ROWS


# ── _handle_missing ──────────────────────────────────────────────────────────

class TestHandleMissing:
    """
    Class for testing _handle_missing function
    """
    def test_clean_data_unchanged(self, synthetic_df):
        """Testing correct dataframe"""
        result = _handle_missing(synthetic_df.copy())
        assert len(result) == len(synthetic_df)

    def test_drops_rows_with_missing_target(self, synthetic_df):
        """Testing one null row in target"""
        df = synthetic_df.copy()
        df.loc[0, TARGET] = None
        result = _handle_missing(df)
        assert len(result) == len(synthetic_df) - 1

    def test_raises_if_all_target_missing(self, synthetic_df):
        """Testing all null rows in target"""
        df = synthetic_df.copy()
        df[TARGET] = None
        with pytest.raises(ValueError, match="empty"):
            _handle_missing(df)

    def test_imputes_numeric_with_median(self, synthetic_df):
        """Testing non target numeric null replaced with median"""
        df = synthetic_df.copy()
        df.loc[0, "monthly_fee"] = None
        result = _handle_missing(df)
        assert result["monthly_fee"].isna().sum() == 0

    def test_imputes_categorical_with_mode(self, synthetic_df):
        """Testing non target categorical null replaced with mode"""
        df = synthetic_df.copy()
        df.loc[0, "region"] = None
        result = _handle_missing(df)
        assert result["region"].isna().sum() == 0


# ── _split_features_target ───────────────────────────────────────────────────

class TestSplitFeaturesTarget:
    """Class for testing _split_features_target function"""
    def test_x_contains_only_defined_features(self, synthetic_df):
        """Testing X contain all defined feature columns"""
        X, _ = _split_features_target(synthetic_df)
        assert set(X.columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    def test_target_not_in_x(self, synthetic_df):
        """Testing X does not contain target column"""
        X, _ = _split_features_target(synthetic_df)
        assert TARGET not in X.columns

    def test_y_is_integer_series(self, synthetic_df):
        """Testing y column is integer series"""
        _, y = _split_features_target(synthetic_df)
        assert y.dtype == int

    def test_lengths_match(self, synthetic_df):
        """Testing all splits have the same number of rows"""
        X, y = _split_features_target(synthetic_df)
        assert len(X) == len(y) == len(synthetic_df)


# ── _split_train_test ────────────────────────────────────────────────────────

class TestSplitTrainTest:
    """Class for testing _split_train_test function"""
    def test_default_split_ratio(self, synthetic_df):
        """default split testing"""
        X, y = _split_features_target(synthetic_df)
        X_train, X_test, _, _ = _split_train_test(X, y)
        total = len(synthetic_df)
        assert len(X_test) == pytest.approx(total * 0.2, abs=1)
        assert len(X_train) + len(X_test) == total

    def test_stratification_preserves_class_ratio(self, synthetic_df):
        """test stratification"""
        X, y = _split_features_target(synthetic_df)
        _, _, y_train, y_test = _split_train_test(X, y)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.15

    def test_reproducible_with_same_seed(self, synthetic_df):
        """split results reproduction test"""
        X, y = _split_features_target(synthetic_df)
        split1 = _split_train_test(X, y, random_state=42)
        split2 = _split_train_test(X, y, random_state=42)
        pd.testing.assert_frame_equal(split1[0], split2[0])

    def test_different_seeds_give_different_splits(self, synthetic_df):
        """split results reproduction with different seeds test"""
        X, y = _split_features_target(synthetic_df)
        X_train_1, *_ = _split_train_test(X, y, random_state=0)
        X_train_2, *_ = _split_train_test(X, y, random_state=99)
        assert not X_train_1.index.equals(X_train_2.index)


# ── _build_preprocessor / _apply_preprocessing ───────────────────────────────

class TestPreprocessing:
    """Preprocessing functions tests"""
    def test_build_preprocessor_returns_scaler_and_encoder(self, synthetic_df):
        """test _build_preprocessor scaler and encoder"""
        X, _ = _split_features_target(synthetic_df)
        scaler, encoder = _build_preprocessor(X)
        assert isinstance(scaler, StandardScaler)
        assert isinstance(encoder, OneHotEncoder)

    def test_apply_preprocessing_no_nulls(self, synthetic_df):
        """test _apply_preprocessing no nulls"""
        X, _ = _split_features_target(synthetic_df)
        X_train, X_test, *_ = _split_train_test(X, _)
        X_train_p, X_test_p = _apply_preprocessing(X_train, X_test)
        assert X_train_p.isna().sum().sum() == 0
        assert X_test_p.isna().sum().sum() == 0

    def test_numeric_columns_are_scaled(self, synthetic_df):
        """test of numeric columns scaling"""
        X, _ = _split_features_target(synthetic_df)
        X_train, X_test, *_ = _split_train_test(X, _)
        X_train_p, _ = _apply_preprocessing(X_train, X_test)
        for col in NUMERIC_FEATURES:
            assert abs(X_train_p[col].mean()) < 0.5   # roughly zero-centred

    def test_categorical_columns_are_encoded(self, synthetic_df):
        """test of categorical columns encoding"""
        X, _ = _split_features_target(synthetic_df)
        X_train, X_test, *_ = _split_train_test(X, _)
        X_train_p, _ = _apply_preprocessing(X_train, X_test)
        # OHE columns are 0/1
        ohe_cols = [c for c in X_train_p.columns if c not in NUMERIC_FEATURES]
        assert set(X_train_p[ohe_cols].values.flatten()).issubset({0.0, 1.0})

    def test_test_set_uses_train_statistics(self, synthetic_df):
        """
        Scaler fitted on train must be applied to test without re-fitting.
        """
        X, _ = _split_features_target(synthetic_df)
        X_train, X_test, *_ = _split_train_test(X, _)
        # Should not raise
        _apply_preprocessing(X_train, X_test)


# ── load_raw_splits / prepare_data ───────────────────────────────────────────

class TestPipelineFunctions:
    """test of pipeline functions"""
    def test_load_raw_splits_returns_four_objects(self, synthetic_csv):
        """test of the load_raw_splits output"""
        result = load_raw_splits(synthetic_csv)
        assert len(result) == 4

    def test_raw_splits_columns_are_unencoded(self, synthetic_csv):
        """test of the load_raw_splits output"""
        X_train, *_ = load_raw_splits(synthetic_csv)
        assert "region" in X_train.columns      # still a string column

    def test_prepare_data_columns_are_encoded(self, synthetic_csv):
        """test of the categorical columns encoded"""
        X_train, *_ = prepare_data(synthetic_csv)
        assert "region" not in X_train.columns  # replaced by OHE columns


# ── get_split_info ───────────────────────────────────────────────────────────

class TestGetSplitInfo:
    """test of get_split_info function"""
    def test_keys(self, raw_splits):
        """test get_split_info output"""
        _, _, y_train, y_test = raw_splits
        info = get_split_info(y_train, y_test)
        assert {
            "train_size", "test_size", "churn_distribution"} == set(info.keys()
                                                                    )

    def test_sizes_correct(self, raw_splits):
        """test get_split_info size corretness"""
        _, _, y_train, y_test = raw_splits
        info = get_split_info(y_train, y_test)
        assert info["train_size"] == len(y_train)
        assert info["test_size"] == len(y_test)

    def test_shares_sum_to_one(self, raw_splits):
        """test get_split_info size corretness"""
        _, _, y_train, y_test = raw_splits
        info = get_split_info(y_train, y_test)
        for split in ("train", "test"):
            total = sum(
                v["share"] for v in info["churn_distribution"][split].values()
                )
            assert abs(total - 1.0) < 1e-4
