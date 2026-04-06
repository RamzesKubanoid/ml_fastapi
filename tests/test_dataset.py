"""
test_dataset.py — unit tests for src/ml/dataset.py

Tests cover: happy path loading, empty file, missing file, schema validation.
"""
import pytest
import pandas as pd
from src.ml.dataset import load_churn_dataset, dataset_info
from src.ml.row_handler import validate_df_rows
from tests.conftest import SYNTHETIC_ROWS


# ── load_churn_dataset ───────────────────────────────────────────────────────

class TestLoadChurnDataset:
    def test_returns_dataframe(self, synthetic_csv):
        df = load_churn_dataset(synthetic_csv)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, synthetic_csv):
        df = load_churn_dataset(synthetic_csv)
        assert len(df) == len(SYNTHETIC_ROWS)

    def test_expected_columns(self, synthetic_csv):
        df = load_churn_dataset(synthetic_csv)
        expected = {
            "monthly_fee", "usage_hours", "support_requests",
            "account_age_months", "failed_payments", "region",
            "device_type", "payment_method", "autopay_enabled", "churn",
        }
        assert expected == set(df.columns)

    def test_correct_dtypes(self, synthetic_csv):
        df = load_churn_dataset(synthetic_csv)
        assert df["monthly_fee"].dtype == float
        assert df["support_requests"].dtype == int
        # pandas may use object or StringDtype depending on version
        assert pd.api.types.is_string_dtype(df["region"])

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_churn_dataset(tmp_path / "nonexistent.csv")

    def test_raises_on_empty_file(self, tmp_path):
        # pd.read_csv raises EmptyDataError before our ValueError check
        # is reached when the file has no content at all; a file with only
        # a header row (but no data rows) triggers our ValueError("empty").
        empty = tmp_path / "empty.csv"
        empty.write_text("")
        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            load_churn_dataset(empty)

    def test_raises_value_error_on_header_only_file(self, tmp_path):
        """A CSV with a header but no data rows triggers our empty check."""
        header_only = tmp_path / "header_only.csv"
        header_only.write_text(
            "monthly_fee,usage_hours,support_requests,account_age_months,"
            "failed_payments,region,device_type,payment_method,"
            "autopay_enabled,churn\n"
        )
        with pytest.raises(ValueError, match="[Ee]mpty"):
            load_churn_dataset(header_only)

    def test_raises_on_missing_required_column(self, tmp_path):
        """A CSV missing a required column should fail Pydantic validation."""
        bad = tmp_path / "bad.csv"
        bad.write_text("monthly_fee,churn\n10.0,0\n")
        raw_df = load_churn_dataset(bad)
        with pytest.raises(Exception):
            validate_df_rows(raw_df)


# ── dataset_info ─────────────────────────────────────────────────────────────

class TestDatasetInfo:
    def test_returns_dict(self, synthetic_csv):
        info = dataset_info(str(synthetic_csv))
        assert isinstance(info, dict)

    def test_required_keys(self, synthetic_csv):
        info = dataset_info(str(synthetic_csv))
        assert {"num_rows", "num_columns", "feature_names",
                "churn_distribution"} == set(info.keys())

    def test_num_rows_correct(self, synthetic_csv):
        info = dataset_info(str(synthetic_csv))
        assert info["num_rows"] == len(SYNTHETIC_ROWS)

    def test_churn_not_in_feature_names(self, synthetic_csv):
        info = dataset_info(str(synthetic_csv))
        assert "churn" not in info["feature_names"]

    def test_churn_distribution_keys(self, synthetic_csv):
        info = dataset_info(str(synthetic_csv))
        dist = info["churn_distribution"]
        assert set(dist.keys()) == {0, 1}

    def test_churn_distribution_shares_sum_to_one(self, synthetic_csv):
        info = dataset_info(str(synthetic_csv))
        total = sum(v["share"] for v in info["churn_distribution"].values())
        assert abs(total - 1.0) < 1e-4
