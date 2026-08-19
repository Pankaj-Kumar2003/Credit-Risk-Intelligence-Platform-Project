import numpy as np
import pandas as pd

from src.validation.data_validation import DataValidator
from src.validation.schema_validation import SchemaValidator


def test_schema_validation_passes_on_clean_data(config, sample_raw_df):
    result = SchemaValidator(config).validate(sample_raw_df)
    assert result.is_valid
    assert result.missing_columns == []


def test_schema_validation_fails_on_bad_target_values(config, sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "default_flag"] = 5
    result = SchemaValidator(config).validate(df)
    assert not result.is_valid
    assert 5 in result.invalid_target_values


def test_data_validation_flags_excess_missingness(config, sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[: int(len(df) * 0.6), "credit_score"] = np.nan
    report = DataValidator(config).validate(df)
    assert not report.is_valid
    assert "credit_score" in report.columns_exceeding_missing_threshold


def test_data_validation_detects_duplicates(config, sample_raw_df):
    df = pd.concat([sample_raw_df, sample_raw_df.head(5)], ignore_index=True)
    report = DataValidator(config).validate(df)
    assert report.n_duplicate_rows == 5
