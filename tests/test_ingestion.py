import pandas as pd

from src.ingestion.data_ingestion import DataIngestion


def test_apply_column_mapping_renames_known_columns(config, sample_raw_df):
    ingestion = DataIngestion(config)
    mapped = ingestion.apply_column_mapping(sample_raw_df)
    for canonical_col in config["data"]["column_mapping"].keys():
        assert canonical_col in mapped.columns


def test_apply_column_mapping_preserves_row_count(config, sample_raw_df):
    ingestion = DataIngestion(config)
    mapped = ingestion.apply_column_mapping(sample_raw_df)
    assert len(mapped) == len(sample_raw_df)
