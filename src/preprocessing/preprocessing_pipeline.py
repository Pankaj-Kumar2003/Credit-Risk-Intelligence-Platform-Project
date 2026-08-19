"""
End-to-end preprocessing orchestration: ingestion -> cleaning ->
feature engineering -> encoding -> train/test split -> persisted
artifacts (fitted preprocessor + processed CSVs) that training,
prediction, and the API all reuse so there is exactly one source of
truth for how raw data becomes model-ready data.
"""

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_engineering import FeatureEngineer
from src.ingestion.data_ingestion import DataIngestion
from src.preprocessing.data_cleaning import DataCleaner
from src.utils.helper import load_config, resolve_path, save_json, save_pickle
from src.utils.logger import get_logger
from src.validation.data_validation import DataValidator
from src.validation.schema_validation import SchemaValidator

logger = get_logger(__name__)

ENGINEERED_NUMERIC = [
    "debt_to_income_ratio", "loan_to_income_ratio", "credit_utilization_ratio",
    "monthly_income_est", "is_long_tenured", "is_new_employee", "is_thin_file",
    "has_previous_default", "adverse_event_count", "inquiry_intensity",
    "heuristic_risk_score", "account_age_years",
]
ENGINEERED_CATEGORICAL = ["income_bracket"]


class PreprocessingPipeline:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.data_config = self.config["data"]
        self.target_column = self.data_config["target_column"]
        self.id_column = self.data_config["id_column"]

        self.cleaner = DataCleaner(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.column_transformer: ColumnTransformer = None
        self.feature_names_out = []

    def _feature_lists(self, df: pd.DataFrame):
        numeric = [c for c in self.data_config["numerical_features"] + ENGINEERED_NUMERIC if c in df.columns]
        categorical = [c for c in self.data_config["categorical_features"] + ENGINEERED_CATEGORICAL if c in df.columns]
        return numeric, categorical

    def _build_transformer(self, numeric_cols, categorical_cols) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([("scale", StandardScaler())]), numeric_cols),
                ("categorical", Pipeline([("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols),
            ]
        )

    def load_and_validate(self) -> pd.DataFrame:
        ingestion_result = DataIngestion(self.config).run()
        df = ingestion_result["data"]

        schema_result = SchemaValidator(self.config).validate(df)
        if not schema_result.is_valid:
            raise ValueError(f"Schema validation failed: {schema_result.errors}")

        quality_validator = DataValidator(self.config)
        quality_report = quality_validator.validate(df)
        quality_validator.save_report(quality_report)
        if not quality_report.is_valid:
            raise ValueError(f"Data quality validation failed: {quality_report.errors}")

        return df

    def build_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            df = self.cleaner.fit_transform(df)
        else:
            df = self.cleaner.transform(df)
        df = self.feature_engineer.engineer(df)
        return df

    def fit_transform_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        df = self.build_features(df, fit=True)

        target = df[self.target_column].astype(int)
        numeric_cols, categorical_cols = self._feature_lists(df)
        feature_df = df[numeric_cols + categorical_cols]

        x_train, x_test, y_train, y_test = train_test_split(
            feature_df, target,
            test_size=self.config["preprocessing"]["test_size"],
            random_state=self.config["project"]["random_seed"],
            stratify=target,
        )

        self.column_transformer = self._build_transformer(numeric_cols, categorical_cols)
        x_train_encoded = self.column_transformer.fit_transform(x_train)
        x_test_encoded = self.column_transformer.transform(x_test)

        cat_feature_names = list(
            self.column_transformer.named_transformers_["categorical"]
            .named_steps["encode"].get_feature_names_out(categorical_cols)
        ) if categorical_cols else []
        self.feature_names_out = numeric_cols + cat_feature_names

        x_train_df = pd.DataFrame(x_train_encoded, columns=self.feature_names_out, index=x_train.index)
        x_test_df = pd.DataFrame(x_test_encoded, columns=self.feature_names_out, index=x_test.index)

        return x_train_df, x_test_df, y_train, y_test

    def transform_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Used at inference time: apply already-fitted cleaner + transformer."""
        if self.column_transformer is None:
            raise RuntimeError("Pipeline must be fit before transform_only().")
        df = self.build_features(df, fit=False)
        numeric_cols, categorical_cols = self._feature_lists(df)
        encoded = self.column_transformer.transform(df[numeric_cols + categorical_cols])
        return pd.DataFrame(encoded, columns=self.feature_names_out, index=df.index)

    def persist_artifacts(self, x_train, x_test, y_train, y_test) -> None:
        train_out = x_train.copy()
        train_out[self.target_column] = y_train.values
        test_out = x_test.copy()
        test_out[self.target_column] = y_test.values

        train_out.to_csv(resolve_path(self.data_config["train_path"]), index=False)
        test_out.to_csv(resolve_path(self.data_config["test_path"]), index=False)

        save_pickle(self.column_transformer, resolve_path(self.config["training"]["preprocessor_path"]))
        save_pickle(self.cleaner, resolve_path("models/cleaner.pkl"))
        save_json(self.feature_names_out, resolve_path(self.config["training"]["feature_names_path"]))
        logger.info("Preprocessing artifacts persisted to models/ and data/processed/.")

    def run(self):
        df = self.load_and_validate()
        x_train, x_test, y_train, y_test = self.fit_transform_split(df)
        self.persist_artifacts(x_train, x_test, y_train, y_test)
        return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    x_train, x_test, y_train, y_test = pipeline.run()
    print(f"Train shape: {x_train.shape} | Test shape: {x_test.shape}")
    print(f"Train default rate: {y_train.mean():.3f} | Test default rate: {y_test.mean():.3f}")
