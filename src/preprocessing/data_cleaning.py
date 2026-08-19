"""
Data cleaning: deduplication, outlier capping, and imputation.
Imputers are fit on the data passed in — callers are responsible for
fitting only on the training split and reusing the fitted cleaner on
test/inference data (see preprocessing_pipeline.py).
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.utils.helper import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.data_config = self.config["data"]
        self.prep_config = self.config["preprocessing"]
        self.numerical_features = self.data_config["numerical_features"]
        self.categorical_features = self.data_config["categorical_features"]

        self._numeric_imputer: SimpleImputer = None
        self._categorical_imputer: SimpleImputer = None
        self._iqr_bounds: Dict[str, Any] = {}
        self._is_fitted = False

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = self.data_config["id_column"]
        before = len(df)
        df = df.drop_duplicates(subset=[id_col]) if id_col in df.columns else df.drop_duplicates()
        removed = before - len(df)
        if removed:
            logger.info(f"Dropped {removed} duplicate rows.")
        return df.reset_index(drop=True)

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        numeric_present = [c for c in self.numerical_features if c in df.columns]
        categorical_present = [c for c in self.categorical_features if c in df.columns]

        self._numeric_imputer = SimpleImputer(strategy=self.prep_config["missing_numeric_strategy"])
        self._numeric_imputer.fit(df[numeric_present])

        self._categorical_imputer = SimpleImputer(strategy=self.prep_config["missing_categorical_strategy"])
        self._categorical_imputer.fit(df[categorical_present])

        multiplier = self.prep_config["outlier_iqr_multiplier"]
        for col in numeric_present:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                # Sparse/near-constant columns (e.g. count features where most
                # applicants have 0 prior defaults) make IQR capping degenerate —
                # Q1==Q3==0 would clip every nonzero value to 0 and destroy the
                # signal. Skip capping for these; fall back to a wide percentile
                # bound instead so genuine data-entry errors still get caught.
                lower, upper = df[col].quantile(0.001), df[col].quantile(0.999)
            else:
                lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
            self._iqr_bounds[col] = (lower, upper)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("DataCleaner must be fit() before transform().")

        df = df.copy()
        numeric_present = [c for c in self.numerical_features if c in df.columns]
        categorical_present = [c for c in self.categorical_features if c in df.columns]

        df[numeric_present] = self._numeric_imputer.transform(df[numeric_present])
        df[categorical_present] = self._categorical_imputer.transform(df[categorical_present])

        for col, (lower, upper) in self._iqr_bounds.items():
            n_capped = int(((df[col] < lower) | (df[col] > upper)).sum())
            if n_capped:
                logger.info(f"Capping {n_capped} outliers in '{col}' to [{lower:.2f}, {upper:.2f}]")
            df[col] = df[col].clip(lower=lower, upper=upper)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.drop_duplicates(df)
        return self.fit(df).transform(df)
