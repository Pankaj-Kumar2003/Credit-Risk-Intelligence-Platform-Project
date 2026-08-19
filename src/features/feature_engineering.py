"""
Business feature engineering for credit risk.

Every feature here maps to a real underwriting concept an analyst
would recognize — debt-to-income, utilization, account age, inquiry
frequency — rather than generic polynomial/statistical transforms.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.helper import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.prep_config = self.config["preprocessing"]

    def add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "application_date" in df.columns:
            app_date = pd.to_datetime(df["application_date"], errors="coerce")
            df["application_year"] = app_date.dt.year
            df["application_month"] = app_date.dt.month
            df["application_quarter"] = app_date.dt.quarter

        if "account_open_date" in df.columns and "application_date" in df.columns:
            open_date = pd.to_datetime(df["account_open_date"], errors="coerce")
            account_age_days = (app_date - open_date).dt.days
            df["account_age_years"] = (account_age_days / 365.25).clip(lower=0).round(2)
        return df

    def add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Debt-to-income, loan-to-income, and credit-utilization — the three
        ratios underwriters check first on every real application."""
        df = df.copy()
        income = df["annual_income"].replace(0, np.nan)
        credit_limit = df["credit_limit"].replace(0, np.nan)

        df["debt_to_income_ratio"] = (df["existing_debt"] / income).fillna(0).round(4)
        df["loan_to_income_ratio"] = (df["loan_amount"] / income).fillna(0).round(4)
        df["credit_utilization_ratio"] = (df["existing_debt"] / credit_limit).clip(upper=3).fillna(0).round(4)
        df["monthly_income_est"] = (df["annual_income"] / 12).round(2)
        return df

    def add_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Employment and credit-history stability signals."""
        df = df.copy()
        df["is_long_tenured"] = (df["employment_years"] >= 5).astype(int)
        df["is_new_employee"] = (df["employment_years"] < 1).astype(int)
        if "account_age_years" in df.columns:
            df["is_thin_file"] = (df["account_age_years"] < 2).astype(int)
        return df

    def add_risk_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregated prior-default and delinquency signal into a single
        composite score, which XGBoost/LightGBM often pick up faster than
        the raw component counts alone."""
        df = df.copy()
        df["has_previous_default"] = (df["num_previous_defaults"] > 0).astype(int)
        df["adverse_event_count"] = (
            df["num_previous_defaults"] + df["delinquencies_2y"] + (df["num_credit_inquiries_6m"] > 3).astype(int)
        )
        df["inquiry_intensity"] = df["num_credit_inquiries_6m"] / (df["num_open_accounts"].replace(0, np.nan))
        df["inquiry_intensity"] = df["inquiry_intensity"].fillna(df["num_credit_inquiries_6m"]).round(4)
        return df

    def add_income_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        bins = self.prep_config["income_bins"]
        labels = self.prep_config["income_bin_labels"]
        df["income_bracket"] = pd.cut(df["annual_income"], bins=bins, labels=labels, include_lowest=True)
        df["income_bracket"] = df["income_bracket"].astype(str)
        return df

    def add_composite_risk_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """A simple, transparent heuristic risk score (0-100) used as an
        additional model input and for the dashboard's quick-glance KPI —
        NOT a replacement for the trained model's probability output."""
        df = df.copy()
        score = (
            (850 - df["credit_score"]) / 550 * 35
            + df["debt_to_income_ratio"].clip(upper=2) / 2 * 25
            + df["credit_utilization_ratio"].clip(upper=1.5) / 1.5 * 20
            + (df["adverse_event_count"].clip(upper=5) / 5) * 20
        )
        df["heuristic_risk_score"] = score.clip(0, 100).round(2)
        return df

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Running feature engineering pipeline...")
        df = self.add_date_features(df)
        df = self.add_ratio_features(df)
        df = self.add_stability_features(df)
        df = self.add_risk_history_features(df)
        df = self.add_income_bins(df)
        df = self.add_composite_risk_score(df)
        logger.info(f"Feature engineering complete. Shape={df.shape}")
        return df
