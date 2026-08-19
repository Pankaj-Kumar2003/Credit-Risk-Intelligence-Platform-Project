"""
Schema validation — checks structural correctness of incoming data
before it ever reaches cleaning or feature engineering: required
columns present, expected dtypes, target column has only allowed
values. This is intentionally separate from data_validation.py,
which checks data *quality* (missingness, duplicates, outliers).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from src.utils.helper import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_DTYPES = {
    "applicant_id": "object",
    "annual_income": "number",
    "loan_amount": "number",
    "credit_score": "number",
    "employment_years": "number",
    "age": "number",
    "existing_debt": "number",
    "credit_limit": "number",
    "num_open_accounts": "number",
    "num_credit_inquiries_6m": "number",
    "num_previous_defaults": "number",
    "delinquencies_2y": "number",
    "loan_purpose": "object",
    "home_ownership": "object",
    "default_flag": "number",
}


@dataclass
class SchemaValidationResult:
    is_valid: bool
    missing_columns: List[str] = field(default_factory=list)
    dtype_mismatches: List[str] = field(default_factory=list)
    invalid_target_values: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class SchemaValidator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.data_config = self.config["data"]
        self.validation_config = self.config["validation"]

    def validate(self, df: pd.DataFrame) -> SchemaValidationResult:
        result = SchemaValidationResult(is_valid=True)

        required_columns = list(self.data_config["column_mapping"].keys())
        result.missing_columns = [c for c in required_columns if c not in df.columns]
        if result.missing_columns:
            result.is_valid = False
            result.errors.append(f"Missing required columns: {result.missing_columns}")

        for col, expected_kind in EXPECTED_DTYPES.items():
            if col not in df.columns:
                continue
            if expected_kind == "number" and not pd.api.types.is_numeric_dtype(df[col]):
                result.dtype_mismatches.append(col)
            elif expected_kind == "object" and not (
                pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
            ):
                result.dtype_mismatches.append(col)
        if result.dtype_mismatches:
            result.is_valid = False
            result.errors.append(f"Dtype mismatches: {result.dtype_mismatches}")

        target_col = self.data_config["target_column"]
        allowed_values = set(self.validation_config["allowed_default_flag_values"])
        if target_col in df.columns:
            observed = set(df[target_col].dropna().unique().tolist())
            invalid = observed - allowed_values
            if invalid:
                result.invalid_target_values = sorted(invalid)
                result.is_valid = False
                result.errors.append(f"Target column has disallowed values: {invalid}")

        if result.is_valid:
            logger.info("Schema validation passed.")
        else:
            logger.error(f"Schema validation failed: {result.errors}")

        return result
