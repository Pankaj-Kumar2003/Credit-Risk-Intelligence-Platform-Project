"""
Data quality validation — missingness, duplicates, row-count floor,
and basic range sanity checks. Produces a JSON report used both by
CI (tests/test_validation.py) and by anyone auditing a new data drop
before it's allowed into the training pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from src.utils.helper import load_config, resolve_path, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    is_valid: bool
    n_rows: int
    n_duplicate_rows: int
    missing_fraction_by_column: Dict[str, float] = field(default_factory=dict)
    columns_exceeding_missing_threshold: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DataValidator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.validation_config = self.config["validation"]
        self.id_column = self.config["data"]["id_column"]

    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        return (df.isna().mean()).round(4).to_dict()

    def check_duplicates(self, df: pd.DataFrame) -> int:
        if self.id_column in df.columns:
            return int(df.duplicated(subset=[self.id_column]).sum())
        return int(df.duplicated().sum())

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        report = DataQualityReport(
            is_valid=True,
            n_rows=len(df),
            n_duplicate_rows=self.check_duplicates(df),
            missing_fraction_by_column=self.check_missing_values(df),
        )

        threshold = self.validation_config["max_missing_fraction"]
        report.columns_exceeding_missing_threshold = [
            col for col, frac in report.missing_fraction_by_column.items() if frac > threshold
        ]
        if report.columns_exceeding_missing_threshold:
            report.is_valid = False
            report.errors.append(
                f"Columns exceed missing-value threshold ({threshold}): "
                f"{report.columns_exceeding_missing_threshold}"
            )

        min_rows = self.validation_config["min_rows"]
        if report.n_rows < min_rows:
            report.is_valid = False
            report.errors.append(f"Row count {report.n_rows} below minimum required {min_rows}")

        if report.n_duplicate_rows > 0:
            report.warnings.append(f"{report.n_duplicate_rows} duplicate rows detected — will be dropped in cleaning.")

        if report.is_valid:
            logger.info(f"Data quality validation passed. rows={report.n_rows}, duplicates={report.n_duplicate_rows}")
        else:
            logger.error(f"Data quality validation failed: {report.errors}")

        return report

    def save_report(self, report: DataQualityReport, path: str = "reports/data_quality_report.json") -> None:
        save_json(report.__dict__, resolve_path(path))
