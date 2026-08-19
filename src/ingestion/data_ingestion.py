"""
Data ingestion layer.

Responsible for reading the raw loan-applicant file from disk,
applying the configurable column mapping so the rest of the pipeline
never has to know the original dataset's exact column names, and
handing back a clean, renamed DataFrame plus a lightweight ingestion
report (row count, column count, checksum) used for data versioning.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.helper import load_config, resolve_path, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionReport:
    source_path: str
    n_rows: int
    n_columns: int
    columns: list
    file_checksum: str
    generated_at: str = field(default_factory=lambda: pd.Timestamp.utcnow().isoformat())


class DataIngestion:
    """Reads raw data and applies the dataset-agnostic column mapping."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.data_config = self.config["data"]

    @staticmethod
    def _checksum(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def load_raw(self, path: str = None) -> pd.DataFrame:
        source = resolve_path(path or self.data_config["raw_path"])
        if not source.exists():
            raise FileNotFoundError(
                f"Raw data not found at {source}. "
                f"Run `python scripts/generate_mock_data.py` or drop a real "
                f"dataset there and update config/config.yaml."
            )
        logger.info(f"Reading raw data from {source}")
        df = pd.read_csv(source)
        logger.info(f"Loaded raw data with shape {df.shape}")
        return df

    def apply_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename source-dataset columns to the platform's canonical names."""
        mapping = self.data_config["column_mapping"]
        inverse_mapping = {source_col: canonical for canonical, source_col in mapping.items()}
        present = {c: inverse_mapping[c] for c in df.columns if c in inverse_mapping}
        missing = set(mapping.keys()) - set(present.values())
        if missing:
            logger.warning(f"Columns missing from source data after mapping: {sorted(missing)}")
        return df.rename(columns=present)

    def build_report(self, df: pd.DataFrame, source_path: Path) -> IngestionReport:
        return IngestionReport(
            source_path=str(source_path),
            n_rows=len(df),
            n_columns=df.shape[1],
            columns=list(df.columns),
            file_checksum=self._checksum(source_path),
        )

    def run(self, path: str = None) -> Dict[str, Any]:
        source = resolve_path(path or self.data_config["raw_path"])
        df_raw = self.load_raw(path)
        df_mapped = self.apply_column_mapping(df_raw)
        report = self.build_report(df_raw, source)

        report_path = resolve_path("reports/ingestion_report.json")
        save_json(report.__dict__, report_path)
        logger.info(f"Ingestion report saved to {report_path}")

        return {"data": df_mapped, "report": report}


if __name__ == "__main__":
    result = DataIngestion().run()
    print(result["data"].head())
    print(result["report"])
