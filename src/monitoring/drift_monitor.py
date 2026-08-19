"""
Data and prediction drift monitoring using Evidently AI. Compares a
reference dataset (training distribution) against current production
data to flag feature drift, target drift, and prediction drift before
model performance silently degrades.
"""

from typing import Any, Dict, Optional

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

from src.utils.helper import load_config, resolve_path, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftMonitor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.monitoring_config = self.config["monitoring"]
        self.target_column = self.config["data"]["target_column"]

    def _column_mapping(self, has_prediction: bool) -> ColumnMapping:
        mapping = ColumnMapping()
        mapping.target = self.target_column
        if has_prediction:
            mapping.prediction = "prediction_proba"
        return mapping

    def run_data_drift_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        save_html: str = None,
        save_json_path: str = None,
    ) -> Dict[str, Any]:
        has_prediction = "prediction_proba" in reference.columns and "prediction_proba" in current.columns
        presets = [DataDriftPreset()]
        if self.target_column in reference.columns and self.target_column in current.columns:
            presets.append(TargetDriftPreset())

        report = Report(metrics=presets)
        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=self._column_mapping(has_prediction),
        )

        html_path = resolve_path(save_html or self.monitoring_config["report_path"])
        html_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(html_path))

        result = report.as_dict()
        json_path = resolve_path(save_json_path or "reports/drift_report.json")
        save_json(result, json_path)

        logger.info(f"Drift report saved to {html_path} and {json_path}")
        return result

    def summarize_drift(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Pulls the headline numbers out of Evidently's verbose report dict
        for a compact summary the dashboard can render as KPI cards."""
        summary = {"dataset_drift_detected": None, "n_drifted_columns": None, "share_drifted_columns": None}
        for metric in result.get("metrics", []):
            if metric.get("metric") == "DatasetDriftMetric":
                res = metric.get("result", {})
                summary["dataset_drift_detected"] = res.get("dataset_drift")
                summary["n_drifted_columns"] = res.get("number_of_drifted_columns")
                summary["share_drifted_columns"] = res.get("share_of_drifted_columns")
        return summary

    def run(self, reference_path: str = None, current_path: str = None) -> Dict[str, Any]:
        reference = pd.read_csv(resolve_path(reference_path or self.monitoring_config["reference_data_path"]))
        current = pd.read_csv(resolve_path(current_path or self.monitoring_config["current_data_path"]))
        result = self.run_data_drift_report(reference, current)
        return self.summarize_drift(result)


if __name__ == "__main__":
    monitor = DriftMonitor()
    print(monitor.run())
