"""
Batch and single-record prediction service. Wraps the fitted cleaner,
feature engineer, and column transformer so a raw applicant record
(exactly the shape of a row in data/raw/) can be scored end-to-end
without re-fitting anything.
"""

from typing import Any, Dict, List

import pandas as pd

from src.features.feature_engineering import FeatureEngineer
from src.utils.logger import get_logger
from src.utils.model_loader import LoadedArtifacts, load_artifacts

logger = get_logger(__name__)

RISK_BANDS = [
    (0.0, 0.10, "Low Risk"),
    (0.10, 0.25, "Moderate Risk"),
    (0.25, 0.50, "High Risk"),
    (0.50, 1.01, "Very High Risk"),
]


def categorize_risk(probability: float) -> str:
    for lower, upper, label in RISK_BANDS:
        if lower <= probability < upper:
            return label
    return "Very High Risk"


class CreditRiskPredictor:
    def __init__(self, artifacts: LoadedArtifacts = None):
        self.artifacts = artifacts or load_artifacts()
        self.feature_engineer = FeatureEngineer(self.artifacts.config)
        self.data_config = self.artifacts.config["data"]

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.artifacts.cleaner.transform(df)
        df = self.feature_engineer.engineer(df)

        numeric = [c for c in self.artifacts.feature_names if c in df.columns]
        # The column transformer expects the exact numeric + categorical
        # column ordering it was fit on; feature_names reflects the
        # *post*-encoding names, so we rebuild the pre-encoding frame here.
        from src.preprocessing.preprocessing_pipeline import ENGINEERED_NUMERIC, ENGINEERED_CATEGORICAL

        numeric_cols = [c for c in self.data_config["numerical_features"] + ENGINEERED_NUMERIC if c in df.columns]
        categorical_cols = [c for c in self.data_config["categorical_features"] + ENGINEERED_CATEGORICAL if c in df.columns]

        encoded = self.artifacts.column_transformer.transform(df[numeric_cols + categorical_cols])
        return pd.DataFrame(encoded, columns=self.artifacts.feature_names, index=df.index)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = self.data_config["id_column"]
        ids = df[id_col] if id_col in df.columns else pd.Series(range(len(df)), name=id_col)

        features = self._prepare_features(df)
        probabilities = self.artifacts.model.predict_proba(features)[:, 1]

        result = pd.DataFrame({
            id_col: ids.values,
            "default_probability": probabilities.round(4),
            "risk_category": [categorize_risk(p) for p in probabilities],
        })
        return result

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([record])
        result = self.predict_batch(df)
        return result.iloc[0].to_dict()


if __name__ == "__main__":
    from src.utils.helper import resolve_path

    predictor = CreditRiskPredictor()
    sample = pd.read_csv(resolve_path(predictor.artifacts.config["data"]["raw_path"])).head(5)
    print(predictor.predict_batch(sample))
