"""
SHAP-based explainability layer: global feature importance for model
governance/documentation, and local per-applicant explanations for
risk analysts who need to justify an approve/decline decision.
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.utils.helper import load_config, resolve_path, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ShapExplainer:
    def __init__(self, model, background_data: pd.DataFrame, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.model = model
        self.explainer = shap.TreeExplainer(model) if self._is_tree_model(model) else shap.Explainer(model, background_data)
        self.reports_dir = resolve_path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_tree_model(model) -> bool:
        return model.__class__.__name__ in {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier"}

    def compute_shap_values(self, x: pd.DataFrame):
        shap_values = self.explainer.shap_values(x)
        # Binary classifiers via TreeExplainer can return a list [class0, class1].
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        return shap_values

    def global_feature_importance(self, x: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        shap_values = self.compute_shap_values(x)
        importance = pd.DataFrame({
            "feature": x.columns,
            "mean_abs_shap": abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)

        importance.to_csv(self.reports_dir / "shap_global_importance.csv", index=False)
        return importance

    def plot_summary(self, x: pd.DataFrame, save_path: str = "reports/shap_summary_plot.png"):
        shap_values = self.compute_shap_values(x)
        plt.figure()
        shap.summary_plot(shap_values, x, show=False)
        plt.tight_layout()
        plt.savefig(resolve_path(save_path), bbox_inches="tight")
        plt.close()

    def plot_dependence(self, x: pd.DataFrame, feature: str, save_path: str = None):
        shap_values = self.compute_shap_values(x)
        save_path = save_path or f"reports/shap_dependence_{feature}.png"
        plt.figure()
        shap.dependence_plot(feature, shap_values, x, show=False)
        plt.tight_layout()
        plt.savefig(resolve_path(save_path), bbox_inches="tight")
        plt.close()

    def explain_instance(self, x_row: pd.DataFrame) -> Dict[str, Any]:
        """Local explanation for a single applicant — feeds the dashboard's
        'why is this applicant high/low risk' panel and the API's
        /feature-importance endpoint for a specific prediction."""
        shap_values = self.compute_shap_values(x_row)
        contributions = dict(zip(x_row.columns, shap_values[0].tolist()))
        ranked = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))

        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, tuple)):
            base_value = base_value[1]

        return {
            "base_value": float(base_value),
            "feature_contributions": {k: round(float(v), 5) for k, v in ranked.items()},
            "top_positive_drivers": [k for k, v in ranked.items() if v > 0][:5],
            "top_negative_drivers": [k for k, v in ranked.items() if v < 0][:5],
        }

    def plot_waterfall(self, x_row: pd.DataFrame, save_path: str = "reports/shap_waterfall_instance.png"):
        shap_values = self.explainer(x_row)
        plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig(resolve_path(save_path), bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    from src.utils.helper import load_pickle

    config = load_config()
    model = load_pickle(resolve_path(config["training"]["champion_model_path"]))
    x_test = pd.read_csv(resolve_path(config["data"]["test_path"]))
    x_test = x_test.drop(columns=[config["data"]["target_column"]])

    explainer = ShapExplainer(model, x_test.sample(min(200, len(x_test)), random_state=42), config)
    importance = explainer.global_feature_importance(x_test.sample(min(500, len(x_test)), random_state=42))
    print(importance)
