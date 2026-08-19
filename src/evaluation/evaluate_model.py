"""
Full evaluation suite for a trained classifier: standard metrics,
confusion matrix, ROC/PR curves, calibration curve, and lift/gain
charts — the set a credit-risk reviewer actually asks for, beyond
plain accuracy (which is close to useless on an imbalanced target).
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve,
)

from src.utils.helper import load_config, resolve_path, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list = field(default_factory=list)


class ModelEvaluator:
    def __init__(self, config: Dict[str, Any] = None, threshold: float = 0.5):
        self.config = config or load_config()
        self.threshold = threshold
        self.reports_dir = resolve_path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(self, y_true: pd.Series, y_proba: np.ndarray) -> EvaluationMetrics:
        y_pred = (y_proba >= self.threshold).astype(int)
        return EvaluationMetrics(
            accuracy=round(accuracy_score(y_true, y_pred), 4),
            precision=round(precision_score(y_true, y_pred, zero_division=0), 4),
            recall=round(recall_score(y_true, y_pred, zero_division=0), 4),
            f1=round(f1_score(y_true, y_pred, zero_division=0), 4),
            roc_auc=round(roc_auc_score(y_true, y_proba), 4),
            pr_auc=round(average_precision_score(y_true, y_proba), 4),
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        )

    def plot_roc_curve(self, y_true, y_proba, save_path="reports/roc_curve.png"):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_proba):.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(resolve_path(save_path)); plt.close()

    def plot_precision_recall_curve(self, y_true, y_proba, save_path="reports/pr_curve.png"):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall Curve"); plt.tight_layout()
        plt.savefig(resolve_path(save_path)); plt.close()

    def plot_confusion_matrix(self, y_true, y_proba, save_path="reports/confusion_matrix.png"):
        y_pred = (y_proba >= self.threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xticks([0, 1], ["No Default", "Default"])
        plt.yticks([0, 1], ["No Default", "Default"])
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
        plt.tight_layout(); plt.savefig(resolve_path(save_path)); plt.close()

    def plot_calibration_curve(self, y_true, y_proba, save_path="reports/calibration_curve.png"):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        plt.figure(figsize=(6, 5))
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
        plt.title("Calibration Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(resolve_path(save_path)); plt.close()

    def plot_lift_gain_chart(self, y_true, y_proba, save_path="reports/lift_gain_chart.png"):
        df = pd.DataFrame({"y_true": y_true.values, "y_proba": y_proba}).sort_values("y_proba", ascending=False)
        df["decile"] = pd.qcut(np.arange(len(df)), 10, labels=False)
        total_positives = df["y_true"].sum()
        gain = df.groupby("decile")["y_true"].sum().sort_index(ascending=False).cumsum() / total_positives
        lift = gain.values / (np.arange(1, 11) / 10)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(range(1, 11), gain.values, marker="o")
        axes[0].set_title("Cumulative Gain Chart"); axes[0].set_xlabel("Decile"); axes[0].set_ylabel("Cumulative % of defaults captured")
        axes[1].plot(range(1, 11), lift, marker="o", color="darkorange")
        axes[1].axhline(1.0, linestyle="--", color="gray")
        axes[1].set_title("Lift Chart"); axes[1].set_xlabel("Decile"); axes[1].set_ylabel("Lift")
        plt.tight_layout(); plt.savefig(resolve_path(save_path)); plt.close()

    def evaluate(self, model, x_test: pd.DataFrame, y_test: pd.Series, model_name: str = "champion") -> EvaluationMetrics:
        y_proba = model.predict_proba(x_test)[:, 1]
        metrics = self.compute_metrics(y_test, y_proba)

        self.plot_roc_curve(y_test, y_proba)
        self.plot_precision_recall_curve(y_test, y_proba)
        self.plot_confusion_matrix(y_test, y_proba)
        self.plot_calibration_curve(y_test, y_proba)
        self.plot_lift_gain_chart(y_test, y_proba)

        save_json(metrics.__dict__, resolve_path(f"reports/{model_name}_evaluation_metrics.json"))
        logger.info(f"Evaluation complete for {model_name}: {metrics}")
        return metrics


if __name__ == "__main__":
    import pandas as pd
    from src.utils.helper import load_pickle

    config = load_config()
    model = load_pickle(resolve_path(config["training"]["champion_model_path"]))
    x_test = pd.read_csv(resolve_path(config["data"]["test_path"]))
    y_test = x_test.pop(config["data"]["target_column"])

    evaluator = ModelEvaluator(config)
    print(evaluator.evaluate(model, x_test, y_test))
