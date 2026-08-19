"""
Trains and compares four candidate models — Logistic Regression
(baseline), Random Forest, XGBoost, LightGBM — using stratified
K-fold cross-validation, logs everything to MLflow, and selects the
champion model by held-out ROC-AUC.
"""

from typing import Any, Dict

import lightgbm as lgb
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils.helper import load_config, resolve_path, save_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_candidate_models(config: Dict[str, Any]) -> Dict[str, Any]:
    seed = config["project"]["random_seed"]
    return {
        "logistic_regression": LogisticRegression(
            max_iter=config["logistic_regression"]["max_iter"],
            C=config["logistic_regression"]["C"],
            random_state=seed,
        ),
        # Baseline. Fast, interpretable, sets the floor every other model
        # must beat before it earns the extra complexity.
        "random_forest": RandomForestClassifier(
            n_estimators=config["random_forest"]["n_estimators"],
            max_depth=config["random_forest"]["max_depth"],
            min_samples_leaf=config["random_forest"]["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1,
        ),
        # Handles nonlinear interactions (e.g. DTI x credit score) without
        # manual feature crosses; robust to outliers left after capping.
        "xgboost": xgb.XGBClassifier(
            n_estimators=config["xgboost"]["n_estimators"],
            max_depth=config["xgboost"]["max_depth"],
            learning_rate=config["xgboost"]["learning_rate"],
            subsample=config["xgboost"]["subsample"],
            colsample_bytree=config["xgboost"]["colsample_bytree"],
            eval_metric="auc",
            random_state=seed,
            n_jobs=-1,
        ),
        # Typically the strongest performer on tabular credit data;
        # included as the primary production candidate.
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=config["lightgbm"]["n_estimators"],
            max_depth=config["lightgbm"]["max_depth"],
            num_leaves=config["lightgbm"]["num_leaves"],
            learning_rate=config["lightgbm"]["learning_rate"],
            subsample=config["lightgbm"]["subsample"],
            colsample_bytree=config["lightgbm"]["colsample_bytree"],
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
        # Fastest to train at scale, competitive with XGBoost, useful when
        # retraining frequently in production.
    }


class ModelTrainer:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        mlflow.set_tracking_uri(resolve_path(self.config["mlflow"]["tracking_uri"]).as_uri())
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

    def cross_validate(self, model, x_train: pd.DataFrame, y_train: pd.Series) -> float:
        cv = StratifiedKFold(
            n_splits=self.config["training"]["cv_folds"],
            shuffle=True,
            random_state=self.config["project"]["random_seed"],
        )
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring=self.config["training"]["scoring_metric"], n_jobs=-1)
        return float(scores.mean())

    def train_all(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
        from sklearn.metrics import roc_auc_score

        models = get_candidate_models(self.config)
        results = {}

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                logger.info(f"Training {name}...")
                cv_auc = self.cross_validate(model, x_train, y_train)
                model.fit(x_train, y_train)

                test_proba = model.predict_proba(x_test)[:, 1]
                test_auc = roc_auc_score(y_test, test_proba)

                mlflow.log_params(model.get_params())
                mlflow.log_metric("cv_roc_auc", cv_auc)
                mlflow.log_metric("test_roc_auc", test_auc)
                mlflow.sklearn.log_model(model, artifact_path="model")

                results[name] = {"model": model, "cv_roc_auc": cv_auc, "test_roc_auc": test_auc}
                logger.info(f"{name}: cv_auc={cv_auc:.4f}, test_auc={test_auc:.4f}")

        return results

    def select_champion(self, results: Dict[str, Any]) -> str:
        champion_name = max(results, key=lambda name: results[name]["test_roc_auc"])
        logger.info(f"Champion model: {champion_name} (test_auc={results[champion_name]['test_roc_auc']:.4f})")
        return champion_name

    def persist_champion(self, results: Dict[str, Any], champion_name: str) -> None:
        champion_model = results[champion_name]["model"]
        save_pickle(champion_model, resolve_path(self.config["training"]["champion_model_path"]))
        logger.info(f"Champion model '{champion_name}' saved to {self.config['training']['champion_model_path']}")

    def run(self, x_train, x_test, y_train, y_test) -> Dict[str, Any]:
        results = self.train_all(x_train, x_test, y_train, y_test)
        champion_name = self.select_champion(results)
        self.persist_champion(results, champion_name)
        return {"results": results, "champion": champion_name}


if __name__ == "__main__":
    from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    x_train, x_test, y_train, y_test = pipeline.run()

    trainer = ModelTrainer()
    outcome = trainer.run(x_train, x_test, y_train, y_test)
    print(f"Champion: {outcome['champion']}")
