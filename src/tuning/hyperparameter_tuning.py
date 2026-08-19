"""
Hyperparameter optimization for the champion model family (LightGBM
by default) using Optuna, maximizing cross-validated ROC-AUC. Logs
every trial to MLflow as a nested run so the full search history is
auditable.
"""

from typing import Any, Dict

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils.helper import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.tuning_config = self.config["tuning"]
        mlflow.set_tracking_uri(resolve_path(self.config["mlflow"]["tracking_uri"]).as_uri())
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

    def _objective(self, trial: optuna.Trial, x_train: pd.DataFrame, y_train: pd.Series) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 800),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        model = lgb.LGBMClassifier(
            **params,
            random_state=self.config["project"]["random_seed"],
            n_jobs=-1,
            verbosity=-1,
        )

        cv = StratifiedKFold(n_splits=self.config["training"]["cv_folds"], shuffle=True,
                              random_state=self.config["project"]["random_seed"])
        score = cross_val_score(model, x_train, y_train, cv=cv, scoring=self.tuning_config["metric"], n_jobs=-1).mean()

        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric(self.tuning_config["metric"], score)

        return score

    def run(self, x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        with mlflow.start_run(run_name="optuna_search"):
            study = optuna.create_study(direction=self.tuning_config["direction"])
            study.optimize(
                lambda trial: self._objective(trial, x_train, y_train),
                n_trials=self.tuning_config["n_trials"],
                show_progress_bar=False,
            )
            mlflow.log_params(study.best_params)
            mlflow.log_metric(f"best_{self.tuning_config['metric']}", study.best_value)

        logger.info(f"Best trial: value={study.best_value:.4f}, params={study.best_params}")
        return {"best_params": study.best_params, "best_value": study.best_value, "study": study}


if __name__ == "__main__":
    from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    x_train, x_test, y_train, y_test = pipeline.run()

    tuner = HyperparameterTuner()
    outcome = tuner.run(x_train, y_train)
    print(outcome["best_params"])
