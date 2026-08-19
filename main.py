"""
Single entry point that runs the full pipeline end-to-end, or any
individual stage, driven off config/config.yaml. This is what CI and
`docker run` call.

Usage:
    python main.py --stage all
    python main.py --stage preprocess
    python main.py --stage train
    python main.py --stage tune
    python main.py --stage evaluate
    python main.py --stage explain
    python main.py --stage monitor
"""

import argparse

from src.evaluation.evaluate_model import ModelEvaluator
from src.explainability.shap_explainer import ShapExplainer
from src.monitoring.drift_monitor import DriftMonitor
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.training.train_model import ModelTrainer
from src.tuning.hyperparameter_tuning import HyperparameterTuner
from src.utils.helper import load_config, load_pickle, resolve_path, save_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_preprocess(config):
    pipeline = PreprocessingPipeline(config)
    return pipeline.run()


def run_train(config, x_train, x_test, y_train, y_test):
    trainer = ModelTrainer(config)
    return trainer.run(x_train, x_test, y_train, y_test)


def run_tune(config, x_train, y_train):
    tuner = HyperparameterTuner(config)
    return tuner.run(x_train, y_train)


def run_evaluate(config):
    import pandas as pd

    model = load_pickle(resolve_path(config["training"]["champion_model_path"]))
    x_test = pd.read_csv(resolve_path(config["data"]["test_path"]))
    y_test = x_test.pop(config["data"]["target_column"])
    evaluator = ModelEvaluator(config)
    return evaluator.evaluate(model, x_test, y_test)


def run_explain(config):
    import pandas as pd

    model = load_pickle(resolve_path(config["training"]["champion_model_path"]))
    x_test = pd.read_csv(resolve_path(config["data"]["test_path"]))
    x_test = x_test.drop(columns=[config["data"]["target_column"]])
    explainer = ShapExplainer(model, x_test.sample(min(200, len(x_test)), random_state=42), config)
    importance = explainer.global_feature_importance(x_test.sample(min(500, len(x_test)), random_state=42))
    explainer.plot_summary(x_test.sample(min(300, len(x_test)), random_state=42))
    logger.info(f"Top features:\n{importance.head(10)}")


def run_monitor(config):
    monitor = DriftMonitor(config)
    return monitor.run()


def main():
    parser = argparse.ArgumentParser(description="Credit Risk Intelligence Platform pipeline runner")
    parser.add_argument(
        "--stage", default="all",
        choices=["all", "preprocess", "train", "tune", "evaluate", "explain", "monitor"],
    )
    args = parser.parse_args()

    config = load_config()

    if args.stage in ("all", "preprocess"):
        logger.info("=== STAGE: PREPROCESS ===")
        x_train, x_test, y_train, y_test = run_preprocess(config)

    if args.stage in ("all", "train"):
        logger.info("=== STAGE: TRAIN ===")
        if args.stage == "train":
            x_train, x_test, y_train, y_test = run_preprocess(config)
        run_train(config, x_train, x_test, y_train, y_test)

    if args.stage == "tune":
        x_train, x_test, y_train, y_test = run_preprocess(config)
        logger.info("=== STAGE: TUNE ===")
        result = run_tune(config, x_train, y_train)
        logger.info(f"Best params: {result['best_params']}")

    if args.stage in ("all", "evaluate"):
        logger.info("=== STAGE: EVALUATE ===")
        run_evaluate(config)

    if args.stage in ("all", "explain"):
        logger.info("=== STAGE: EXPLAIN ===")
        run_explain(config)

    if args.stage in ("all", "monitor"):
        logger.info("=== STAGE: MONITOR ===")
        run_monitor(config)

    logger.info("Pipeline run complete.")


if __name__ == "__main__":
    main()
