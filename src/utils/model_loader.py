"""
Single place responsible for loading the champion model and its
preprocessing artifacts (fitted cleaner + column transformer + feature
names). Both the API and the dashboard import this instead of
duplicating pickle-loading logic.
"""

from dataclasses import dataclass
from typing import Any, List

from src.utils.helper import load_config, load_json, load_pickle, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoadedArtifacts:
    model: Any
    cleaner: Any
    column_transformer: Any
    feature_names: List[str]
    config: dict


def load_artifacts(config: dict = None) -> LoadedArtifacts:
    config = config or load_config()

    model_path = resolve_path(config["training"]["champion_model_path"])
    cleaner_path = resolve_path("models/cleaner.pkl")
    transformer_path = resolve_path(config["training"]["preprocessor_path"])
    feature_names_path = resolve_path(config["training"]["feature_names_path"])

    for path in [model_path, cleaner_path, transformer_path, feature_names_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact missing: {path}. Run the training pipeline first: "
                f"python main.py --stage all"
            )

    logger.info("Loading champion model and preprocessing artifacts...")
    return LoadedArtifacts(
        model=load_pickle(model_path),
        cleaner=load_pickle(cleaner_path),
        column_transformer=load_pickle(transformer_path),
        feature_names=load_json(feature_names_path),
        config=config,
    )
