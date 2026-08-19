"""
Shared helper utilities: config loading, path resolution, and small
serialization helpers used across ingestion, training, and prediction.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load the YAML config and resolve it relative to the project root."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found at {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Turn a config-relative path into an absolute path under the project root."""
    return PROJECT_ROOT / relative_path


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: Path) -> None:
    ensure_parent_dir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: Path) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
