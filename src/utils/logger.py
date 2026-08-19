"""
Centralized logging configuration for the platform.
Every module imports get_logger(__name__) instead of calling
logging.basicConfig directly, so log format/level stays consistent
across ingestion, training, the API, and the dashboard.
"""

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "reports" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-scoped logger that writes to stdout and to a shared file."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (e.g. re-imported in a notebook) — don't duplicate handlers.
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
