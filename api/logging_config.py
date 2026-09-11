"""Application file logging configuration."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(log_file: str | Path) -> logging.Logger:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("shizhen_anonymizer")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for existing in list(logger.handlers):
        if getattr(existing, "_shizhen_handler", False):
            logger.removeHandler(existing)
            existing.close()

    handler = RotatingFileHandler(
        path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler._shizhen_handler = True  # type: ignore[attr-defined]
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    return logger
