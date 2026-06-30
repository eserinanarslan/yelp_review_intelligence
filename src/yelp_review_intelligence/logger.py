from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "yelp_review_intelligence",
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a reusable project logger.

    Logs are written both to the console and to a file.
    This makes local debugging and production troubleshooting easier.
    """
    logger = logging.getLogger(name)

    # Avoid duplicated handlers when modules are imported multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger