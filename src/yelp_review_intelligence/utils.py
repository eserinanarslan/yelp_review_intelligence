from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .exceptions import ModelPersistenceError


def save_model(model: Any, path: str | Path) -> None:
    """
    Save a trained model artifact.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

    except Exception as exc:
        raise ModelPersistenceError(f"Failed to save model to: {path}") from exc


def load_model(path: str | Path) -> Any:
    """
    Load a trained model artifact.
    """
    try:
        path = Path(path)

        if not path.exists():
            raise ModelPersistenceError(f"Model file does not exist: {path}")

        return joblib.load(path)

    except ModelPersistenceError:
        raise
    except Exception as exc:
        raise ModelPersistenceError(f"Failed to load model from: {path}") from exc


def save_json(obj: Any, path: str | Path) -> None:
    """
    Save a JSON-serializable object.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(obj, file, indent=2, ensure_ascii=False, default=str)

    except Exception as exc:
        raise ModelPersistenceError(f"Failed to save JSON to: {path}") from exc