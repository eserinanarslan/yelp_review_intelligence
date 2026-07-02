from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .exceptions import ModelPersistenceError


def save_model(model: Any, path: str | Path) -> None:
    """
    Save a trained machine learning model to disk.

    Centralizing model persistence keeps serialization logic
    consistent across the project.
    """
    try:
        path = Path(path)

        # Create the target directory if it does not exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize the model using Joblib
        joblib.dump(model, path)

    except Exception as exc:
        raise ModelPersistenceError(
            f"Failed to save model to: {path}"
        ) from exc


def load_model(path: str | Path) -> Any:
    """
    Load a previously trained machine learning model.

    Used by the inference service during application startup.
    """
    try:
        path = Path(path)

        # Verify that the model artifact exists
        if not path.exists():
            raise ModelPersistenceError(
                f"Model file does not exist: {path}"
            )

        # Deserialize the model
        return joblib.load(path)

    except ModelPersistenceError:
        raise

    except Exception as exc:
        raise ModelPersistenceError(
            f"Failed to load model from: {path}"
        ) from exc


def save_json(obj: Any, path: str | Path) -> None:
    """
    Save a Python object as a formatted JSON file.

    Used for model metrics, dashboard outputs,
    business insights and API-ready artifacts.
    """
    try:
        path = Path(path)

        # Create the output directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize the object as JSON
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                obj,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    except Exception as exc:
        raise ModelPersistenceError(
            f"Failed to save JSON to: {path}"
        ) from exc