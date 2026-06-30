from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .exceptions import ConfigError


@dataclass
class ProjectConfig:
    raw: Dict[str, Any]
    project_root: Path

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ProjectConfig":
        """
        Load project configuration from a YAML file.
        """
        try:
            config_path = Path(config_path).resolve()

            if not config_path.exists():
                raise ConfigError(f"Config file does not exist: {config_path}")

            with open(config_path, "r", encoding="utf-8") as file:
                raw_config = yaml.safe_load(file)

            if not raw_config:
                raise ConfigError(f"Config file is empty: {config_path}")

            return cls(raw=raw_config, project_root=config_path.parent.parent)

        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(f"Failed to load config file: {config_path}") from exc

    def path(self, *keys: str) -> Path:
        value = self.raw

        try:
            for key in keys:
                value = value[key]

            return self.project_root / value

        except KeyError as exc:
            raise ConfigError(f"Missing config key: {' -> '.join(keys)}") from exc

    @property
    def random_state(self) -> int:
        return int(self.raw["project"]["random_state"])

    @property
    def data_dir(self) -> Path:
        return self.path("paths", "data_dir")

    @property
    def processed_dir(self) -> Path:
        path = self.path("paths", "processed_dir")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_dir(self) -> Path:
        path = self.path("paths", "model_dir")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def output_dir(self) -> Path:
        path = self.path("paths", "output_dir")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def raw_file_path(self, file_key: str) -> Path:
        return self.data_dir / self.raw["raw_files"][file_key]