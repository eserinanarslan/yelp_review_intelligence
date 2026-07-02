from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .exceptions import ConfigError


@dataclass
class ProjectConfig:
    """
    Central configuration object for the entire project.

    Loads all settings from a YAML file and provides convenient
    access to project paths and configuration parameters.
    """

    raw: Dict[str, Any]
    project_root: Path

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ProjectConfig":
        """
        Load project configuration from a YAML file.

        This method validates that the configuration file exists
        and is not empty before creating the configuration object.
        """
        try:
            config_path = Path(config_path).resolve()

            # Verify that the configuration file exists
            if not config_path.exists():
                raise ConfigError(f"Config file does not exist: {config_path}")

            # Load YAML configuration
            with open(config_path, "r", encoding="utf-8") as file:
                raw_config = yaml.safe_load(file)

            # Ensure the configuration is not empty
            if not raw_config:
                raise ConfigError(f"Config file is empty: {config_path}")

            return cls(
                raw=raw_config,
                project_root=config_path.parent.parent,
            )

        except ConfigError:
            raise

        except Exception as exc:
            raise ConfigError(
                f"Failed to load config file: {config_path}"
            ) from exc

    def path(self, *keys: str) -> Path:
        """
        Resolve a filesystem path from the YAML configuration.

        Example:
            path("paths", "model_dir")
        """

        value = self.raw

        try:
            for key in keys:
                value = value[key]

            return self.project_root / value

        except KeyError as exc:
            raise ConfigError(
                f"Missing config key: {' -> '.join(keys)}"
            ) from exc

    @property
    def random_state(self) -> int:
        """
        Return the project's random seed.
        """
        return int(self.raw["project"]["random_state"])

    @property
    def data_dir(self) -> Path:
        """
        Directory containing raw input datasets.
        """
        return self.path("paths", "data_dir")

    @property
    def processed_dir(self) -> Path:
        """
        Directory containing processed datasets.

        Automatically creates the directory if it does not exist.
        """
        path = self.path("paths", "processed_dir")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_dir(self) -> Path:
        """
        Directory for storing trained models.

        Automatically creates the directory if needed.
        """
        path = self.path("paths", "model_dir")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def output_dir(self) -> Path:
        """
        Directory for reports, metrics and business outputs.

        Automatically creates the directory if needed.
        """
        path = self.path("paths", "output_dir")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def raw_file_path(self, file_key: str) -> Path:
        """
        Return the full path of a raw dataset file
        defined in the configuration.
        """
        return self.data_dir / self.raw["raw_files"][file_key]