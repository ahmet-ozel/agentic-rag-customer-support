"""Configuration manager for AgentDesk RAG Platform.

Loads config.yaml, resolves ${ENV_VAR} patterns, validates with Pydantic,
and provides hot-reload support. Secrets are loaded from .env via python-dotenv.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from src.config.models import AppConfig

# Module-level singleton
_config: AppConfig | None = None


class ConfigError(Exception):
    """Raised when configuration is invalid."""


class ConfigManager:
    """Manages application configuration lifecycle.

    - Loads .env secrets on init
    - Reads config.yaml and resolves ${ENV_VAR} patterns
    - Validates with Pydantic AppConfig model
    - Supports hot-reload at runtime
    """

    def __init__(self) -> None:
        load_dotenv()
        self._config: AppConfig | None = None
        self._config_path: Path | None = None

    def load(self, config_path: str = "config.yaml") -> AppConfig:
        """Read YAML file, resolve env vars, validate with Pydantic."""
        global _config
        self._config_path = Path(config_path)

        if not self._config_path.exists():
            raise ConfigError(f"Config file not found: {self._config_path}")

        raw_text = self._config_path.read_text(encoding="utf-8")
        raw_dict = yaml.safe_load(raw_text) or {}

        resolved = self._resolve_env_vars(raw_dict)

        try:
            self._config = AppConfig(**resolved)
        except ValidationError as exc:
            field_errors = []
            for err in exc.errors():
                loc = " -> ".join(str(part) for part in err["loc"])
                field_errors.append(f"  [{loc}] {err['msg']}")
            raise ConfigError(
                "Invalid configuration:\n" + "\n".join(field_errors)
            ) from exc

        _config = self._config
        return self._config

    def reload(self) -> AppConfig:
        """Hot-reload config at runtime by re-reading the file."""
        if self._config_path is None:
            raise ConfigError("Cannot reload: config has not been loaded yet")
        load_dotenv(override=True)
        return self.load(str(self._config_path))

    def get_config(self) -> AppConfig:
        """Return current validated config."""
        if self._config is None:
            raise ConfigError("Config not loaded. Call load() first.")
        return self._config

    @staticmethod
    def _resolve_env_vars(value):  # noqa: ANN001
        """Recursively resolve ${ENV_VAR} patterns in config values."""
        if isinstance(value, str):
            return re.sub(
                r"\$\{([^}]+)\}",
                lambda m: os.environ.get(m.group(1), ""),
                value,
            )
        if isinstance(value, dict):
            return {k: ConfigManager._resolve_env_vars(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ConfigManager._resolve_env_vars(item) for item in value]
        return value

    # Public alias so callers can use it directly
    resolve_env_vars = _resolve_env_vars


def get_config() -> AppConfig:
    """Module-level convenience accessor for the singleton config."""
    if _config is None:
        raise ConfigError("Config not loaded. Call ConfigManager().load() first.")
    return _config
