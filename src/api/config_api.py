"""Configuration endpoints.

GET /api/v1/config - View current configuration
PUT /api/v1/config - Update configuration at runtime
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_config_manager
from src.config.manager import ConfigManager, ConfigError

router = APIRouter(prefix="/api/v1")


class ConfigUpdateRequest(BaseModel):
    """Partial config update payload."""

    updates: dict


@router.get("/config")
async def get_config(
    config_manager: ConfigManager = Depends(get_config_manager),
) -> dict:
    """Return the current application configuration."""
    try:
        config = config_manager.get_config()
        return config.model_dump()
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/config")
async def update_config(
    body: ConfigUpdateRequest,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> dict:
    """Update configuration at runtime.

    Merges updates into the current config, writes to disk, and reloads.
    """
    try:
        current = config_manager.get_config()
        current_dict = current.model_dump()

        _deep_merge(current_dict, body.updates)

        import yaml

        config_path = config_manager._config_path
        if config_path is not None and config_path.exists():
            config_path.write_text(
                yaml.dump(current_dict, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
            config_manager.reload()

        return {
            "message": "Configuration updated",
            "config": config_manager.get_config().model_dump(),
        }

    except ConfigError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "config_update_error",
                "message": str(exc),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        ) from exc


def _deep_merge(base: dict, updates: dict) -> None:
    """Recursively merge *updates* into *base* in place."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
