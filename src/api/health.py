"""Health and system info endpoints.

GET /health — System health status
GET /info   — Version, active LLM provider, active model, vector store, enabled MCP servers
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_config_manager
from src.config.manager import ConfigManager
from src.models.schemas import SystemInfoResponse

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Return system health status."""
    return {"status": "ok"}


@router.get("/info", response_model=SystemInfoResponse)
async def info(
    config_manager: ConfigManager = Depends(get_config_manager),
) -> SystemInfoResponse:
    """Return system information."""
    config = config_manager.get_config()

    enabled_servers = [
        name for name, srv in config.mcp_servers.items() if srv.enabled
    ]

    return SystemInfoResponse(
        version=config.app.version,
        active_llm_provider=config.llm.default_provider,
        active_llm_model=config.llm.get_active().model,
        active_vector_store=config.vector_store.provider,
        enabled_mcp_servers=enabled_servers,
    )
