"""MCP status endpoints.

GET /api/v1/mcp/status - Report status of all MCP servers
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_mcp_manager
from src.mcp.manager import MCPManager
from src.models.schemas import MCPStatusResponse

router = APIRouter(prefix="/api/v1")


@router.get("/mcp/status", response_model=MCPStatusResponse)
async def mcp_status(
    mcp_manager: MCPManager = Depends(get_mcp_manager),
) -> MCPStatusResponse:
    """Return the status of all configured MCP servers."""
    statuses = mcp_manager.get_status()
    return MCPStatusResponse(servers=statuses)
