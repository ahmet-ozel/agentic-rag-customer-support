"""Customer endpoints.

GET /api/v1/customers - Customer info query (placeholder for MVP)
"""

from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/v1")


@router.get("/customers")
async def list_customers(
    q: str | None = Query(None, description="Search query for customer name or email"),
) -> dict:
    """Query customer information (placeholder).

    In production this would route through postgres-mcp.
    For MVP, returns a placeholder response.
    """
    return {
        "message": "Customer query endpoint (placeholder - requires postgres-mcp)",
        "query": q,
        "customers": [],
    }
