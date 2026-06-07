"""Statistics endpoints.

GET /api/v1/stats - Token usage, conversation count, tool call distribution, avg response time
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.chat import get_chat_stats
from src.models.schemas import StatsResponse, TokenUsage

router = APIRouter(prefix="/api/v1")


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Return aggregated system statistics."""
    s = get_chat_stats()

    total_convos = s["total_conversations"]
    avg_response_ms = (
        s["total_response_time_ms"] / total_convos if total_convos > 0 else 0.0
    )

    return StatsResponse(
        total_conversations=total_convos,
        total_tokens=TokenUsage(
            prompt_tokens=s["total_prompt_tokens"],
            completion_tokens=s["total_completion_tokens"],
            total_tokens=s["total_tokens"],
        ),
        tool_call_distribution=dict(s["tool_calls"]),
        average_response_time_ms=round(avg_response_ms, 2),
    )
