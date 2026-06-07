"""Chat endpoints.

POST      /api/v1/chat        - Synchronous chat (Intent Router  Agent Loop  response)
WebSocket /api/v1/chat/stream  - Streaming chat (placeholder)
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from src.agent.loop import AgentLoop
from src.api.dependencies import get_agent_loop, get_intent_router, get_session_manager
from src.models.schemas import ChatRequest, ChatResponse, TokenUsage
from src.router.intent import IntentRouter
from src.session.manager import SessionManager

router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------------
# In-memory stats tracking (shared with stats endpoint)
# ---------------------------------------------------------------------------

_chat_stats: dict = {
    "total_conversations": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "tool_calls": {},  # tool_name -> count
    "total_response_time_ms": 0.0,
}


def get_chat_stats() -> dict:
    """Return the in-memory chat stats dict (used by stats endpoint)."""
    return _chat_stats


# ---------------------------------------------------------------------------
# POST /api/v1/chat
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    intent_router: IntentRouter = Depends(get_intent_router),
    agent_loop: AgentLoop = Depends(get_agent_loop),
    session_manager: SessionManager = Depends(get_session_manager),
) -> ChatResponse:
    """Synchronous chat: classify intent, run agent loop, return response."""
    start = time.perf_counter()

    # Session handling: reuse existing or create new
    session = None
    if request.session_id:
        session = session_manager.get_session(request.session_id)
    if session is None:
        session = session_manager.create_session()

    # Add user message to session
    session_manager.add_message(
        session.session_id, {"role": "user", "content": request.message}
    )

    # Classify intent
    intent_result = intent_router.classify(request.message)

    # Chitchat: return pre-defined response without LLM call
    if intent_result.intent == "chitchat" and intent_result.response:
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        session_manager.add_message(
            session.session_id,
            {"role": "assistant", "content": intent_result.response},
        )
        _update_stats(usage, [], time.perf_counter() - start)
        return ChatResponse(
            response=intent_result.response,
            session_id=session.session_id,
            citations=[],
            tool_traces=[],
            usage=usage,
        )

    # Build message history for LLM
    messages = list(session.messages)

    # Run agent loop
    agent_response = await agent_loop.run(
        messages=messages,
        session=session,
        intent=intent_result,
    )

    # Add assistant response to session
    session_manager.add_message(
        session.session_id,
        {"role": "assistant", "content": agent_response.content},
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    _update_stats(agent_response.usage, agent_response.tool_traces, elapsed_ms / 1000)

    return ChatResponse(
        response=agent_response.content,
        session_id=session.session_id,
        citations=agent_response.citations,
        tool_traces=agent_response.tool_traces,
        usage=agent_response.usage,
    )


# ---------------------------------------------------------------------------
# WebSocket /api/v1/chat/stream (placeholder)
# ---------------------------------------------------------------------------


@router.websocket("/chat/stream")
async def chat_stream(websocket: WebSocket) -> None:
    """Streaming chat via WebSocket (basic placeholder implementation)."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            await websocket.send_json(
                {
                    "type": "message",
                    "content": f"[Streaming not yet implemented] Received: {message}",
                    "done": True,
                }
            )
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _update_stats(
    usage: TokenUsage, tool_traces: list, elapsed_s: float
) -> None:
    """Update in-memory stats counters."""
    _chat_stats["total_conversations"] += 1
    _chat_stats["total_prompt_tokens"] += usage.prompt_tokens
    _chat_stats["total_completion_tokens"] += usage.completion_tokens
    _chat_stats["total_tokens"] += usage.total_tokens
    _chat_stats["total_response_time_ms"] += elapsed_s * 1000
    for trace in tool_traces:
        name = trace.tool_name
        _chat_stats["tool_calls"][name] = _chat_stats["tool_calls"].get(name, 0) + 1
