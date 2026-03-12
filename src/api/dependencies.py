"""Shared dependency factories for API endpoints.

These module-level singletons are set during application startup (in main.py).
Endpoint functions use ``Depends(get_xxx)`` to receive them.
"""

from __future__ import annotations

from src.agent.loop import AgentLoop
from src.chunking.engine import ChunkingEngine
from src.config.manager import ConfigManager
from src.mcp.manager import MCPManager
from src.router.intent import IntentRouter
from src.session.manager import SessionManager
from src.store.reference import ReferenceStore
from src.vectorstore.adapter import VectorStoreAdapter

# Module-level singletons — set during app startup
_config_manager: ConfigManager | None = None
_intent_router: IntentRouter | None = None
_agent_loop: AgentLoop | None = None
_session_manager: SessionManager | None = None
_mcp_manager: MCPManager | None = None
_reference_store: ReferenceStore | None = None
_chunking_engine: ChunkingEngine | None = None
_vector_store: VectorStoreAdapter | None = None


def configure(
    config_manager: ConfigManager | None = None,
    intent_router: IntentRouter | None = None,
    agent_loop: AgentLoop | None = None,
    session_manager: SessionManager | None = None,
    mcp_manager: MCPManager | None = None,
    reference_store: ReferenceStore | None = None,
    chunking_engine: ChunkingEngine | None = None,
    vector_store: VectorStoreAdapter | None = None,
) -> None:
    """Set the module-level singletons (called once at startup)."""
    global _config_manager, _intent_router, _agent_loop, _session_manager
    global _mcp_manager, _reference_store, _chunking_engine, _vector_store
    if config_manager is not None:
        _config_manager = config_manager
    if intent_router is not None:
        _intent_router = intent_router
    if agent_loop is not None:
        _agent_loop = agent_loop
    if session_manager is not None:
        _session_manager = session_manager
    if mcp_manager is not None:
        _mcp_manager = mcp_manager
    if reference_store is not None:
        _reference_store = reference_store
    if chunking_engine is not None:
        _chunking_engine = chunking_engine
    if vector_store is not None:
        _vector_store = vector_store


def get_config_manager() -> ConfigManager:
    assert _config_manager is not None, "ConfigManager not configured"
    return _config_manager


def get_intent_router() -> IntentRouter:
    assert _intent_router is not None, "IntentRouter not configured"
    return _intent_router


def get_agent_loop() -> AgentLoop:
    assert _agent_loop is not None, "AgentLoop not configured"
    return _agent_loop


def get_session_manager() -> SessionManager:
    assert _session_manager is not None, "SessionManager not configured"
    return _session_manager


def get_mcp_manager() -> MCPManager:
    assert _mcp_manager is not None, "MCPManager not configured"
    return _mcp_manager


def get_reference_store() -> ReferenceStore:
    assert _reference_store is not None, "ReferenceStore not configured"
    return _reference_store


def get_chunking_engine() -> ChunkingEngine:
    assert _chunking_engine is not None, "ChunkingEngine not configured"
    return _chunking_engine


def get_vector_store() -> VectorStoreAdapter:
    assert _vector_store is not None, "VectorStoreAdapter not configured"
    return _vector_store
