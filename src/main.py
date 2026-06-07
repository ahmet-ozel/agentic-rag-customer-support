"""AgentDesk RAG Platform - FastAPI application entry point.

Run with:
    python -m src.main
    uvicorn src.main:app --host 0.0.0.0 --port 8000

Validates: Requirements 4.1, 3.3, 5.4
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config.manager import ConfigManager
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (populated during lifespan)
# ---------------------------------------------------------------------------

_config_manager = ConfigManager()


def _build_app() -> FastAPI:
    """Construct the FastAPI application with lifespan and routers."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Startup / shutdown lifecycle."""
        # --- Startup ---
        config = _config_manager.load()
        setup_logging(config.logging)
        logger.info(
            "Starting %s v%s on %s:%s",
            config.app.name,
            config.app.version,
            config.app.host,
            config.app.port,
        )

        # Initialise core components
        from src.llm.client import LLMClient
        from src.mcp.manager import MCPManager
        from src.router.intent import IntentRouter
        from src.store.reference import ReferenceStore
        from src.session.manager import SessionManager
        from src.chunking.engine import ChunkingEngine
        from src.vectorstore.adapter import create_vector_store
        from src.agent.loop import AgentLoop
        from src.api.dependencies import configure as configure_deps

        llm_client = LLMClient(config.llm)
        mcp_manager = MCPManager(config.mcp_servers)
        intent_router = IntentRouter(config.intent_router)
        reference_store = ReferenceStore()
        session_manager = SessionManager(
            max_messages=config.session.max_messages,
            timeout_minutes=config.session.timeout_minutes,
        )
        chunking_engine = ChunkingEngine()
        vector_store = create_vector_store(config.vector_store, config.embedding)
        agent_loop = AgentLoop(
            llm_client=llm_client,
            mcp_manager=mcp_manager,
            reference_store=reference_store,
            max_iterations=config.app.max_agent_iterations,
        )

        # Wire up dependency injection for API endpoints
        configure_deps(
            config_manager=_config_manager,
            intent_router=intent_router,
            agent_loop=agent_loop,
            session_manager=session_manager,
            mcp_manager=mcp_manager,
            reference_store=reference_store,
            chunking_engine=chunking_engine,
            vector_store=vector_store,
        )

        # Start enabled MCP servers
        await mcp_manager.start_all()
        logger.info("MCP servers started")

        logger.info("%s is ready", config.app.name)

        yield

        # --- Shutdown ---
        logger.info("Shutting down…")
        await mcp_manager.stop_all()
        logger.info("MCP servers stopped. Goodbye.")

    application = FastAPI(
        title="AgentDesk",
        description="Agentic RAG customer support assistant platform",
        lifespan=lifespan,
    )

    # Register routers
    from src.api import (
        health_router,
        chat_router,
        documents_router,
        customers_router,
        config_router,
        mcp_router,
        stats_router,
    )

    application.include_router(health_router)
    application.include_router(chat_router)
    application.include_router(documents_router)
    application.include_router(customers_router)
    application.include_router(config_router)
    application.include_router(mcp_router)
    application.include_router(stats_router)

    return application


app = _build_app()


if __name__ == "__main__":
    import uvicorn

    config = _config_manager.load()
    uvicorn.run(
        "src.main:app",
        host=config.app.host,
        port=config.app.port,
        reload=False,
    )
