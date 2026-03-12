"""API routers for AgentDesk RAG Platform."""

from src.api.health import router as health_router
from src.api.chat import router as chat_router
from src.api.documents import router as documents_router
from src.api.customers import router as customers_router
from src.api.config_api import router as config_router
from src.api.mcp_api import router as mcp_router
from src.api.stats import router as stats_router
from src.api.dependencies import configure as configure_dependencies

__all__ = [
    "health_router",
    "chat_router",
    "documents_router",
    "customers_router",
    "config_router",
    "mcp_router",
    "stats_router",
    "configure_dependencies",
]
