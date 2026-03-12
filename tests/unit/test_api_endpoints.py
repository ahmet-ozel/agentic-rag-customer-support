"""Unit tests for REST API endpoints.

Uses FastAPI TestClient (backed by httpx) to test all endpoint routers.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.health import router as health_router
from src.api.chat import router as chat_router, _chat_stats
from src.api.documents import router as documents_router, _documents
from src.api.customers import router as customers_router
from src.api.config_api import router as config_router
from src.api.mcp_api import router as mcp_router
from src.api.stats import router as stats_router
from src.config.manager import ConfigManager
from src.config.models import (
    AppConfig,
    AppSettings,
    LLMConfig,
    MCPServerConfig,
    VectorStoreConfig,
)
from src.models.schemas import ServerStatus, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    """Create a FastAPI app with all routers registered."""
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    app.include_router(customers_router)
    app.include_router(config_router)
    app.include_router(mcp_router)
    app.include_router(stats_router)
    return app


@pytest.fixture(autouse=True)
def _reset_global_state():
    """Reset in-memory state between tests."""
    _chat_stats["total_conversations"] = 0
    _chat_stats["total_prompt_tokens"] = 0
    _chat_stats["total_completion_tokens"] = 0
    _chat_stats["total_tokens"] = 0
    _chat_stats["tool_calls"] = {}
    _chat_stats["total_response_time_ms"] = 0.0
    _documents.clear()
    # Reset dependency singletons
    deps._config_manager = None
    deps._intent_router = None
    deps._agent_loop = None
    deps._session_manager = None
    deps._mcp_manager = None
    deps._reference_store = None
    deps._chunking_engine = None
    deps._vector_store = None
    yield


# ===================================================================
# Health & Info
# ===================================================================


class TestHealthEndpoint:
    def test_health_returns_ok(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestInfoEndpoint:
    def test_info_returns_system_info(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
                app:
                  name: TestApp
                  version: "0.1.0"
                llm:
                  default_provider: openai
                  providers:
                    openai:
                      base_url: http://localhost/v1
                      model: gpt-4
                vector_store:
                  provider: qdrant
                mcp_servers:
                  postgres-mcp:
                    enabled: true
                    command: pg
                  qdrant-mcp:
                    enabled: false
                    command: qd
            """),
            encoding="utf-8",
        )
        mgr = ConfigManager()
        mgr.load(str(cfg_file))
        deps._config_manager = mgr

        app = _make_app()
        client = TestClient(app)
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "0.1.0"
        assert data["active_llm_provider"] == "openai"
        assert data["active_llm_model"] == "gpt-4"
        assert data["active_vector_store"] == "qdrant"
        assert "postgres-mcp" in data["enabled_mcp_servers"]
        assert "qdrant-mcp" not in data["enabled_mcp_servers"]


# ===================================================================
# Documents
# ===================================================================


class TestDocumentEndpoints:
    def test_upload_document(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("test.txt", b"Hello world content", "text/plain")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "test.txt"
        assert data["status"] == "completed"
        assert data["document_id"]

    def test_list_documents_empty(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_documents_after_upload(self) -> None:
        app = _make_app()
        client = TestClient(app)
        client.post(
            "/api/v1/documents",
            files={"file": ("a.txt", b"content", "text/plain")},
        )
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 200
        docs = resp.json()
        assert len(docs) == 1
        assert docs[0]["filename"] == "a.txt"

    def test_delete_document(self) -> None:
        app = _make_app()
        client = TestClient(app)
        upload = client.post(
            "/api/v1/documents",
            files={"file": ("b.txt", b"data", "text/plain")},
        )
        doc_id = upload.json()["document_id"]

        resp = client.delete(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 200

        listing = client.get("/api/v1/documents")
        assert len(listing.json()) == 0

    def test_delete_nonexistent_document(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.delete("/api/v1/documents/nonexistent-id")
        assert resp.status_code == 404

    def test_document_status_completed_after_upload(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("c.txt", b"some text", "text/plain")},
        )
        doc_id = resp.json()["document_id"]
        docs = client.get("/api/v1/documents").json()
        doc = next(d for d in docs if d["document_id"] == doc_id)
        assert doc["status"] == "completed"


# ===================================================================
# Customers
# ===================================================================


class TestCustomerEndpoints:
    def test_customers_placeholder(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/customers")
        assert resp.status_code == 200
        data = resp.json()
        assert "customers" in data

    def test_customers_with_query(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/customers", params={"q": "test"})
        assert resp.status_code == 200
        assert resp.json()["query"] == "test"


# ===================================================================
# Config
# ===================================================================


class TestConfigEndpoints:
    def test_get_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("app:\n  name: CfgTest\n", encoding="utf-8")
        mgr = ConfigManager()
        mgr.load(str(cfg_file))
        deps._config_manager = mgr

        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/config")
        assert resp.status_code == 200
        assert resp.json()["app"]["name"] == "CfgTest"

    def test_put_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("app:\n  name: Before\n", encoding="utf-8")
        mgr = ConfigManager()
        mgr.load(str(cfg_file))
        deps._config_manager = mgr

        app = _make_app()
        client = TestClient(app)
        resp = client.put(
            "/api/v1/config",
            json={"updates": {"app": {"name": "After"}}},
        )
        assert resp.status_code == 200
        assert resp.json()["config"]["app"]["name"] == "After"

        # Verify GET reflects the change
        get_resp = client.get("/api/v1/config")
        assert get_resp.json()["app"]["name"] == "After"


# ===================================================================
# MCP Status
# ===================================================================


class TestMCPStatusEndpoint:
    def test_mcp_status(self) -> None:
        mock_manager = MagicMock()
        mock_manager.get_status.return_value = {
            "postgres-mcp": ServerStatus(
                name="postgres-mcp",
                status="running",
                transport="stdio",
                uptime_seconds=120.0,
            ),
        }
        deps._mcp_manager = mock_manager

        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/mcp/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "postgres-mcp" in data["servers"]
        assert data["servers"]["postgres-mcp"]["status"] == "running"


# ===================================================================
# Stats
# ===================================================================


class TestStatsEndpoint:
    def test_stats_initial(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_conversations"] == 0
        assert data["total_tokens"]["total_tokens"] == 0
        assert data["average_response_time_ms"] == 0.0

    def test_stats_after_activity(self) -> None:
        _chat_stats["total_conversations"] = 5
        _chat_stats["total_prompt_tokens"] = 100
        _chat_stats["total_completion_tokens"] = 50
        _chat_stats["total_tokens"] = 150
        _chat_stats["tool_calls"] = {"search": 3, "query": 2}
        _chat_stats["total_response_time_ms"] = 2500.0

        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_conversations"] == 5
        assert data["total_tokens"]["prompt_tokens"] == 100
        assert data["total_tokens"]["completion_tokens"] == 50
        assert data["total_tokens"]["total_tokens"] == 150
        assert data["tool_call_distribution"] == {"search": 3, "query": 2}
        assert data["average_response_time_ms"] == 500.0
