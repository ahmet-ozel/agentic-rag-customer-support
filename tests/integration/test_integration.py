"""Integration tests - end-to-end flows with mocked external dependencies.

Verifies the full pipeline through real components (ConfigManager, IntentRouter,
SessionManager, ReferenceStore, ChunkingEngine, AgentLoop) while mocking only
external services (LLM API, Qdrant, PostgreSQL / MCP servers).

Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5
"""

from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.chat import router as chat_router, _chat_stats
from src.api.documents import router as documents_router, _documents
from src.api.health import router as health_router
from src.agent.loop import AgentLoop
from src.chunking.engine import ChunkingEngine
from src.config.manager import ConfigManager
from src.config.models import (
    AppConfig,
    IntentCategory,
    IntentRouterConfig,
    LLMConfig,
    MCPServerConfig,
)
from src.llm.client import LLMClient, LLMResponse, ToolCall
from src.mcp.manager import MCPManager
from src.models.schemas import TokenUsage
from src.router.intent import IntentRouter
from src.session.manager import SessionManager
from src.store.reference import ReferenceStore
from src.vectorstore.adapter import VectorStoreAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_intent_router_config() -> IntentRouterConfig:
    """Minimal intent router config with utterances for each category."""
    return IntentRouterConfig(
        categories={
            "chitchat": IntentCategory(
                utterances=["hello", "hi", "hey", "merhaba"],
                action="direct_response",
                response="Merhaba! Size nasıl yardımcı olabilirim?",
            ),
            "customer_query": IntentCategory(
                utterances=[
                    "müşteri bilgisi",
                    "customer info",
                    "müşteri sorgula",
                    "show customer",
                    "müşteri detayları",
                ],
                action="agent_loop",
            ),
            "faq_query": IntentCategory(
                utterances=["sss", "faq", "sıkça sorulan", "nasıl yapılır"],
                action="agent_loop",
            ),
            "document_upload": IntentCategory(
                utterances=["doküman yükle", "dosya yükle", "upload document"],
                action="agent_loop",
            ),
            "document_query": IntentCategory(
                utterances=["dokümanda ara", "search document"],
                action="agent_loop",
            ),
        }
    )


def _make_mock_llm_client() -> MagicMock:
    """Create a mock LLMClient with an async chat_completion method."""
    mock = MagicMock(spec=LLMClient)
    mock.chat_completion = AsyncMock()
    return mock


def _make_mock_mcp_manager() -> MagicMock:
    """Create a mock MCPManager with async methods."""
    mock = MagicMock(spec=MCPManager)
    mock.call_tool = AsyncMock()
    mock.start_all = AsyncMock()
    mock.stop_all = AsyncMock()
    mock.get_status.return_value = {}
    mock.list_available_tools.return_value = [
        {
            "server_name": "postgres-mcp",
            "name": "query_database",
            "description": "Run a read-only SQL query",
            "inputSchema": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
            },
        },
        {
            "server_name": "qdrant-mcp",
            "name": "vector_search",
            "description": "Search the vector store",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    ]
    return mock


def _make_mock_vector_store() -> MagicMock:
    """Create a mock VectorStoreAdapter with async methods."""
    mock = MagicMock(spec=VectorStoreAdapter)
    mock.store_chunks = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.delete_document = AsyncMock()
    return mock


def _build_app() -> FastAPI:
    """Create a FastAPI app with the routers needed for integration tests."""
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(documents_router)
    return app


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset all in-memory global state between tests."""
    _chat_stats["total_conversations"] = 0
    _chat_stats["total_prompt_tokens"] = 0
    _chat_stats["total_completion_tokens"] = 0
    _chat_stats["total_tokens"] = 0
    _chat_stats["tool_calls"] = {}
    _chat_stats["total_response_time_ms"] = 0.0
    _documents.clear()
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
# 1. Chat flow integration test
# ===================================================================


class TestChatFlowIntegration:
    """User message  Intent Router  Agent Loop  LLM  MCP  Response.

    Validates: Requirements 10.1, 10.4, 10.5
    """

    def test_chat_with_tool_call_produces_full_response(self) -> None:
        """Full chat pipeline: LLM returns a tool call, then a final answer."""
        # --- Set up real components ---
        intent_router = IntentRouter(_build_intent_router_config())
        session_manager = SessionManager(max_messages=50, timeout_minutes=30)
        reference_store = ReferenceStore()

        # --- Mock external services ---
        mock_llm = _make_mock_llm_client()
        mock_mcp = _make_mock_mcp_manager()

        # First LLM call: returns a tool call
        tool_call_response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_001",
                    name="query_database",
                    arguments={"sql": "SELECT * FROM customers WHERE id = 1"},
                )
            ],
            usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        )
        # Second LLM call: returns final text
        final_response = LLMResponse(
            content="Müşteri Ahmet Yılmaz, Pro planında aktif durumda.",
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=80, completion_tokens=30, total_tokens=110),
        )
        mock_llm.chat_completion.side_effect = [tool_call_response, final_response]

        # MCP returns a customer record
        mock_mcp.call_tool.return_value = {
            "rows": [{"id": 1, "name": "Ahmet Yılmaz", "plan": "Pro", "status": "active"}]
        }

        agent_loop = AgentLoop(
            llm_client=mock_llm,
            mcp_manager=mock_mcp,
            reference_store=reference_store,
            max_iterations=10,
        )

        # Wire up DI
        deps.configure(
            intent_router=intent_router,
            agent_loop=agent_loop,
            session_manager=session_manager,
            mcp_manager=mock_mcp,
            reference_store=reference_store,
        )

        app = _build_app()
        client = TestClient(app)

        # --- Execute ---
        resp = client.post(
            "/api/v1/chat",
            json={"message": "müşteri bilgisi göster"},
        )

        # --- Verify ---
        assert resp.status_code == 200
        data = resp.json()

        # ChatResponse must have all required fields
        assert "response" in data
        assert "session_id" in data
        assert "citations" in data
        assert "tool_traces" in data
        assert "usage" in data

        # Response content should be the final LLM answer
        assert "Ahmet Yılmaz" in data["response"]

        # Tool traces should record the tool call
        assert len(data["tool_traces"]) == 1
        trace = data["tool_traces"][0]
        assert trace["tool_name"] == "query_database"
        assert trace["server_name"] == "postgres-mcp"
        assert trace["duration_ms"] >= 0

        # Token usage should be accumulated from both LLM calls
        assert data["usage"]["total_tokens"] == 180  # 70 + 110

        # Session should have been created
        assert data["session_id"]

    def test_chitchat_returns_predefined_response_without_llm(self) -> None:
        """Chitchat intent should return a canned response, no LLM call."""
        intent_router = IntentRouter(_build_intent_router_config())
        session_manager = SessionManager(max_messages=50, timeout_minutes=30)

        mock_llm = _make_mock_llm_client()
        mock_mcp = _make_mock_mcp_manager()

        agent_loop = AgentLoop(
            llm_client=mock_llm,
            mcp_manager=mock_mcp,
            reference_store=ReferenceStore(),
            max_iterations=10,
        )

        deps.configure(
            intent_router=intent_router,
            agent_loop=agent_loop,
            session_manager=session_manager,
        )

        app = _build_app()
        client = TestClient(app)

        resp = client.post("/api/v1/chat", json={"message": "merhaba"})

        assert resp.status_code == 200
        data = resp.json()
        assert "Merhaba" in data["response"]
        assert data["tool_traces"] == []
        assert data["usage"]["total_tokens"] == 0

        # LLM should NOT have been called
        mock_llm.chat_completion.assert_not_called()

    def test_session_continuity_across_messages(self) -> None:
        """Sending a second message with the same session_id preserves history."""
        intent_router = IntentRouter(_build_intent_router_config())
        session_manager = SessionManager(max_messages=50, timeout_minutes=30)

        mock_llm = _make_mock_llm_client()
        mock_mcp = _make_mock_mcp_manager()

        # Both calls return a direct final response (no tool calls)
        mock_llm.chat_completion.return_value = LLMResponse(
            content="İşte yanıtınız.",
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        agent_loop = AgentLoop(
            llm_client=mock_llm,
            mcp_manager=mock_mcp,
            reference_store=ReferenceStore(),
            max_iterations=10,
        )

        deps.configure(
            intent_router=intent_router,
            agent_loop=agent_loop,
            session_manager=session_manager,
        )

        app = _build_app()
        client = TestClient(app)

        # First message - creates a session
        resp1 = client.post(
            "/api/v1/chat",
            json={"message": "müşteri sorgula id=5"},
        )
        assert resp1.status_code == 200
        session_id = resp1.json()["session_id"]

        # Second message - reuses the session
        resp2 = client.post(
            "/api/v1/chat",
            json={"message": "müşteri detayları göster", "session_id": session_id},
        )
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == session_id

        # The second LLM call should have received conversation history
        second_call_messages = mock_llm.chat_completion.call_args_list[1][1]["messages"]
        # Should contain at least: user msg 1, assistant msg 1, user msg 2
        assert len(second_call_messages) >= 3


# ===================================================================
# 2. Document flow integration test
# ===================================================================


class TestDocumentFlowIntegration:
    """Upload  Parser  Reference Store  Chunking  Embedding  Vector Store.

    Validates: Requirements 10.2, 10.3
    """

    def test_upload_text_file_full_pipeline(self) -> None:
        """Upload a text file and verify it goes through the full pipeline."""
        reference_store = ReferenceStore()
        chunking_engine = ChunkingEngine()
        mock_vector_store = _make_mock_vector_store()

        deps.configure(
            reference_store=reference_store,
            chunking_engine=chunking_engine,
            vector_store=mock_vector_store,
        )

        app = _build_app()
        client = TestClient(app)

        # Create a text file with enough content to produce multiple chunks
        content = textwrap.dedent("""\
            AgentDesk Kullanım Kılavuzu

            AgentDesk, müşteri destek operasyonlarını otomatikleştiren bir platformdur.
            Sistem, yapay zeka destekli sohbet asistanı ile müşteri sorularını yanıtlar.

            Kurulum Adımları

            Docker Compose ile tüm servisleri başlatabilirsiniz.
            Yapılandırma dosyasını düzenleyerek LLM sağlayıcısını değiştirebilirsiniz.

            Sıkça Sorulan Sorular

            Sistem hangi LLM sağlayıcılarını destekler?
            vLLM, OpenAI, Anthropic, Google ve Ollama desteklenmektedir.
        """)

        resp = client.post(
            "/api/v1/documents",
            files={"file": ("guide.txt", content.encode("utf-8"), "text/plain")},
        )

        assert resp.status_code == 201
        data = resp.json()

        # Verify response fields
        assert data["document_id"]
        assert data["filename"] == "guide.txt"
        assert data["status"] == "completed"
        assert "chunks" in data["message"].lower() or "processed" in data["message"].lower()

        # Verify chunks were stored in the vector store
        mock_vector_store.store_chunks.assert_called_once()
        stored_chunks = mock_vector_store.store_chunks.call_args[0][0]
        assert len(stored_chunks) > 0
        # Each chunk should have a doc_id in metadata
        for chunk in stored_chunks:
            assert chunk.metadata.get("doc_id") == data["document_id"]

        # Verify the document appears in the listing
        list_resp = client.get("/api/v1/documents")
        assert list_resp.status_code == 200
        docs = list_resp.json()
        assert len(docs) == 1
        assert docs[0]["document_id"] == data["document_id"]
        assert docs[0]["status"] == "completed"

    def test_document_status_reaches_completed(self) -> None:
        """Verify the document transitions through statuses to completed."""
        reference_store = ReferenceStore()
        chunking_engine = ChunkingEngine()
        mock_vector_store = _make_mock_vector_store()

        deps.configure(
            reference_store=reference_store,
            chunking_engine=chunking_engine,
            vector_store=mock_vector_store,
        )

        app = _build_app()
        client = TestClient(app)

        resp = client.post(
            "/api/v1/documents",
            files={"file": ("test.txt", b"Some test content for chunking.", "text/plain")},
        )

        assert resp.status_code == 201
        doc_id = resp.json()["document_id"]

        # After successful upload, the in-memory store should show "completed"
        from src.api.documents import _documents as docs_store
        assert docs_store[doc_id]["status"] == "completed"

    def test_document_delete_removes_from_listing(self) -> None:
        """Uploading then deleting a document removes it from the list."""
        reference_store = ReferenceStore()
        chunking_engine = ChunkingEngine()
        mock_vector_store = _make_mock_vector_store()

        deps.configure(
            reference_store=reference_store,
            chunking_engine=chunking_engine,
            vector_store=mock_vector_store,
        )

        app = _build_app()
        client = TestClient(app)

        # Upload
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("doc.txt", b"Hello world content.", "text/plain")},
        )
        doc_id = resp.json()["document_id"]

        # Delete
        del_resp = client.delete(f"/api/v1/documents/{doc_id}")
        assert del_resp.status_code == 200

        # Listing should be empty
        list_resp = client.get("/api/v1/documents")
        assert list_resp.json() == []

        # Vector store delete should have been called
        mock_vector_store.delete_document.assert_called_once_with(doc_id)


# ===================================================================
# 3. Customer query flow integration test
# ===================================================================


class TestCustomerQueryFlowIntegration:
    """Intent  postgres-mcp  LLM  Response with tool traces.

    Validates: Requirements 10.1, 10.4
    """

    def test_customer_query_includes_tool_traces(self) -> None:
        """A customer query should route through postgres-mcp and include traces."""
        intent_router = IntentRouter(_build_intent_router_config())
        session_manager = SessionManager(max_messages=50, timeout_minutes=30)
        reference_store = ReferenceStore()

        mock_llm = _make_mock_llm_client()
        mock_mcp = _make_mock_mcp_manager()

        # LLM first calls the database tool, then gives a final answer
        tool_call_response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_100",
                    name="query_database",
                    arguments={"sql": "SELECT name, email FROM customers WHERE id = 42"},
                )
            ],
            usage=TokenUsage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
        )
        final_response = LLMResponse(
            content="Müşteri: Ayşe Demir, e-posta: ayse@example.com",
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=60, completion_tokens=25, total_tokens=85),
        )
        mock_llm.chat_completion.side_effect = [tool_call_response, final_response]

        mock_mcp.call_tool.return_value = {
            "rows": [{"name": "Ayşe Demir", "email": "ayse@example.com"}]
        }

        agent_loop = AgentLoop(
            llm_client=mock_llm,
            mcp_manager=mock_mcp,
            reference_store=reference_store,
            max_iterations=10,
        )

        deps.configure(
            intent_router=intent_router,
            agent_loop=agent_loop,
            session_manager=session_manager,
            mcp_manager=mock_mcp,
            reference_store=reference_store,
        )

        app = _build_app()
        client = TestClient(app)

        resp = client.post(
            "/api/v1/chat",
            json={"message": "müşteri bilgisi id=42"},
        )

        assert resp.status_code == 200
        data = resp.json()

        # Response should contain the customer info
        assert "Ayşe Demir" in data["response"]

        # Tool traces must be present
        assert len(data["tool_traces"]) >= 1
        trace = data["tool_traces"][0]
        assert trace["tool_name"] == "query_database"
        assert trace["server_name"] == "postgres-mcp"
        assert "arguments" in trace
        assert "result_summary" in trace
        assert trace["duration_ms"] >= 0

        # MCP call_tool should have been invoked with the right server
        mock_mcp.call_tool.assert_called_once_with(
            server_name="postgres-mcp",
            tool_name="query_database",
            arguments={"sql": "SELECT name, email FROM customers WHERE id = 42"},
        )

    def test_multiple_tool_calls_in_single_loop(self) -> None:
        """LLM makes two sequential tool calls before producing a final answer."""
        intent_router = IntentRouter(_build_intent_router_config())
        session_manager = SessionManager(max_messages=50, timeout_minutes=30)
        reference_store = ReferenceStore()

        mock_llm = _make_mock_llm_client()
        mock_mcp = _make_mock_mcp_manager()

        # Iteration 1: LLM calls query_database
        first_tool_response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_a",
                    name="query_database",
                    arguments={"sql": "SELECT * FROM customers WHERE id = 1"},
                )
            ],
            usage=TokenUsage(prompt_tokens=30, completion_tokens=10, total_tokens=40),
        )
        # Iteration 2: LLM calls vector_search
        second_tool_response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_b",
                    name="vector_search",
                    arguments={"query": "fatura bilgisi"},
                )
            ],
            usage=TokenUsage(prompt_tokens=50, completion_tokens=15, total_tokens=65),
        )
        # Iteration 3: final answer
        final_response = LLMResponse(
            content="Müşteri Ahmet'in son faturası 150 TL.",
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=70, completion_tokens=20, total_tokens=90),
        )
        mock_llm.chat_completion.side_effect = [
            first_tool_response,
            second_tool_response,
            final_response,
        ]

        mock_mcp.call_tool.side_effect = [
            {"rows": [{"id": 1, "name": "Ahmet"}]},
            {"results": [{"text": "Son fatura: 150 TL"}]},
        ]

        agent_loop = AgentLoop(
            llm_client=mock_llm,
            mcp_manager=mock_mcp,
            reference_store=reference_store,
            max_iterations=10,
        )

        deps.configure(
            intent_router=intent_router,
            agent_loop=agent_loop,
            session_manager=session_manager,
            mcp_manager=mock_mcp,
            reference_store=reference_store,
        )

        app = _build_app()
        client = TestClient(app)

        resp = client.post(
            "/api/v1/chat",
            json={"message": "müşteri bilgisi ve fatura detayları"},
        )

        assert resp.status_code == 200
        data = resp.json()

        # Should have two tool traces
        assert len(data["tool_traces"]) == 2
        assert data["tool_traces"][0]["tool_name"] == "query_database"
        assert data["tool_traces"][1]["tool_name"] == "vector_search"

        # Total tokens accumulated across all 3 LLM calls
        assert data["usage"]["total_tokens"] == 195  # 40 + 65 + 90

        # LLM was called 3 times
        assert mock_llm.chat_completion.call_count == 3
