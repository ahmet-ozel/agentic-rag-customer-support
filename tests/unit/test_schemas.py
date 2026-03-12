"""Unit tests for API schema models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentUploadResponse,
    ErrorResponse,
    MCPStatusResponse,
    ServerStatus,
    StatsResponse,
    SystemInfoResponse,
    TokenUsage,
    ToolTrace,
)


class TestChatRequest:
    def test_minimal_request(self):
        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.session_id is None
        assert req.customer_id is None

    def test_full_request(self):
        req = ChatRequest(message="Hi", session_id="s1", customer_id=42)
        assert req.session_id == "s1"
        assert req.customer_id == 42

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest()


class TestChatResponse:
    def test_full_response(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp = ChatResponse(
            response="Answer",
            session_id="s1",
            citations=[Citation(source="doc.pdf", page=1, text="snippet", score=0.9)],
            tool_traces=[
                ToolTrace(
                    tool_name="search",
                    server_name="qdrant-mcp",
                    arguments={"query": "test"},
                    result_summary="found 3 results",
                    duration_ms=120.5,
                )
            ],
            usage=usage,
        )
        assert resp.response == "Answer"
        assert len(resp.citations) == 1
        assert len(resp.tool_traces) == 1
        assert resp.usage.total_tokens == 15

    def test_defaults_for_lists(self):
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        resp = ChatResponse(response="Hi", session_id="s1", usage=usage)
        assert resp.citations == []
        assert resp.tool_traces == []


class TestTokenUsage:
    def test_with_estimated_cost(self):
        usage = TokenUsage(
            prompt_tokens=100, completion_tokens=50, total_tokens=150, estimated_cost=0.003
        )
        assert usage.estimated_cost == 0.003

    def test_estimated_cost_defaults_none(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert usage.estimated_cost is None


class TestDocumentUploadResponse:
    def test_creation(self):
        resp = DocumentUploadResponse(
            document_id="doc_123", filename="report.pdf", status="completed", message="OK"
        )
        assert resp.document_id == "doc_123"
        assert resp.status == "completed"


class TestMCPStatusResponse:
    def test_with_servers(self):
        resp = MCPStatusResponse(
            servers={
                "postgres-mcp": ServerStatus(
                    name="postgres-mcp", status="running", transport="stdio", uptime_seconds=120.0
                ),
                "qdrant-mcp": ServerStatus(
                    name="qdrant-mcp", status="error", transport="sse", error_message="connection refused"
                ),
            }
        )
        assert resp.servers["postgres-mcp"].status == "running"
        assert resp.servers["qdrant-mcp"].error_message == "connection refused"


class TestStatsResponse:
    def test_creation(self):
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        stats = StatsResponse(
            total_conversations=42,
            total_tokens=usage,
            tool_call_distribution={"search": 10, "query": 5},
            average_response_time_ms=250.0,
        )
        assert stats.total_conversations == 42
        assert stats.tool_call_distribution["search"] == 10


class TestSystemInfoResponse:
    def test_creation(self):
        info = SystemInfoResponse(
            version="1.0.0",
            active_llm_provider="openai",
            active_llm_model="gpt-4",
            active_vector_store="qdrant",
            enabled_mcp_servers=["postgres-mcp", "qdrant-mcp"],
        )
        assert info.version == "1.0.0"
        assert len(info.enabled_mcp_servers) == 2


class TestErrorResponse:
    def test_creation(self):
        err = ErrorResponse(
            error="config_validation_error",
            message="Invalid field",
            details={"field": "llm.model"},
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )
        assert err.error == "config_validation_error"
        assert err.details["field"] == "llm.model"

    def test_details_defaults_none(self):
        err = ErrorResponse(
            error="not_found",
            message="Resource not found",
            timestamp=datetime.now(),
        )
        assert err.details is None
