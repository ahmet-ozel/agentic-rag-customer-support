"""Unit tests for AgentLoop."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.loop import AgentLoop, AgentResponse
from src.llm.client import LLMClient, LLMResponse, ToolCall
from src.mcp.manager import MCPManager
from src.models.schemas import Citation, TokenUsage, ToolTrace
from src.router.intent import IntentResult
from src.session.manager import Session
from src.store.reference import ReferenceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_response(
    content: str | None = "Hello!",
    tool_calls: list[ToolCall] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _make_tool_call(
    tc_id: str = "call_1",
    name: str = "search",
    arguments: dict | None = None,
) -> ToolCall:
    return ToolCall(id=tc_id, name=name, arguments=arguments or {})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm_client() -> LLMClient:
    client = MagicMock(spec=LLMClient)
    client.chat_completion = AsyncMock()
    return client


@pytest.fixture()
def mock_mcp_manager() -> MCPManager:
    manager = MagicMock(spec=MCPManager)
    manager.call_tool = AsyncMock()
    manager.list_available_tools = MagicMock(return_value=[
        {"name": "db_query", "description": "Query database", "parameters": {}, "server_name": "postgres-mcp"},
        {"name": "vector_search", "description": "Search vectors", "parameters": {}, "server_name": "qdrant-mcp"},
    ])
    return manager


@pytest.fixture()
def reference_store() -> ReferenceStore:
    return ReferenceStore(ttl_minutes=30)


@pytest.fixture()
def session() -> Session:
    return Session(session_id="test-session-123")


@pytest.fixture()
def intent() -> IntentResult:
    return IntentResult(intent="customer_query", confidence=0.95)


@pytest.fixture()
def agent_loop(
    mock_llm_client: LLMClient,
    mock_mcp_manager: MCPManager,
    reference_store: ReferenceStore,
) -> AgentLoop:
    return AgentLoop(
        llm_client=mock_llm_client,
        mcp_manager=mock_mcp_manager,
        reference_store=reference_store,
        max_iterations=5,
        reference_threshold=4000,
    )


# ---------------------------------------------------------------------------
# Test: Normal flow - LLM returns content immediately
# ---------------------------------------------------------------------------


class TestNormalFlow:
    @pytest.mark.asyncio
    async def test_llm_returns_content_immediately(
        self, agent_loop: AgentLoop, mock_llm_client, session, intent
    ) -> None:
        mock_llm_client.chat_completion.return_value = _make_llm_response(
            content="Here is your answer."
        )

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Hello"}],
            session=session,
            intent=intent,
        )

        assert isinstance(result, AgentResponse)
        assert result.content == "Here is your answer."
        assert result.tool_traces == []
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_string(
        self, agent_loop: AgentLoop, mock_llm_client, session, intent
    ) -> None:
        mock_llm_client.chat_completion.return_value = _make_llm_response(
            content=None
        )

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Hi"}],
            session=session,
            intent=intent,
        )

        assert result.content == ""


# ---------------------------------------------------------------------------
# Test: Tool call flow - LLM calls tool, then returns content
# ---------------------------------------------------------------------------


class TestToolCallFlow:
    @pytest.mark.asyncio
    async def test_single_tool_call_then_response(
        self, agent_loop: AgentLoop, mock_llm_client, mock_mcp_manager, session, intent
    ) -> None:
        # First call: LLM requests a tool call
        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "db_query", {"sql": "SELECT 1"})],
        )
        # Second call: LLM returns final content
        final_response = _make_llm_response(content="Query result: 1")

        mock_llm_client.chat_completion.side_effect = [tool_call_response, final_response]
        mock_mcp_manager.call_tool.return_value = {"rows": [{"result": 1}]}

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Run query"}],
            session=session,
            intent=intent,
        )

        assert result.content == "Query result: 1"
        assert len(result.tool_traces) == 1
        assert result.tool_traces[0].tool_name == "db_query"
        assert result.tool_traces[0].server_name == "postgres-mcp"
        assert result.tool_traces[0].arguments == {"sql": "SELECT 1"}
        mock_mcp_manager.call_tool.assert_awaited_once_with(
            server_name="postgres-mcp", tool_name="db_query", arguments={"sql": "SELECT 1"}
        )

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_response(
        self, agent_loop: AgentLoop, mock_llm_client, mock_mcp_manager, session, intent
    ) -> None:
        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[
                _make_tool_call("call_1", "db_query", {"sql": "SELECT 1"}),
                _make_tool_call("call_2", "vector_search", {"query": "help"}),
            ],
        )
        final_response = _make_llm_response(content="Combined result")

        mock_llm_client.chat_completion.side_effect = [tool_call_response, final_response]
        mock_mcp_manager.call_tool.return_value = {"data": "ok"}

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Search"}],
            session=session,
            intent=intent,
        )

        assert result.content == "Combined result"
        assert len(result.tool_traces) == 2
        assert result.tool_traces[0].tool_name == "db_query"
        assert result.tool_traces[1].tool_name == "vector_search"

    @pytest.mark.asyncio
    async def test_tool_call_error_is_handled(
        self, agent_loop: AgentLoop, mock_llm_client, mock_mcp_manager, session, intent
    ) -> None:
        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "db_query", {"sql": "BAD"})],
        )
        final_response = _make_llm_response(content="Error occurred")

        mock_llm_client.chat_completion.side_effect = [tool_call_response, final_response]
        mock_mcp_manager.call_tool.side_effect = Exception("Connection refused")

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Query"}],
            session=session,
            intent=intent,
        )

        assert result.content == "Error occurred"
        assert len(result.tool_traces) == 1
        assert "Connection refused" in result.tool_traces[0].result_summary


# ---------------------------------------------------------------------------
# Test: Max iteration limit
# ---------------------------------------------------------------------------


class TestMaxIterationLimit:
    @pytest.mark.asyncio
    async def test_exceeds_max_iterations(
        self, agent_loop: AgentLoop, mock_llm_client, mock_mcp_manager, session, intent
    ) -> None:
        # LLM always returns tool calls, never a final response
        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call("call_x", "db_query", {"sql": "SELECT 1"})],
        )
        mock_llm_client.chat_completion.return_value = tool_call_response
        mock_mcp_manager.call_tool.return_value = {"result": "data"}

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Loop forever"}],
            session=session,
            intent=intent,
        )

        assert "Maksimum iterasyon" in result.content
        assert len(result.tool_traces) == 5  # max_iterations=5
        assert mock_llm_client.chat_completion.await_count == 5

    @pytest.mark.asyncio
    async def test_max_iterations_one(
        self, mock_llm_client, mock_mcp_manager, reference_store, session, intent
    ) -> None:
        loop = AgentLoop(
            llm_client=mock_llm_client,
            mcp_manager=mock_mcp_manager,
            reference_store=reference_store,
            max_iterations=1,
        )
        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "db_query", {})],
        )
        mock_llm_client.chat_completion.return_value = tool_call_response
        mock_mcp_manager.call_tool.return_value = {"ok": True}

        result = await loop.run(
            messages=[{"role": "user", "content": "test"}],
            session=session,
            intent=intent,
        )

        assert "Maksimum iterasyon" in result.content
        assert mock_llm_client.chat_completion.await_count == 1


# ---------------------------------------------------------------------------
# Test: Reference store integration (large tool results)
# ---------------------------------------------------------------------------


class TestReferenceStoreIntegration:
    @pytest.mark.asyncio
    async def test_large_result_stored_in_reference_store(
        self, agent_loop: AgentLoop, mock_llm_client, mock_mcp_manager, reference_store, session, intent
    ) -> None:
        large_data = {"content": "x" * 5000}  # Exceeds 4000 char threshold

        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "db_query", {"sql": "SELECT *"})],
        )
        final_response = _make_llm_response(content="Here is the summary")

        mock_llm_client.chat_completion.side_effect = [tool_call_response, final_response]
        mock_mcp_manager.call_tool.return_value = large_data

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Get all data"}],
            session=session,
            intent=intent,
        )

        assert result.content == "Here is the summary"
        # Verify reference store has an entry
        assert len(reference_store._entries) == 1
        ref_id = list(reference_store._entries.keys())[0]
        assert ref_id.startswith("ref_")
        # Verify stored data matches
        stored = reference_store.retrieve(ref_id)
        assert stored == json.dumps(large_data)

    @pytest.mark.asyncio
    async def test_small_result_not_stored_in_reference_store(
        self, agent_loop: AgentLoop, mock_llm_client, mock_mcp_manager, reference_store, session, intent
    ) -> None:
        small_data = {"result": "ok"}

        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "db_query", {})],
        )
        final_response = _make_llm_response(content="Done")

        mock_llm_client.chat_completion.side_effect = [tool_call_response, final_response]
        mock_mcp_manager.call_tool.return_value = small_data

        await agent_loop.run(
            messages=[{"role": "user", "content": "Quick query"}],
            session=session,
            intent=intent,
        )

        assert len(reference_store._entries) == 0


# ---------------------------------------------------------------------------
# Test: Token usage accumulation
# ---------------------------------------------------------------------------


class TestTokenUsageAccumulation:
    @pytest.mark.asyncio
    async def test_usage_accumulated_across_iterations(
        self, agent_loop: AgentLoop, mock_llm_client, mock_mcp_manager, session, intent
    ) -> None:
        # First call: tool call with 10 prompt + 5 completion
        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "db_query", {})],
            prompt_tokens=20,
            completion_tokens=10,
        )
        # Second call: final response with 30 prompt + 15 completion
        final_response = _make_llm_response(
            content="Final answer",
            prompt_tokens=30,
            completion_tokens=15,
        )

        mock_llm_client.chat_completion.side_effect = [tool_call_response, final_response]
        mock_mcp_manager.call_tool.return_value = {"data": "result"}

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Query"}],
            session=session,
            intent=intent,
        )

        assert result.usage.prompt_tokens == 50  # 20 + 30
        assert result.usage.completion_tokens == 25  # 10 + 15
        assert result.usage.total_tokens == 75  # 50 + 25

    @pytest.mark.asyncio
    async def test_single_call_usage(
        self, agent_loop: AgentLoop, mock_llm_client, session, intent
    ) -> None:
        mock_llm_client.chat_completion.return_value = _make_llm_response(
            content="Direct answer",
            prompt_tokens=100,
            completion_tokens=50,
        )

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Hi"}],
            session=session,
            intent=intent,
        )

        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150


# ---------------------------------------------------------------------------
# Test: Tool trace collection
# ---------------------------------------------------------------------------


class TestToolTraceCollection:
    @pytest.mark.asyncio
    async def test_traces_contain_required_fields(
        self, agent_loop: AgentLoop, mock_llm_client, mock_mcp_manager, session, intent
    ) -> None:
        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "db_query", {"sql": "SELECT 1"})],
        )
        final_response = _make_llm_response(content="Done")

        mock_llm_client.chat_completion.side_effect = [tool_call_response, final_response]
        mock_mcp_manager.call_tool.return_value = {"rows": [1, 2, 3]}

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Query"}],
            session=session,
            intent=intent,
        )

        assert len(result.tool_traces) == 1
        trace = result.tool_traces[0]
        assert trace.tool_name == "db_query"
        assert trace.server_name == "postgres-mcp"
        assert trace.arguments == {"sql": "SELECT 1"}
        assert isinstance(trace.result_summary, str)
        assert len(trace.result_summary) <= 200
        assert isinstance(trace.duration_ms, float)
        assert trace.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_trace_result_summary_truncated(
        self, agent_loop: AgentLoop, mock_llm_client, mock_mcp_manager, session, intent
    ) -> None:
        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "db_query", {})],
        )
        final_response = _make_llm_response(content="Done")

        mock_llm_client.chat_completion.side_effect = [tool_call_response, final_response]
        # Return a very long result
        mock_mcp_manager.call_tool.return_value = {"data": "a" * 500}

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Query"}],
            session=session,
            intent=intent,
        )

        assert len(result.tool_traces[0].result_summary) <= 200

    @pytest.mark.asyncio
    async def test_unknown_tool_gets_unknown_server(
        self, agent_loop: AgentLoop, mock_llm_client, mock_mcp_manager, session, intent
    ) -> None:
        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "unknown_tool", {})],
        )
        final_response = _make_llm_response(content="Done")

        mock_llm_client.chat_completion.side_effect = [tool_call_response, final_response]
        mock_mcp_manager.call_tool.return_value = {"ok": True}

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Test"}],
            session=session,
            intent=intent,
        )

        assert result.tool_traces[0].server_name == "unknown"


# ---------------------------------------------------------------------------
# Test: Citation extraction
# ---------------------------------------------------------------------------


class TestCitationExtraction:
    @pytest.mark.asyncio
    async def test_citations_extracted_from_content(
        self, agent_loop: AgentLoop, mock_llm_client, session, intent
    ) -> None:
        mock_llm_client.chat_completion.return_value = _make_llm_response(
            content="Answer based on [source: manual.pdf, page: 5] and [source: faq.md]"
        )

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Help"}],
            session=session,
            intent=intent,
        )

        assert len(result.citations) == 2
        assert result.citations[0].source == "manual.pdf"
        assert result.citations[0].page == 5
        assert result.citations[1].source == "faq.md"
        assert result.citations[1].page is None

    @pytest.mark.asyncio
    async def test_citations_from_tool_results(
        self, agent_loop: AgentLoop, mock_llm_client, mock_mcp_manager, session, intent
    ) -> None:
        tool_call_response = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call("call_1", "vector_search", {"query": "help"})],
        )
        final_response = _make_llm_response(content="Here is the answer")

        mock_llm_client.chat_completion.side_effect = [tool_call_response, final_response]
        mock_mcp_manager.call_tool.return_value = {
            "source": "docs/guide.pdf",
            "page": 3,
            "text": "Relevant section text",
            "score": 0.92,
        }

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Search"}],
            session=session,
            intent=intent,
        )

        assert any(c.source == "docs/guide.pdf" for c in result.citations)
        assert any(c.page == 3 for c in result.citations)

    @pytest.mark.asyncio
    async def test_no_citations_when_none_present(
        self, agent_loop: AgentLoop, mock_llm_client, session, intent
    ) -> None:
        mock_llm_client.chat_completion.return_value = _make_llm_response(
            content="Simple answer with no sources"
        )

        result = await agent_loop.run(
            messages=[{"role": "user", "content": "Hi"}],
            session=session,
            intent=intent,
        )

        assert result.citations == []
