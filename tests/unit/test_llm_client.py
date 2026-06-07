"""Unit tests for LLMClient."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.config.models import LLMConfig, LLMProviderConfig, TieredLLMConfig
from src.llm.client import LLMClient, LLMClientError, LLMResponse, ToolCall
from src.models.schemas import TokenUsage


# ---------------------------------------------------------------------------
# Helpers - build fake OpenAI SDK response objects
# ---------------------------------------------------------------------------


def _make_usage(prompt: int = 10, completion: int = 5):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)


def _make_tool_call(tc_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _make_response(
    content: str | None = "Hello!",
    tool_calls: list | None = None,
    usage=None,
):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        choices=[choice],
        usage=usage or _make_usage(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_config() -> LLMConfig:
    return LLMConfig(
        default_provider="openai",
        providers={
            "openai": LLMProviderConfig(
                base_url="http://localhost:8080/v1",
                model="gpt-4",
                api_key="test-key",
            ),
        },
    )


@pytest.fixture()
def tiered_config() -> LLMConfig:
    return LLMConfig(
        default_provider="openai",
        providers={
            "openai": LLMProviderConfig(
                base_url="http://localhost:8080/v1",
                model="gpt-4",
                api_key="gen-key",
            ),
            "routing": LLMProviderConfig(
                base_url="http://localhost:8081/v1",
                model="gpt-3.5-turbo",
                api_key="route-key",
            ),
        },
        tiered=TieredLLMConfig(
            enabled=True,
            routing_provider="routing",
            routing_model="gpt-3.5-turbo",
        ),
    )


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------


class TestClientInit:
    def test_init_with_basic_config(self, base_config: LLMConfig) -> None:
        client = LLMClient(base_config)
        assert client._client is not None
        assert client._routing_client is None

    def test_init_with_tiered_config(self, tiered_config: LLMConfig) -> None:
        client = LLMClient(tiered_config)
        assert client._client is not None
        assert client._routing_client is not None

    @pytest.mark.parametrize(
        "provider",
        ["vllm", "openai", "anthropic", "google", "ollama"],
    )
    def test_init_all_providers(self, provider: str) -> None:
        """All providers should initialise without error."""
        cfg = LLMConfig(
            default_provider=provider,
            providers={
                provider: LLMProviderConfig(
                    base_url="http://localhost/v1",
                    model="some-model",
                ),
            },
        )
        client = LLMClient(cfg)
        assert client._client is not None

    def test_init_tiered_disabled_no_routing_client(self) -> None:
        cfg = LLMConfig(
            default_provider="openai",
            providers={
                "openai": LLMProviderConfig(
                    base_url="http://localhost/v1",
                    model="gpt-4",
                ),
            },
            tiered=TieredLLMConfig(enabled=False, routing_provider="openai", routing_model="m"),
        )
        client = LLMClient(cfg)
        assert client._routing_client is None


# ---------------------------------------------------------------------------
# Tiered model selection
# ---------------------------------------------------------------------------


class TestTieredModelSelection:
    def test_generation_tier_uses_main_model(self, tiered_config: LLMConfig) -> None:
        client = LLMClient(tiered_config)
        selected_client, model = client._select_client_and_model("generation")
        assert selected_client is client._client
        assert model == "gpt-4"

    def test_routing_tier_uses_routing_model(self, tiered_config: LLMConfig) -> None:
        client = LLMClient(tiered_config)
        selected_client, model = client._select_client_and_model("routing")
        assert selected_client is client._routing_client
        assert model == "gpt-3.5-turbo"

    def test_routing_tier_falls_back_when_tiered_disabled(self, base_config: LLMConfig) -> None:
        client = LLMClient(base_config)
        selected_client, model = client._select_client_and_model("routing")
        assert selected_client is client._client
        assert model == "gpt-4"


# ---------------------------------------------------------------------------
# Token usage extraction
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_usage_extracted_correctly(self) -> None:
        resp = _make_response(usage=_make_usage(prompt=20, completion=15))
        result = LLMClient._parse_response(resp)
        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 15
        assert result.usage.total_tokens == 35

    def test_usage_defaults_to_zero_when_missing(self) -> None:
        resp = _make_response(usage=None)
        # Patch usage to None
        resp.usage = None
        result = LLMClient._parse_response(resp)
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0

    def test_total_equals_prompt_plus_completion(self) -> None:
        resp = _make_response(usage=_make_usage(prompt=100, completion=50))
        result = LLMClient._parse_response(resp)
        assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------


class TestToolCallParsing:
    def test_no_tool_calls(self) -> None:
        resp = _make_response(content="Just text", tool_calls=None)
        result = LLMClient._parse_response(resp)
        assert result.content == "Just text"
        assert result.tool_calls is None

    def test_single_tool_call(self) -> None:
        tc = _make_tool_call("call_1", "get_weather", {"city": "Istanbul"})
        resp = _make_response(content=None, tool_calls=[tc])
        result = LLMClient._parse_response(resp)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Istanbul"}

    def test_multiple_tool_calls(self) -> None:
        tc1 = _make_tool_call("call_1", "search", {"q": "hello"})
        tc2 = _make_tool_call("call_2", "lookup", {"id": 42})
        resp = _make_response(content=None, tool_calls=[tc1, tc2])
        result = LLMClient._parse_response(resp)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[1].name == "lookup"
        assert result.tool_calls[1].arguments == {"id": 42}

    def test_tool_call_with_dict_arguments(self) -> None:
        """If arguments are already a dict (not a JSON string), handle gracefully."""
        tc = SimpleNamespace(
            id="call_x",
            function=SimpleNamespace(name="fn", arguments={"a": 1}),
        )
        resp = _make_response(content=None, tool_calls=[tc])
        result = LLMClient._parse_response(resp)
        assert result.tool_calls is not None
        assert result.tool_calls[0].arguments == {"a": 1}


# ---------------------------------------------------------------------------
# chat_completion - mocked end-to-end
# ---------------------------------------------------------------------------


class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_basic_completion(self, base_config: LLMConfig) -> None:
        client = LLMClient(base_config)
        fake_resp = _make_response(content="Hi there!")
        with patch.object(client._client.chat.completions, "create", new_callable=AsyncMock, return_value=fake_resp):
            result = await client.chat_completion(messages=[{"role": "user", "content": "Hello"}])
        assert isinstance(result, LLMResponse)
        assert result.content == "Hi there!"
        assert result.tool_calls is None

    @pytest.mark.asyncio
    async def test_completion_with_tools(self, base_config: LLMConfig) -> None:
        client = LLMClient(base_config)
        tc = _make_tool_call("c1", "db_query", {"sql": "SELECT 1"})
        fake_resp = _make_response(content=None, tool_calls=[tc])
        with patch.object(client._client.chat.completions, "create", new_callable=AsyncMock, return_value=fake_resp):
            result = await client.chat_completion(
                messages=[{"role": "user", "content": "query"}],
                tools=[{"type": "function", "function": {"name": "db_query"}}],
            )
        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "db_query"

    @pytest.mark.asyncio
    async def test_routing_tier_uses_routing_client(self, tiered_config: LLMConfig) -> None:
        client = LLMClient(tiered_config)
        fake_resp = _make_response(content="routed")
        assert client._routing_client is not None
        with patch.object(
            client._routing_client.chat.completions, "create", new_callable=AsyncMock, return_value=fake_resp
        ) as mock_create:
            result = await client.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                model_tier="routing",
            )
        mock_create.assert_awaited_once()
        assert result.content == "routed"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_timeout_raises_descriptive_error(self, base_config: LLMConfig) -> None:
        from openai import APITimeoutError

        client = LLMClient(base_config)
        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=APITimeoutError(request=None),
        ):
            with pytest.raises(LLMClientError, match="timed out"):
                await client.chat_completion(messages=[{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_connection_error_raises_descriptive_error(self, base_config: LLMConfig) -> None:
        from openai import APIConnectionError

        client = LLMClient(base_config)
        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=APIConnectionError(request=None),
        ):
            with pytest.raises(LLMClientError, match="Failed to connect"):
                await client.chat_completion(messages=[{"role": "user", "content": "hi"}])

