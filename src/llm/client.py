"""LLM Client - unified interface for all providers via OpenAI SDK.

All providers (vLLM, Ollama, OpenAI, Anthropic, Google) expose
OpenAI-compatible endpoints, so we use a single ``openai.AsyncOpenAI``
client with different ``base_url`` / ``api_key`` values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError

from src.config.models import LLMConfig
from src.models.schemas import TokenUsage


# ---------------------------------------------------------------------------
# Data classes for structured responses
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool/function call returned by the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Structured response from an LLM chat completion."""

    content: str | None
    tool_calls: list[ToolCall] | None
    usage: TokenUsage


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClientError(Exception):
    """Raised when the LLM client encounters an error."""


class LLMClient:
    """Unified LLM client that wraps ``openai.AsyncOpenAI``.

    Parameters
    ----------
    config:
        An ``LLMConfig`` instance (from config.yaml).
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

        active = config.get_active()

        # Main (generation) client
        self._client = AsyncOpenAI(
            base_url=active.base_url or None,
            api_key=active.api_key or "not-needed",
            timeout=active.timeout,
        )
        self._model = active.model
        self._max_tokens = active.max_tokens
        self._temperature = active.temperature

        # Optional routing client for tiered architecture
        self._routing_client: AsyncOpenAI | None = None
        self._routing_model: str = active.model
        if config.tiered.enabled:
            routing = config.get_routing()
            self._routing_client = AsyncOpenAI(
                base_url=routing.base_url or None,
                api_key=routing.api_key or "not-needed",
                timeout=routing.timeout,
            )
            self._routing_model = routing.model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
        model_tier: str = "generation",
    ) -> LLMResponse:
        """Send a chat completion request.

        Parameters
        ----------
        messages:
            OpenAI-format message list.
        tools:
            Optional tool definitions (OpenAI function-calling format).
        stream:
            Whether to stream the response (not yet implemented).
        model_tier:
            ``"generation"`` (default) uses the main model;
            ``"routing"`` uses the cheaper routing model when tiered LLM
            is enabled.

        Returns
        -------
        LLMResponse
        """
        client, model = self._select_client_and_model(model_tier)

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = await client.chat.completions.create(**kwargs)
        except APITimeoutError as exc:
            raise LLMClientError(
                f"LLM request timed out "
                f"(provider={self._config.default_provider}, model={model})"
            ) from exc
        except APIConnectionError as exc:
            raise LLMClientError(
                f"Failed to connect to LLM provider "
                f"(provider={self._config.default_provider}, base_url={client.base_url}): {exc}"
            ) from exc

        return self._parse_response(response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_client_and_model(
        self, model_tier: str
    ) -> tuple[AsyncOpenAI, str]:
        """Return the appropriate client and model name for *model_tier*."""
        if (
            model_tier == "routing"
            and self._config.tiered.enabled
            and self._routing_client is not None
        ):
            return self._routing_client, self._routing_model
        return self._client, self._model

    @staticmethod
    def _parse_response(response) -> LLMResponse:
        """Convert the raw OpenAI SDK response into an ``LLMResponse``."""
        choice = response.choices[0]
        message = choice.message

        # --- content ---
        content = message.content

        # --- tool calls ---
        parsed_tool_calls: list[ToolCall] | None = None
        if message.tool_calls:
            import json

            parsed_tool_calls = []
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                parsed_tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        # --- usage ---
        usage_data = response.usage
        prompt_tokens = usage_data.prompt_tokens if usage_data else 0
        completion_tokens = usage_data.completion_tokens if usage_data else 0
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        return LLMResponse(
            content=content,
            tool_calls=parsed_tool_calls,
            usage=usage,
        )
