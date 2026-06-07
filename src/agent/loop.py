"""Agent Loop - iterative tool-calling loop between LLM and MCP servers.

The ``AgentLoop`` sends messages + available tools to the LLM, executes any
tool calls via ``MCPManager``, appends results, and loops until the LLM
returns a final text response or the iteration limit is reached.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from src.llm.client import LLMClient, LLMResponse
from src.mcp.manager import MCPManager
from src.models.schemas import Citation, TokenUsage, ToolTrace
from src.router.intent import IntentResult
from src.session.manager import Session
from src.store.reference import ReferenceStore

logger = logging.getLogger(__name__)

# Default reference threshold in characters
DEFAULT_REFERENCE_THRESHOLD = 4000


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """Final response produced by the agent loop."""

    content: str
    citations: list[Citation] = field(default_factory=list)
    tool_traces: list[ToolTrace] = field(default_factory=list)
    usage: TokenUsage = field(
        default_factory=lambda: TokenUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
    )


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    """Core iterative tool-calling loop.

    Parameters
    ----------
    llm_client:
        The unified LLM client for chat completions.
    mcp_manager:
        Manager that routes tool calls to MCP servers.
    reference_store:
        In-memory store for large tool results.
    max_iterations:
        Maximum number of LLM  tool round-trips before forced stop.
    reference_threshold:
        Character count above which tool results are stored in the
        reference store and replaced with a ``ref_xxx`` code.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        mcp_manager: MCPManager,
        reference_store: ReferenceStore,
        max_iterations: int = 10,
        reference_threshold: int = DEFAULT_REFERENCE_THRESHOLD,
    ) -> None:
        self._llm_client = llm_client
        self._mcp_manager = mcp_manager
        self._reference_store = reference_store
        self._max_iterations = max_iterations
        self._reference_threshold = reference_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: list[dict],
        session: Session,
        intent: IntentResult,
    ) -> AgentResponse:
        """Execute the agent loop.

        Sends *messages* plus available tools to the LLM.  If the LLM
        returns tool calls, each is executed via ``MCPManager`` and the
        results are appended to the conversation.  The loop continues
        until the LLM produces a final text response or
        ``max_iterations`` is exceeded.
        """
        tools = self._mcp_manager.list_available_tools()
        tool_definitions = self._build_tool_definitions(tools)

        total_usage = TokenUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
        all_traces: list[ToolTrace] = []
        all_citations: list[Citation] = []

        working_messages = list(messages)

        for iteration in range(self._max_iterations):
            llm_response = await self._llm_client.chat_completion(
                messages=working_messages,
                tools=tool_definitions if tool_definitions else None,
            )

            # Accumulate token usage
            total_usage = self._accumulate_usage(total_usage, llm_response.usage)

            # If no tool calls, we have a final response
            if not llm_response.tool_calls:
                content = llm_response.content or ""
                citations = self._extract_citations(content)
                all_citations.extend(citations)
                return AgentResponse(
                    content=content,
                    citations=all_citations,
                    tool_traces=all_traces,
                    usage=total_usage,
                )

            # Process each tool call
            assistant_message = {
                "role": "assistant",
                "content": llm_response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in llm_response.tool_calls
                ],
            }
            working_messages.append(assistant_message)

            for tc in llm_response.tool_calls:
                server_name, tool_name = self._resolve_tool(tc.name, tools)

                start = time.perf_counter()
                try:
                    result = await self._mcp_manager.call_tool(
                        server_name=server_name,
                        tool_name=tool_name,
                        arguments=tc.arguments,
                    )
                except Exception as exc:
                    logger.error(
                        "Tool call failed: server=%s tool=%s error=%s",
                        server_name, tool_name, exc,
                    )
                    result = {"error": str(exc)}
                duration_ms = (time.perf_counter() - start) * 1000

                result_text = json.dumps(result) if isinstance(result, dict) else str(result)

                # Extract citations from tool results
                tool_citations = self._extract_citations_from_tool_result(
                    result, server_name
                )
                all_citations.extend(tool_citations)

                # Reference store: if result is too large, store and use ref code
                if len(result_text) > self._reference_threshold:
                    ref_id = self._reference_store.store(
                        result_text,
                        metadata={"tool_name": tool_name, "server_name": server_name},
                    )
                    tool_result_for_llm = (
                        f"[Result stored as reference: {ref_id}. "
                        f"Use this reference code to access the full data.]"
                    )
                else:
                    tool_result_for_llm = result_text

                # Record trace
                all_traces.append(
                    ToolTrace(
                        tool_name=tool_name,
                        server_name=server_name,
                        arguments=tc.arguments,
                        result_summary=result_text[:200],
                        duration_ms=round(duration_ms, 2),
                    )
                )

                # Append tool result message
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result_for_llm,
                    }
                )

        # Max iterations exceeded
        logger.warning(
            "Agent loop exceeded max iterations (%d)", self._max_iterations
        )
        return AgentResponse(
            content=(
                "Maksimum iterasyon sayısına ulaşıldı. "
                "Lütfen sorunuzu daha spesifik şekilde yeniden ifade edin."
            ),
            citations=all_citations,
            tool_traces=all_traces,
            usage=total_usage,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tool_definitions(tools: list[dict]) -> list[dict]:
        """Convert MCP tool list into OpenAI function-calling format."""
        definitions: list[dict] = []
        for tool in tools:
            definition = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
            definitions.append(definition)
        return definitions

    @staticmethod
    def _resolve_tool(
        tool_name: str, tools: list[dict]
    ) -> tuple[str, str]:
        """Find the server_name for a given tool_name.

        Returns ``(server_name, tool_name)``.  If the tool is not found
        in the available tools list, the server_name defaults to
        ``"unknown"``.
        """
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool.get("server_name", "unknown"), tool_name
        return "unknown", tool_name

    @staticmethod
    def _accumulate_usage(total: TokenUsage, new: TokenUsage) -> TokenUsage:
        """Sum token usage across multiple LLM calls."""
        return TokenUsage(
            prompt_tokens=total.prompt_tokens + new.prompt_tokens,
            completion_tokens=total.completion_tokens + new.completion_tokens,
            total_tokens=total.total_tokens + new.total_tokens,
        )

    @staticmethod
    def _extract_citations(content: str) -> list[Citation]:
        """Extract citations from LLM response content.

        Looks for patterns like [source: X, page: Y] in the text.
        """
        import re

        citations: list[Citation] = []
        pattern = r"\[source:\s*(.+?)(?:,\s*page:\s*(\d+))?\]"
        for match in re.finditer(pattern, content, re.IGNORECASE):
            source = match.group(1).strip()
            page = int(match.group(2)) if match.group(2) else None
            citations.append(
                Citation(
                    source=source,
                    page=page,
                    text=match.group(0),
                    score=1.0,
                )
            )
        return citations

    @staticmethod
    def _extract_citations_from_tool_result(
        result: dict | str, server_name: str
    ) -> list[Citation]:
        """Extract citations from tool results that contain source info."""
        citations: list[Citation] = []
        if not isinstance(result, dict):
            return citations

        # Handle results that directly contain source/page info
        if "source" in result:
            citations.append(
                Citation(
                    source=result.get("source", ""),
                    page=result.get("page"),
                    text=result.get("text", str(result.get("source", "")))[:200],
                    score=result.get("score", 1.0),
                )
            )

        # Handle results with a list of sources/documents
        for key in ("results", "documents", "sources"):
            items = result.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "source" in item:
                        citations.append(
                            Citation(
                                source=item.get("source", ""),
                                page=item.get("page"),
                                text=item.get("text", "")[:200],
                                score=item.get("score", 1.0),
                            )
                        )

        return citations
