"""Structured logging and monitoring for AgentDesk RAG Platform.

Provides:
- Configurable log level and file-based logging
- Conversation logging (session_id, message, response, tools, timestamp)
- Token usage and estimated cost logging per LLM call
- Tool call logging (name, input, output, duration)

All loggers use Python's standard ``logging`` module.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from src.config.models import LoggingConfig


# ---------------------------------------------------------------------------
# Logger names (importable by other modules)
# ---------------------------------------------------------------------------

CONVERSATION_LOGGER = "agentdesk.conversation"
TOOL_CALL_LOGGER = "agentdesk.tool_call"
TOKEN_USAGE_LOGGER = "agentdesk.token_usage"

# ---------------------------------------------------------------------------
# Default cost rates (USD per 1K tokens) — rough estimates
# ---------------------------------------------------------------------------

_DEFAULT_COST_PER_1K_PROMPT = 0.0015
_DEFAULT_COST_PER_1K_COMPLETION = 0.002


def setup_logging(config: LoggingConfig, level: str = "INFO") -> None:
    """Configure the root logger and specialised sub-loggers.

    Parameters
    ----------
    config:
        ``LoggingConfig`` from the application configuration.
    level:
        Root log level (e.g. ``"DEBUG"``, ``"INFO"``).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Ensure log directories exist
    _ensure_log_dir(config.log_file)
    _ensure_log_dir(config.cost_log_file)

    # --- Root logger ---
    root = logging.getLogger()
    root.setLevel(log_level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root.addHandler(console)

    # File handler (main log)
    if config.log_file:
        fh = logging.FileHandler(config.log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        root.addHandler(fh)

    # --- Cost / token usage file handler ---
    if config.track_token_usage and config.cost_log_file:
        cost_handler = logging.FileHandler(config.cost_log_file, encoding="utf-8")
        cost_handler.setLevel(logging.INFO)
        cost_handler.setFormatter(logging.Formatter("%(message)s"))
        token_logger = logging.getLogger(TOKEN_USAGE_LOGGER)
        token_logger.addHandler(cost_handler)
        token_logger.propagate = False


# ---------------------------------------------------------------------------
# Structured log helpers
# ---------------------------------------------------------------------------


def log_conversation(
    session_id: str,
    user_message: str,
    llm_response: str,
    tools_used: list[str],
) -> None:
    """Log a completed conversation turn.

    Validates: Requirements 14.1
    """
    logger = logging.getLogger(CONVERSATION_LOGGER)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "user_message": user_message,
        "llm_response": llm_response[:500],
        "tools_used": tools_used,
    }
    logger.info(json.dumps(entry, ensure_ascii=False))


def log_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    model: str = "",
    cost_per_1k_prompt: float = _DEFAULT_COST_PER_1K_PROMPT,
    cost_per_1k_completion: float = _DEFAULT_COST_PER_1K_COMPLETION,
) -> None:
    """Log token usage and estimated cost for an LLM call.

    Validates: Requirements 14.2
    """
    estimated_cost = (
        (prompt_tokens / 1000) * cost_per_1k_prompt
        + (completion_tokens / 1000) * cost_per_1k_completion
    )
    logger = logging.getLogger(TOKEN_USAGE_LOGGER)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
    }
    logger.info(json.dumps(entry, ensure_ascii=False))


def log_tool_call(
    tool_name: str,
    tool_input: dict | str,
    tool_output: str,
    duration_ms: float,
) -> None:
    """Log a single tool call with input, output, and duration.

    Validates: Requirements 14.3
    """
    logger = logging.getLogger(TOOL_CALL_LOGGER)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tool_name": tool_name,
        "input": tool_input if isinstance(tool_input, str) else json.dumps(tool_input, ensure_ascii=False),
        "output": tool_output[:500],
        "duration_ms": round(duration_ms, 2),
    }
    logger.info(json.dumps(entry, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_log_dir(path: str) -> None:
    """Create parent directories for a log file path if they don't exist."""
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
