"""Unit tests for src/logging_config.py."""

from __future__ import annotations

import json
import logging

import pytest

from src.config.models import LoggingConfig
from src.logging_config import (
    CONVERSATION_LOGGER,
    TOKEN_USAGE_LOGGER,
    TOOL_CALL_LOGGER,
    log_conversation,
    log_token_usage,
    log_tool_call,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_loggers():
    """Remove handlers added by setup_logging after each test."""
    yield
    for name in (None, CONVERSATION_LOGGER, TOKEN_USAGE_LOGGER, TOOL_CALL_LOGGER):
        lgr = logging.getLogger(name)
        lgr.handlers.clear()


def test_setup_logging_creates_file_handler(tmp_path):
    log_file = str(tmp_path / "test.log")
    config = LoggingConfig(log_file=log_file, cost_log_file=str(tmp_path / "cost.log"))
    setup_logging(config, level="DEBUG")

    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) >= 1
    assert any(h.baseFilename.endswith("test.log") for h in file_handlers)


def test_setup_logging_creates_cost_handler(tmp_path):
    config = LoggingConfig(
        log_file=str(tmp_path / "app.log"),
        cost_log_file=str(tmp_path / "cost.log"),
        track_token_usage=True,
    )
    setup_logging(config)

    token_logger = logging.getLogger(TOKEN_USAGE_LOGGER)
    assert len(token_logger.handlers) >= 1


def test_log_conversation(caplog):
    with caplog.at_level(logging.INFO, logger=CONVERSATION_LOGGER):
        log_conversation(
            session_id="sess-1",
            user_message="hello",
            llm_response="hi there",
            tools_used=["postgres_mcp"],
        )

    assert len(caplog.records) == 1
    data = json.loads(caplog.records[0].message)
    assert data["session_id"] == "sess-1"
    assert data["user_message"] == "hello"
    assert "timestamp" in data
    assert data["tools_used"] == ["postgres_mcp"]


def test_log_token_usage(tmp_path):
    """Verify token usage is written to the cost log file."""
    cost_file = tmp_path / "cost.log"
    config = LoggingConfig(
        log_file=str(tmp_path / "app.log"),
        cost_log_file=str(cost_file),
        track_token_usage=True,
    )
    setup_logging(config)

    log_token_usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        model="gpt-4",
    )

    # Flush handlers
    for h in logging.getLogger(TOKEN_USAGE_LOGGER).handlers:
        h.flush()

    content = cost_file.read_text()
    data = json.loads(content.strip())
    assert data["prompt_tokens"] == 100
    assert data["completion_tokens"] == 50
    assert data["total_tokens"] == 150
    assert data["estimated_cost_usd"] > 0


def test_log_tool_call(caplog):
    with caplog.at_level(logging.INFO, logger=TOOL_CALL_LOGGER):
        log_tool_call(
            tool_name="search",
            tool_input={"query": "test"},
            tool_output="result data",
            duration_ms=42.5,
        )

    assert len(caplog.records) == 1
    data = json.loads(caplog.records[0].message)
    assert data["tool_name"] == "search"
    assert data["duration_ms"] == 42.5
