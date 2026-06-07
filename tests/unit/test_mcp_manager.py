"""Unit tests for MCPManager.

Tests cover server lifecycle (start/stop), status reporting,
tool call routing, and restart logic - all with mocked subprocesses.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.models import MCPServerConfig
from src.mcp.manager import MCPManager, MCPManagerError, MAX_RESTART_RETRIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _stdio_config(enabled: bool = True) -> MCPServerConfig:
    return MCPServerConfig(
        enabled=enabled,
        transport="stdio",
        command="python",
        args=["-m", "mcp_server"],
        env={"KEY": "val"},
    )


def _sse_config(enabled: bool = True) -> MCPServerConfig:
    return MCPServerConfig(
        enabled=enabled,
        transport="sse",
        command="http://localhost:9000",
    )


def _make_manager(**overrides: MCPServerConfig) -> MCPManager:
    servers: dict[str, MCPServerConfig] = {
        "db": _stdio_config(),
        "search": _sse_config(),
        **overrides,
    }
    return servers, MCPManager(servers)


def _mock_process() -> MagicMock:
    """Return a mock that behaves like ``asyncio.subprocess.Process``."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(return_value=b'{"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n')
    proc.stderr = MagicMock()
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    proc.returncode = 0
    return proc


# ---------------------------------------------------------------------------
# Start / Stop
# ---------------------------------------------------------------------------


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_stdio_server(self) -> None:
        _, mgr = _make_manager()
        mock_proc = _mock_process()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await mgr.start_server("db")

        status = mgr.get_status()
        assert status["db"].status == "running"
        assert status["db"].uptime_seconds is not None

    @pytest.mark.asyncio
    async def test_start_sse_server(self) -> None:
        _, mgr = _make_manager()
        await mgr.start_server("search")

        status = mgr.get_status()
        assert status["search"].status == "running"

    @pytest.mark.asyncio
    async def test_stop_stdio_server(self) -> None:
        _, mgr = _make_manager()
        mock_proc = _mock_process()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await mgr.start_server("db")
            await mgr.stop_server("db")

        assert mgr.get_status()["db"].status == "stopped"
        mock_proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_sse_server(self) -> None:
        _, mgr = _make_manager()
        await mgr.start_server("search")
        await mgr.stop_server("search")

        assert mgr.get_status()["search"].status == "stopped"

    @pytest.mark.asyncio
    async def test_start_already_running_is_noop(self) -> None:
        _, mgr = _make_manager()
        mock_proc = _mock_process()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc) as mock_exec:
            await mgr.start_server("db")
            await mgr.start_server("db")

        # Should only have been called once
        assert mock_exec.await_count == 1

    @pytest.mark.asyncio
    async def test_stop_already_stopped_is_noop(self) -> None:
        _, mgr = _make_manager()
        # Server never started - stop should not raise
        await mgr.stop_server("db")
        assert mgr.get_status()["db"].status == "stopped"

    @pytest.mark.asyncio
    async def test_start_unknown_server_raises(self) -> None:
        _, mgr = _make_manager()
        with pytest.raises(MCPManagerError, match="Unknown server"):
            await mgr.start_server("nonexistent")

    @pytest.mark.asyncio
    async def test_start_all_starts_only_enabled(self) -> None:
        servers = {
            "enabled_srv": _stdio_config(enabled=True),
            "disabled_srv": _stdio_config(enabled=False),
        }
        mgr = MCPManager(servers)
        mock_proc = _mock_process()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await mgr.start_all()

        status = mgr.get_status()
        assert status["enabled_srv"].status == "running"
        assert status["disabled_srv"].status == "stopped"

    @pytest.mark.asyncio
    async def test_stop_all(self) -> None:
        _, mgr = _make_manager()
        mock_proc = _mock_process()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await mgr.start_all()
            await mgr.stop_all()

        status = mgr.get_status()
        assert status["db"].status == "stopped"
        assert status["search"].status == "stopped"


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------


class TestStatusReporting:
    def test_initial_status_all_stopped(self) -> None:
        _, mgr = _make_manager()
        status = mgr.get_status()
        for srv in status.values():
            assert srv.status == "stopped"
            assert srv.uptime_seconds is None

    @pytest.mark.asyncio
    async def test_status_reflects_running(self) -> None:
        _, mgr = _make_manager()
        mock_proc = _mock_process()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await mgr.start_server("db")

        s = mgr.get_status()["db"]
        assert s.status == "running"
        assert s.transport == "stdio"
        assert s.uptime_seconds is not None and s.uptime_seconds >= 0

    @pytest.mark.asyncio
    async def test_status_reflects_error(self) -> None:
        _, mgr = _make_manager()

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            side_effect=OSError("spawn failed"),
        ):
            with pytest.raises(OSError):
                await mgr.start_server("db")

        s = mgr.get_status()["db"]
        assert s.status == "error"
        assert s.error_message is not None


# ---------------------------------------------------------------------------
# Tool call routing
# ---------------------------------------------------------------------------


class TestToolCallRouting:
    @pytest.mark.asyncio
    async def test_call_tool_stdio(self) -> None:
        _, mgr = _make_manager()
        mock_proc = _mock_process()
        expected_result = {"rows": [{"id": 1}]}
        mock_proc.stdout.readline = AsyncMock(
            return_value=json.dumps(
                {"jsonrpc": "2.0", "id": 1, "result": expected_result}
            ).encode() + b"\n"
        )

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await mgr.start_server("db")
            result = await mgr.call_tool("db", "query", {"sql": "SELECT 1"})

        assert result == expected_result

    @pytest.mark.asyncio
    async def test_call_tool_sse(self) -> None:
        _, mgr = _make_manager()
        await mgr.start_server("search")

        expected_result = {"matches": []}
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": expected_result,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await mgr.call_tool("search", "vector_search", {"query": "test"})

        assert result == expected_result

    @pytest.mark.asyncio
    async def test_call_tool_on_stopped_server_raises(self) -> None:
        _, mgr = _make_manager()
        with pytest.raises(MCPManagerError, match="not running"):
            await mgr.call_tool("db", "query", {})

    @pytest.mark.asyncio
    async def test_call_tool_stdio_error_response(self) -> None:
        _, mgr = _make_manager()
        mock_proc = _mock_process()
        mock_proc.stdout.readline = AsyncMock(
            return_value=json.dumps(
                {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "fail"}}
            ).encode() + b"\n"
        )

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await mgr.start_server("db")
            with pytest.raises(MCPManagerError, match="Tool call error"):
                await mgr.call_tool("db", "bad_tool", {})


# ---------------------------------------------------------------------------
# Restart logic
# ---------------------------------------------------------------------------


class TestRestartLogic:
    @pytest.mark.asyncio
    async def test_restart_succeeds(self) -> None:
        _, mgr = _make_manager()
        mock_proc = _mock_process()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await mgr.start_server("db")
            await mgr.restart_server("db")

        assert mgr.get_status()["db"].status == "running"

    @pytest.mark.asyncio
    async def test_restart_increments_count(self) -> None:
        _, mgr = _make_manager()
        mock_proc = _mock_process()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await mgr.start_server("db")
            await mgr.restart_server("db")
            await mgr.restart_server("db")

        # Access internal state for restart count verification
        assert mgr._servers["db"].restart_count == 2

    @pytest.mark.asyncio
    async def test_restart_fails_after_max_retries(self) -> None:
        _, mgr = _make_manager()

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            side_effect=OSError("spawn failed"),
        ):
            with pytest.raises(MCPManagerError, match="Failed to restart"):
                await mgr.restart_server("db")

        assert mgr.get_status()["db"].status == "error"


# ---------------------------------------------------------------------------
# list_available_tools
# ---------------------------------------------------------------------------


class TestListAvailableTools:
    @pytest.mark.asyncio
    async def test_lists_tools_from_running_servers(self) -> None:
        _, mgr = _make_manager()
        mock_proc = _mock_process()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc):
            await mgr.start_server("db")

        # Inject tools into internal state for testing
        mgr._servers["db"].tools = [
            {"name": "query", "description": "Run SQL"},
        ]

        tools = mgr.list_available_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "query"
        assert tools[0]["server_name"] == "db"

    def test_no_tools_when_all_stopped(self) -> None:
        _, mgr = _make_manager()
        assert mgr.list_available_tools() == []
