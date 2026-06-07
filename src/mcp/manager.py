"""MCP Manager - lifecycle management for MCP server subprocesses.

Handles starting, stopping, restarting, and communicating with MCP servers
configured in ``config.yaml``.  Supports **stdio** (subprocess with
stdin/stdout JSON-RPC) and **SSE** (HTTP via httpx) transports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import httpx

from src.config.models import MCPServerConfig
from src.models.schemas import ServerStatus

logger = logging.getLogger(__name__)

MAX_RESTART_RETRIES = 3


# ---------------------------------------------------------------------------
# Internal bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class _ServerState:
    """Runtime state for a single MCP server."""

    config: MCPServerConfig
    process: asyncio.subprocess.Process | None = None
    status: str = "stopped"  # running | stopped | error
    error_message: str | None = None
    start_time: float | None = None
    restart_count: int = 0
    tools: list[dict] = field(default_factory=list)
    _request_id: int = field(default=0, repr=False)

    def next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id


# ---------------------------------------------------------------------------
# MCPManager
# ---------------------------------------------------------------------------


class MCPManagerError(Exception):
    """Raised when an MCP operation fails."""


class MCPManager:
    """Manages the lifecycle of MCP servers.

    Parameters
    ----------
    servers:
        Mapping of server name  ``MCPServerConfig`` (from config.yaml).
    """

    def __init__(self, servers: dict[str, MCPServerConfig]) -> None:
        self._servers: dict[str, _ServerState] = {
            name: _ServerState(config=cfg) for name, cfg in servers.items()
        }

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    async def start_all(self) -> None:
        """Start every server whose config has ``enabled=True``."""
        tasks = [
            self.start_server(name)
            for name, state in self._servers.items()
            if state.config.enabled
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Gracefully stop all running servers."""
        tasks = [
            self.stop_server(name)
            for name, state in self._servers.items()
            if state.status == "running"
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Single-server lifecycle
    # ------------------------------------------------------------------

    async def start_server(self, name: str) -> None:
        """Start a single MCP server by *name*."""
        state = self._get_state(name)

        if state.status == "running":
            logger.info("Server '%s' is already running", name)
            return

        cfg = state.config
        try:
            if cfg.transport == "stdio":
                await self._start_stdio(name, state)
            elif cfg.transport == "sse":
                self._start_sse(name, state)
            else:
                raise MCPManagerError(
                    f"Unsupported transport '{cfg.transport}' for server '{name}'"
                )
            state.status = "running"
            state.start_time = time.time()
            state.error_message = None
            logger.info("Server '%s' started (transport=%s)", name, cfg.transport)
        except Exception as exc:
            state.status = "error"
            state.error_message = str(exc)
            logger.error("Failed to start server '%s': %s", name, exc)
            raise

    async def stop_server(self, name: str) -> None:
        """Gracefully stop a single MCP server."""
        state = self._get_state(name)

        if state.status != "running":
            logger.info("Server '%s' is not running (status=%s)", name, state.status)
            return

        cfg = state.config
        try:
            if cfg.transport == "stdio" and state.process is not None:
                await self._stop_stdio(name, state)
            # SSE servers are stateless HTTP - nothing to terminate
            state.status = "stopped"
            state.start_time = None
            state.error_message = None
            logger.info("Server '%s' stopped", name)
        except Exception as exc:
            state.status = "error"
            state.error_message = str(exc)
            logger.error("Failed to stop server '%s': %s", name, exc)

    async def restart_server(self, name: str) -> None:
        """Restart a server with up to ``MAX_RESTART_RETRIES`` attempts."""
        state = self._get_state(name)

        for attempt in range(1, MAX_RESTART_RETRIES + 1):
            try:
                if state.status == "running":
                    await self.stop_server(name)
                await self.start_server(name)
                state.restart_count += 1
                logger.info(
                    "Server '%s' restarted (attempt %d/%d)",
                    name, attempt, MAX_RESTART_RETRIES,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Restart attempt %d/%d for '%s' failed: %s",
                    attempt, MAX_RESTART_RETRIES, name, exc,
                )
                if attempt == MAX_RESTART_RETRIES:
                    state.status = "error"
                    state.error_message = (
                        f"Failed to restart after {MAX_RESTART_RETRIES} attempts: {exc}"
                    )
                    raise MCPManagerError(state.error_message) from exc

    # ------------------------------------------------------------------
    # Tool interaction
    # ------------------------------------------------------------------

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict
    ) -> dict:
        """Route a tool call to the correct MCP server and return the result."""
        state = self._get_state(server_name)

        if state.status != "running":
            raise MCPManagerError(
                f"Cannot call tool '{tool_name}' - server '{server_name}' "
                f"is not running (status={state.status})"
            )

        cfg = state.config
        if cfg.transport == "stdio":
            return await self._call_tool_stdio(state, tool_name, arguments)
        elif cfg.transport == "sse":
            return await self._call_tool_sse(state, tool_name, arguments)
        else:
            raise MCPManagerError(f"Unsupported transport '{cfg.transport}'")

    # ------------------------------------------------------------------
    # Status / discovery
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, ServerStatus]:
        """Return the status of every configured server."""
        result: dict[str, ServerStatus] = {}
        now = time.time()
        for name, state in self._servers.items():
            uptime: float | None = None
            if state.status == "running" and state.start_time is not None:
                uptime = now - state.start_time
            result[name] = ServerStatus(
                name=name,
                status=state.status,
                transport=state.config.transport,
                uptime_seconds=uptime,
                error_message=state.error_message,
            )
        return result

    def list_available_tools(self) -> list[dict]:
        """List tools from all running servers."""
        tools: list[dict] = []
        for name, state in self._servers.items():
            if state.status == "running":
                for tool in state.tools:
                    tools.append({**tool, "server_name": name})
        return tools

    # ------------------------------------------------------------------
    # stdio transport helpers
    # ------------------------------------------------------------------

    async def _start_stdio(self, name: str, state: _ServerState) -> None:
        cfg = state.config
        if not cfg.command:
            raise MCPManagerError(f"No command configured for stdio server '{name}'")

        cmd_parts = [cfg.command, *cfg.args]
        env = cfg.env if cfg.env else None

        state.process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def _stop_stdio(self, name: str, state: _ServerState) -> None:
        proc = state.process
        if proc is None:
            return

        # Try graceful termination first
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Server '%s' did not terminate in time, killing", name)
            proc.kill()
            await proc.wait()
        finally:
            state.process = None

    async def _call_tool_stdio(
        self, state: _ServerState, tool_name: str, arguments: dict
    ) -> dict:
        proc = state.process
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise MCPManagerError("stdio process is not available")

        request_id = state.next_request_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        payload = json.dumps(request) + "\n"
        proc.stdin.write(payload.encode())
        await proc.stdin.drain()

        raw_line = await proc.stdout.readline()
        if not raw_line:
            raise MCPManagerError("No response from stdio server")

        response = json.loads(raw_line.decode())
        if "error" in response:
            raise MCPManagerError(
                f"Tool call error: {response['error']}"
            )
        return response.get("result", {})

    # ------------------------------------------------------------------
    # SSE transport helpers
    # ------------------------------------------------------------------

    def _start_sse(self, name: str, state: _ServerState) -> None:
        """SSE servers are remote HTTP endpoints - just validate the config."""
        cfg = state.config
        if not cfg.command:
            raise MCPManagerError(
                f"No URL (command field) configured for SSE server '{name}'"
            )
        # For SSE, ``command`` holds the base URL of the remote server.

    async def _call_tool_sse(
        self, state: _ServerState, tool_name: str, arguments: dict
    ) -> dict:
        base_url = state.config.command.rstrip("/")
        request_body = {
            "jsonrpc": "2.0",
            "id": state.next_request_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base_url}/rpc",
                json=request_body,
            )
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise MCPManagerError(f"Tool call error: {data['error']}")
        return data.get("result", {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self, name: str) -> _ServerState:
        if name not in self._servers:
            raise MCPManagerError(f"Unknown server: '{name}'")
        return self._servers[name]
