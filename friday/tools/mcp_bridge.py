"""Use any MCP server's tools as FRIDAY tools — e.g. Jarvis for Gmail and Calendar.

FRIDAY's assistant loop is synchronous (it runs in voice threads), while MCP clients are
async. The bridge owns a private event loop in a daemon thread that keeps one client
session open; tool calls are submitted to it with ``run_coroutine_threadsafe``.

MCP results are third-party content, so bridged tools are registered as *untrusted*: they
taint the turn, and FRIDAY's own actions are then blocked (see registry.py). Jarvis's
side-effect tools are still queued for approval on Jarvis's side.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import threading
from typing import Any

from .registry import Tool, ToolRegistry

log = logging.getLogger(__name__)


class MCPBridge:
    def __init__(self, target: Any, prefix: str = "jarvis_", timeout: float = 60.0):
        """``target`` is anything ``mcp.client.Client`` accepts: a server object (in-process,
        used in tests) or ``StdioServerParameters`` for a subprocess."""
        self.target, self.prefix, self.timeout = target, prefix, timeout
        self._loop = asyncio.new_event_loop()
        self._client: Any = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="mcp-bridge", daemon=True)

    @classmethod
    def from_command(cls, command: str, **kwargs: Any) -> MCPBridge:
        from mcp.client.stdio import StdioServerParameters

        parts = shlex.split(command)
        return cls(StdioServerParameters(command=parts[0], args=parts[1:]), **kwargs)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        from mcp.client import Client

        self._stop = asyncio.Event()
        try:
            async with Client(self.target) as client:
                self._client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller of start()
            self._error = exc
            self._ready.set()

    def start(self) -> MCPBridge:
        self._thread.start()
        if not self._ready.wait(self.timeout) or self._error:
            raise RuntimeError(f"Could not connect to MCP server: {self._error or 'timeout'}")
        return self

    def _call(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout)

    def list_tools(self) -> list[Any]:
        return list(self._call(self._client.list_tools()).tools)

    def call(self, name: str, args: dict[str, Any]) -> str:
        result = self._call(self._client.call_tool(name, args))
        text = "\n".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            raise RuntimeError(text or "tool error")
        return text

    def register_into(self, registry: ToolRegistry) -> list[str]:
        added = []
        for t in self.list_tools():
            name = f"{self.prefix}{t.name}"

            def func(_remote: str = t.name, **kwargs: Any) -> str:
                return self.call(_remote, kwargs)

            schema = dict(t.input_schema or {"type": "object", "properties": {}})
            registry.add(
                Tool(name, f"[via {self.prefix.rstrip('_')}] {t.description or ''}", schema, func, untrusted=True)
            )
            added.append(name)
        log.info("Bridged %d MCP tools", len(added))
        return added

    def close(self) -> None:
        if self._stop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=5)

    def __repr__(self) -> str:  # pragma: no cover
        return f"MCPBridge({json.dumps(str(self.target))[:60]})"
