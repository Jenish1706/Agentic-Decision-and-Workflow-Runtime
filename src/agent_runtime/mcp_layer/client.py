"""MCP client wrapper.

This module is named `mcp_layer`, not `mcp`, specifically so it never
shadows the third-party `mcp` SDK package in absolute imports - a package
literally named `mcp` that also needs to `import mcp` is asking for
confusion, even though Python's absolute-import resolution would technically
keep them apart.

Each configured server is spawned as its own subprocess over stdio and kept
alive for the lifetime of this client (opened once per script/eval run, not
once per query) - `MCPToolClient` is an async context manager for exactly
that reason. Everything above this file (agent/tools/mcp_tool.py, the agent
loop) only ever sees `Tool` objects; this is the one place that actually
speaks the MCP client protocol (list_tools / call_tool).
"""

import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult

from ..config import REPO_ROOT

_DEFAULT_SERVER_MODULES = {
    "knowledge": "agent_runtime.mcp_layer.servers.knowledge_server",
    "records": "agent_runtime.mcp_layer.servers.records_server",
    "analytics": "agent_runtime.mcp_layer.servers.analytics_server",
}


@dataclass(frozen=True)
class DiscoveredTool:
    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPToolClient:
    def __init__(self, server_modules: dict[str, str] | None = None):
        self._server_modules = server_modules or _DEFAULT_SERVER_MODULES
        self._clients: dict[str, Client] = {}
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "MCPToolClient":
        for server_name, module in self._server_modules.items():
            params = StdioServerParameters(command=sys.executable, args=["-m", module], cwd=str(REPO_ROOT))
            client = await self._stack.enter_async_context(Client(params))
            self._clients[server_name] = client
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._stack.aclose()

    async def discover_tools(self) -> list[DiscoveredTool]:
        discovered: list[DiscoveredTool] = []
        for server_name, client in self._clients.items():
            listing = await client.list_tools()
            for tool in listing.tools:
                discovered.append(
                    DiscoveredTool(
                        server_name=server_name,
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.input_schema,
                    )
                )
        return discovered

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        client = self._clients[server_name]
        return await client.call_tool(tool_name, arguments)
