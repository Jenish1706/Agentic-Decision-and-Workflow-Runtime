"""MCP client wrapper.

This package is named `mcp_layer`, not `mcp`, so it never shadows the
third-party `mcp` SDK in absolute imports.

WHAT
    `MCPToolClient` spawns each configured MCP server as its own subprocess,
    speaks the MCP protocol to it over stdio, discovers its tools, and routes
    `call_tool` requests to the right server.

WHY A CONTEXT MANAGER
    Servers are launched once per client session (a script, an evaluation
    run) and kept alive for every call within it, rather than started per
    query. The knowledge server builds an embedding index at startup, so
    per-query spawning would repeat that cost on every question.

HOW SERVERS ARE SPECIFIED
    A server is either a dotted module path (`agent_runtime.mcp_layer.servers.
    knowledge_server`, launched with `python -m`) or a filesystem path to a
    `.py` file (`domains/credit_risk/server.py`, launched as a script).
    Path support is what lets business domains keep their servers inside
    their own `domains/<name>/` folder, outside the generic runtime package.

DEFAULTS
    With no servers given, only the built-in knowledge base is connected
    (`knowledge` + `records`, serving the Nimbus corpus in data/). Domain
    servers are opt-in. An earlier version also connected the product
    analytics server by default, which quietly exposed `execute_sql` to every
    knowledge-base evaluation run - tools irrelevant to those questions that
    the model could still waste steps on.
"""

import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult

from ..config import REPO_ROOT

DEFAULT_SERVER_MODULES: dict[str, str] = {
    "knowledge": "agent_runtime.mcp_layer.servers.knowledge_server",
    "records": "agent_runtime.mcp_layer.servers.records_server",
}


@dataclass(frozen=True)
class DiscoveredTool:
    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]
    # Server-declared metadata, e.g. {"category": "sql"}; see agent/tools/base.py.
    meta: dict[str, Any] = field(default_factory=dict)


def server_parameters(spec: str | Path) -> StdioServerParameters:
    """Build launch parameters for one server spec (module name or .py path)."""
    spec_text = str(spec)
    if spec_text.endswith(".py"):
        script = Path(spec_text)
        if not script.is_absolute():
            script = REPO_ROOT / script
        if not script.is_file():
            raise FileNotFoundError(f"MCP server script not found: {script}")
        args = [str(script)]
    else:
        args = ["-m", spec_text]
    # cwd=REPO_ROOT keeps relative data paths and `-m` imports consistent no
    # matter where the calling script was started from.
    return StdioServerParameters(command=sys.executable, args=args, cwd=str(REPO_ROOT))


class MCPToolClient:
    def __init__(self, server_modules: dict[str, str | Path] | None = None):
        self._server_specs = dict(server_modules) if server_modules else dict(DEFAULT_SERVER_MODULES)
        self._clients: dict[str, Client] = {}
        self._stack = AsyncExitStack()

    @property
    def server_names(self) -> list[str]:
        return list(self._server_specs)

    async def __aenter__(self) -> "MCPToolClient":
        try:
            for server_name, spec in self._server_specs.items():
                client = await self._stack.enter_async_context(Client(server_parameters(spec)))
                self._clients[server_name] = client
        except BaseException:
            # If the third server fails to start, the first two are already
            # running subprocesses; close them rather than leaking processes.
            await self._stack.aclose()
            raise
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
                        meta=dict(tool.meta or {}),
                    )
                )
        return discovered

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        client = self._clients.get(server_name)
        if client is None:
            raise KeyError(f"MCP server '{server_name}' is not connected (connected: {sorted(self._clients)}).")
        return await client.call_tool(tool_name, arguments)
