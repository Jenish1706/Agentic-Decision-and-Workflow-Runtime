"""Adapter: wraps one MCP-discovered tool behind the same `Tool` interface
as a local tool (calculator.py). The agent loop cannot tell the difference;
`run()` here is the one place that actually crosses the protocol boundary -
a call_tool round trip to a separate subprocess - instead of a direct
Python function call."""

import json
from typing import Any

from ...mcp_layer.client import MCPToolClient
from .base import Tool, ToolResult


class MCPTool(Tool):
    origin = "mcp"

    def __init__(
        self,
        client: MCPToolClient,
        server_name: str,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
    ):
        self._client = client
        self._server_name = server_name
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema

    async def run(self, **kwargs: Any) -> ToolResult:
        result = await self._client.call_tool(self._server_name, self.name, kwargs)
        text_blocks = [block.text for block in result.content if hasattr(block, "text")]

        if result.is_error:
            return ToolResult(ok=False, content="", error="; ".join(text_blocks) or "MCP tool returned an error")
        if result.structured_content is not None:
            return ToolResult(ok=True, content=json.dumps(result.structured_content), data=result.structured_content)
        return ToolResult(ok=True, content="\n".join(text_blocks))


async def register_mcp_tools(client: MCPToolClient, registry) -> None:
    """Discovers every tool across the client's connected servers and
    registers each as an MCPTool - this is the tool-discovery step made
    concrete: schemas come from the servers at runtime, not from a
    hand-written list in this codebase."""
    for discovered in await client.discover_tools():
        registry.register(
            MCPTool(
                client=client,
                server_name=discovered.server_name,
                name=discovered.name,
                description=discovered.description,
                parameters_schema=discovered.input_schema,
            )
        )
