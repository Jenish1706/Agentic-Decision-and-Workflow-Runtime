"""Adapter: wraps one MCP-discovered tool behind the same `Tool` interface as
a local tool (calculator.py).

The agent loop cannot tell the difference. `run()` here is the one place
that actually crosses the protocol boundary - a `call_tool` round trip to a
separate server subprocess - instead of a direct Python function call.
"""

import json
from typing import Any

from ...mcp_layer.client import MCPToolClient
from .base import Tool, ToolCategory, ToolResult
from .registry import ToolRegistry


class MCPTool(Tool):
    origin = "mcp"

    def __init__(
        self,
        client: MCPToolClient,
        server_name: str,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
        category: ToolCategory = ToolCategory.RECORD,
    ):
        self._client = client
        self._server_name = server_name
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema
        self.category = category

    async def run(self, **kwargs: Any) -> ToolResult:
        result = await self._client.call_tool(self._server_name, self.name, kwargs)
        text_blocks = [block.text for block in result.content if hasattr(block, "text")]

        if result.is_error:
            return ToolResult(ok=False, content="", error="; ".join(text_blocks) or "MCP tool returned an error")
        if result.structured_content is not None:
            # Structured content is the tool's typed return value as JSON; it
            # becomes the evidence payload verbatim.
            return ToolResult(ok=True, content=json.dumps(result.structured_content), data=result.structured_content)
        return ToolResult(ok=True, content="\n".join(text_blocks))


def _category_from_meta(meta: dict[str, Any] | None) -> ToolCategory:
    """Read the category a server declared for its tool. Unknown or absent
    values fall back to RECORD rather than failing discovery."""
    raw = (meta or {}).get("category")
    try:
        return ToolCategory(raw)
    except ValueError:
        return ToolCategory.RECORD


async def register_mcp_tools(client: MCPToolClient, registry: ToolRegistry) -> None:
    """Discover every tool across the client's connected servers and register
    each as an MCPTool. This is tool discovery made concrete: names, schemas
    and categories come from the servers at runtime, not from a hand-written
    list in this codebase."""
    for discovered in await client.discover_tools():
        registry.register(
            MCPTool(
                client=client,
                server_name=discovered.server_name,
                name=discovered.name,
                description=discovered.description,
                parameters_schema=discovered.input_schema,
                category=_category_from_meta(discovered.meta),
            )
        )
