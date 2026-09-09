"""Tool registration, discovery, and dispatch - one flat namespace whether
a tool came from a local Python class or from MCP discovery."""

from typing import Any

from .base import Tool, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
                "origin": tool.origin,
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            known = ", ".join(sorted(self._tools)) or "(none registered)"
            return ToolResult(ok=False, content="", error=f"Unknown tool '{name}'. Known tools: {known}")
        try:
            return await tool.run(**arguments)
        except TypeError as exc:
            return ToolResult(ok=False, content="", error=f"Bad arguments for '{name}': {exc}")
        except Exception as exc:  # noqa: BLE001 - tool failures must never crash the agent loop
            return ToolResult(ok=False, content="", error=f"'{name}' raised an error: {exc}")


def build_restricted_registry(full_registry: "ToolRegistry", exclude_tool_names: set[str]) -> "ToolRegistry":
    """A copy of `full_registry` leaving out specific tools by name. Used to
    keep a capability out of an agent's reach even though it was discovered
    - e.g. `search_documents` is intentionally not exposed as a generic
    callable tool anywhere (see agent/mcp_retriever.py for why)."""
    restricted = ToolRegistry()
    for schema in full_registry.list_schemas():
        if schema["name"] not in exclude_tool_names:
            tool = full_registry.get(schema["name"])
            if tool is not None:
                restricted.register(tool)
    return restricted
