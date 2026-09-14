"""Tool registration, discovery and dispatch.

One flat namespace, whether a tool came from a local Python class or from
MCP discovery. Two behaviours here are deliberate safety properties:

- Registering a second tool under an existing name raises. It used to
  silently overwrite, and two MCP servers both expose `execute_sql`, so
  connecting both would have routed every query to whichever server happened
  to register last - against the wrong database, with no error.
- `execute` never raises. Unknown tools, bad arguments and tool crashes all
  come back as `ToolResult(ok=False, ...)`, which the loop turns into an
  observation the model can react to, instead of aborting the run.
"""

import inspect
from typing import Any

from .base import Tool, ToolResult


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named '{tool.name}' is already registered. Tool names must be unique "
                "within one runtime; connect only one server that provides it."
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def list_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
                "origin": tool.origin,
                "category": tool.category.value,
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            known = ", ".join(self.names()) or "(none registered)"
            return ToolResult(ok=False, content="", error=f"Unknown tool '{name}'. Known tools: {known}")

        # Check the arguments against the signature *before* calling, so that
        # a TypeError raised inside the tool's own logic is not misreported as
        # "bad arguments" (the previous version caught TypeError around the
        # whole call and could not tell the two apart).
        try:
            inspect.signature(tool.run).bind(**arguments)
        except TypeError as exc:
            return ToolResult(ok=False, content="", error=f"Bad arguments for '{name}': {exc}")

        try:
            return await tool.run(**arguments)
        except Exception as exc:  # noqa: BLE001 - a tool failure must never crash the agent loop
            return ToolResult(ok=False, content="", error=f"'{name}' raised an error: {exc}")


def build_restricted_registry(full_registry: ToolRegistry, exclude_tool_names: set[str]) -> ToolRegistry:
    """A copy of `full_registry` without some tools.

    Used to keep a discovered capability out of an agent's reach - e.g.
    `search_documents` is never exposed as a generic callable tool, because
    the dedicated `retrieve` action already reaches it and is the only path
    that feeds citation tracking (see agent/mcp_retriever.py).
    """
    restricted = ToolRegistry()
    for name in full_registry.names():
        tool = full_registry.get(name)
        if tool is not None and name not in exclude_tool_names:
            restricted.register(tool)
    return restricted
