"""Tool interface shared by local (in-process) and MCP-routed tools. The
agent loop and the LLM-facing schema builder only ever see this interface -
neither knows or cares whether a given tool runs as a plain Python call or
as a round trip through an MCP server subprocess. That indifference is the
point: `origin` is metadata for humans reading the trace, not something the
agent loop branches on.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel


class ToolResult(BaseModel):
    ok: bool
    content: str
    data: dict[str, Any] | None = None
    error: str | None = None


class Tool(ABC):
    name: str
    description: str
    parameters_schema: dict[str, Any]
    origin: Literal["local", "mcp"]

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult: ...
