"""Tool interface shared by local (in-process) and MCP-routed tools.

The agent loop and the LLM-facing schema builder only ever see this
interface; neither cares whether a tool runs as a plain Python call or as a
round trip to an MCP server subprocess. `origin` is metadata for humans
reading a trace, not something the loop branches on.

`category` IS something the loop branches on, and it replaced a real bug.
Budgets and evidence provenance used to be decided by guessing from the tool
*name*: a name containing "sql" was a SQL query, one containing "document" or
"chart" was vision, and so on. That was brittle in every direction: any tool
named "...document..." (e.g. a plain text search) would be filed as vision
evidence, schema discovery (`describe_tables`) was charged against the SQL
query budget, the calculator was recorded as a record lookup, and
`max_vision_calls` was never enforced at all because no tool mapped to it.
Each tool now declares what it is. Local tools set the class attribute; MCP
tools declare it in their server-side metadata
(`@server.tool(meta={"category": "sql"})`), which the MCP protocol carries
through discovery.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel


class ToolCategory(str, Enum):
    """What kind of work a tool does. Drives budget accounting and the
    `EvidenceSource` recorded for its results."""

    SQL = "sql"  # queries against a structured database (budgeted)
    STATISTICS = "statistics"  # statistical tests / derived calculations
    VISION = "vision"  # document or chart extraction (budgeted)
    RECORD = "record"  # structured record lookups
    CALCULATION = "calculation"  # arithmetic


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
    category: ToolCategory = ToolCategory.RECORD

    @abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult: ...
