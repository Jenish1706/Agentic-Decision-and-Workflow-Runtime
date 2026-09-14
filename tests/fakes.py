"""Deterministic test doubles.

- `FakeLLMClient`: a scripted LLMClient, so agent-loop tests exercise real
  control flow (retries, termination, tool dispatch) without Ollama.
- `FunctionTool`: exposes a domain MCP server function as an in-process Tool.

Every domain flow test used to define its own copy of a fake LLM and its own
hand-written tool wrappers, each choosing a different subset of the server's
fields to pass on as evidence. Those subsets were not what the real MCP path
produces, so tests could pass on evidence the runtime never sees. One shared
double of each now mirrors the real path.
"""

import json
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from agent_runtime.agent.tools.base import Tool, ToolCategory, ToolResult
from agent_runtime.llm.base import LLMClient, LLMError, Message

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient(LLMClient):
    def __init__(self, decisions: list | None = None, generations: list[str] | None = None):
        self._decisions = list(decisions or [])
        self._generations = list(generations or [])
        self.decide_calls: list[list[Message]] = []
        self.generate_calls: list[list[Message]] = []

    async def generate(self, messages: list[Message]) -> str:
        self.generate_calls.append(messages)
        if self._generations:
            return self._generations.pop(0)
        return "generated text"

    async def decide(self, messages: list[Message], schema: type[T], *, max_retries: int = 2) -> T:
        self.decide_calls.append(messages)
        if not self._decisions:
            raise AssertionError("FakeLLMClient ran out of scripted decisions")
        item = self._decisions.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def raises_llm_error() -> LLMError:
    return LLMError("scripted failure: model never produced valid JSON")


class FunctionTool(Tool):
    """A domain server function behind the local Tool interface.

    Mirrors `MCPTool` exactly where it matters for evidence: the function's
    full pydantic result becomes `data` (as MCP structured content does), and
    exceptions surface as failed tool calls via the registry. `category` must
    match the `meta={"category": ...}` the server declares for the function;
    tests/test_mcp_servers.py checks those declarations over the real protocol.
    """

    origin = "local"

    def __init__(self, fn: Callable[..., BaseModel], category: ToolCategory):
        self._fn = fn
        self.name = fn.__name__
        self.description = (fn.__doc__ or fn.__name__).strip().splitlines()[0]
        self.parameters_schema = {"type": "object", "properties": {}}
        self.category = category

    async def run(self, **kwargs: Any) -> ToolResult:
        payload = self._fn(**kwargs).model_dump(mode="json")
        return ToolResult(ok=True, content=json.dumps(payload), data=payload)
