import pytest

from agent_runtime.agent.tools.base import Tool, ToolResult
from agent_runtime.agent.tools.calculator import CalculatorTool
from agent_runtime.agent.tools.registry import ToolRegistry


class _AlwaysFailsTool(Tool):
    name = "always_fails"
    description = "A tool that always raises, to test registry error handling."
    parameters_schema = {"type": "object", "properties": {}}
    origin = "local"

    async def run(self, **kwargs):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_register_and_execute():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    result = await registry.execute("calculator", {"expression": "1 + 1"})
    assert result.ok
    assert result.data["result"] == 2


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_not_exception():
    registry = ToolRegistry()
    result = await registry.execute("nonexistent", {})
    assert not result.ok
    assert "Unknown tool" in result.error


@pytest.mark.asyncio
async def test_tool_exception_is_caught_as_tool_result():
    registry = ToolRegistry()
    registry.register(_AlwaysFailsTool())
    result = await registry.execute("always_fails", {})
    assert not result.ok
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_bad_arguments_reported_as_tool_result_not_exception():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    result = await registry.execute("calculator", {"wrong_arg": "1+1"})
    assert not result.ok


def test_list_schemas_reports_origin():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    schemas = registry.list_schemas()
    assert schemas[0]["name"] == "calculator"
    assert schemas[0]["origin"] == "local"
