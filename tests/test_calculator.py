import pytest

from agent_runtime.agent.tools.calculator import CalculatorTool


@pytest.mark.asyncio
async def test_calculator_evaluates_arithmetic():
    tool = CalculatorTool()
    result = await tool.run(expression="5000 * 60 * 10")
    assert result.ok
    assert result.data["result"] == 3_000_000


@pytest.mark.asyncio
async def test_calculator_handles_precedence_and_parens():
    tool = CalculatorTool()
    result = await tool.run(expression="(43 - 17) + 2 ** 3")
    assert result.ok
    assert result.data["result"] == 34


@pytest.mark.asyncio
async def test_calculator_rejects_unsupported_syntax():
    tool = CalculatorTool()
    result = await tool.run(expression="__import__('os').system('echo pwned')")
    assert not result.ok
    assert result.error is not None


@pytest.mark.asyncio
async def test_calculator_rejects_malformed_expression():
    tool = CalculatorTool()
    result = await tool.run(expression="5 + ")
    assert not result.ok
