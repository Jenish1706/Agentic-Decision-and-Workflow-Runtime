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


@pytest.mark.asyncio
async def test_calculator_bounds_work_before_doing_it():
    # Regression: `9**9**9` is whitelisted arithmetic that would hang the
    # process computing a number with hundreds of millions of digits.
    tool = CalculatorTool()
    for expression in ("9**9**9", "10**400", "2**101", "1e17 * 1e17"):
        result = await tool.run(expression=expression)
        assert not result.ok, expression


@pytest.mark.asyncio
async def test_calculator_rejects_booleans_as_numbers():
    tool = CalculatorTool()
    for expression in ("True + 1", "(1 < 2) + 1", "5 and 6", "not 3"):
        result = await tool.run(expression=expression)
        assert not result.ok, expression


@pytest.mark.asyncio
async def test_calculator_evaluates_comparisons_and_boolean_logic():
    # Observed with a real 7B model: it checks policy thresholds this way, and
    # every such call used to fail, ending one run at the step limit.
    tool = CalculatorTool()
    cases = {
        "(740 >= 650) and (0.28 <= 0.43) and (3000 <= 5000)": True,
        "0.01415 < 0.05 and 100 <= 50": False,
        "100 <= 50 or 0.48 > 0.43": True,
        "not (0.48 <= 0.43)": True,
        "1 < 3 <= 3": True,
        "1 < 3 < 2": False,
    }
    for expression, expected in cases.items():
        result = await tool.run(expression=expression)
        assert result.ok and result.data["result"] is expected, expression
