"""A local tool: registered directly, no protocol hop. Exists specifically
to contrast with the MCP-routed tools (agent/tools/mcp_tool.py) - same
`Tool` interface, same registry, but `origin='local'` means the registry
calls straight into this Python function instead of going through an MCP
client session.

Arithmetic is evaluated over a parsed AST with a fixed operator whitelist,
not `eval()` - the input is model-generated text, so treating it as trusted
code would be a real code-execution hole, not a theoretical one.

A whitelist stops code execution but not resource exhaustion: `9**9**9` is
valid arithmetic that would hang the process computing a number with
hundreds of millions of digits. Exponents and intermediate magnitudes are
therefore bounded, which is ample for any business calculation.

Comparisons (`0.48 <= 0.43`) and `and` / `or` / `not` are supported and
return true/false. Observed with qwen2.5:7b on the domain benchmarks: asked to
use the calculator for numeric work, the model repeatedly checked policy
thresholds this way - in one run seven calculator calls in a row, six of them
failing comparisons, until the step limit ended the run with no answer.
Booleans are kept out of arithmetic (`(1 < 2) + 1` is rejected), so a
comparison can never silently become the number 1.
"""

import ast
import operator
from typing import Any

from .base import Tool, ToolCategory, ToolResult

_MAX_EXPONENT = 100
_MAX_MAGNITUDE = 1e18

_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS: dict[type, Any] = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_COMPARISONS: dict[type, Any] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _checked(value: float) -> float:
    if abs(value) > _MAX_MAGNITUDE:
        raise ValueError(f"Intermediate result exceeds the supported magnitude ({_MAX_MAGNITUDE:g}).")
    return value


def _number(node: ast.AST) -> float:
    """Evaluate a node that must produce a number (not a boolean)."""
    value = _eval_node(node)
    if isinstance(value, bool):
        raise ValueError("A comparison result cannot be used in arithmetic.")
    return value


def _truth(node: ast.AST) -> bool:
    """Evaluate a node that must produce a boolean."""
    value = _eval_node(node)
    if not isinstance(value, bool):
        raise ValueError("'and', 'or' and 'not' need comparisons as operands, e.g. '(740 >= 650) and (0.28 <= 0.43)'.")
    return value


def _eval_node(node: ast.AST) -> float | bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return _checked(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _number(node.left), _number(node.right)
        # Bound the exponent *before* computing the power - checking the result
        # afterwards would be too late, the work is the problem.
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"Exponent {right:g} exceeds the supported limit of {_MAX_EXPONENT}.")
        return _checked(_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _checked(_UNARYOPS[type(node.op)](_number(node.operand)))
    if isinstance(node, ast.Compare) and all(type(op) in _COMPARISONS for op in node.ops):
        # Chained like Python: 1 < x <= 5 means (1 < x) and (x <= 5).
        left = _number(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _number(comparator)
            if not _COMPARISONS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        values = [_truth(value) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _truth(node.operand)
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def safe_eval(expression: str) -> float | bool:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate arithmetic or a numeric comparison, e.g. '5000 * 60 * 10', '(43 - 17)' or "
        "'(740 >= 650) and (0.48 <= 0.43)'. Supports + - * / // % **, parentheses, < <= > >= == != "
        "and and/or/not over comparisons. Numbers only - no variable names. Use this for any numeric "
        "computation rather than doing arithmetic yourself."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Arithmetic expression to evaluate"}},
        "required": ["expression"],
    }
    origin = "local"
    category = ToolCategory.CALCULATION

    async def run(self, expression: str) -> ToolResult:
        try:
            result = safe_eval(expression)
        except Exception as exc:  # noqa: BLE001 - any parse/eval failure is a normal tool failure, not a crash
            return ToolResult(ok=False, content="", error=f"Could not evaluate '{expression}': {exc}")
        # The result is recorded as evidence, which is what lets numerical
        # verification accept a derived figure the model computed here.
        return ToolResult(ok=True, content=f"{expression} = {result}", data={"expression": expression, "result": result})
