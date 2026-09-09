"""A local tool: registered directly, no protocol hop. Exists specifically
to contrast with the MCP-routed tools (agent/tools/mcp_tool.py) - same
`Tool` interface, same registry, but `origin='local'` means the registry
calls straight into this Python function instead of going through an MCP
client session.

Arithmetic is evaluated over a parsed AST with a fixed operator whitelist,
not `eval()` - the input is model-generated text, so treating it as trusted
code would be a real code-execution hole, not a theoretical one.
"""

import ast
import operator
from typing import Any

from .base import Tool, ToolResult

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


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate a basic arithmetic expression, e.g. '5000 * 60 * 10' or '(43 - 17)'. "
        "Supports + - * / // % ** and parentheses. Use this for any numeric computation "
        "rather than doing arithmetic yourself."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Arithmetic expression to evaluate"}},
        "required": ["expression"],
    }
    origin = "local"

    async def run(self, expression: str) -> ToolResult:
        try:
            result = safe_eval(expression)
        except Exception as exc:  # noqa: BLE001 - any parse/eval failure is a normal tool failure, not a crash
            return ToolResult(ok=False, content="", error=f"Could not evaluate '{expression}': {exc}")
        return ToolResult(ok=True, content=str(result), data={"result": result})
