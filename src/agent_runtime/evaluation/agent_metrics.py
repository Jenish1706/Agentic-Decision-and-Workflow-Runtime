"""Agent-side metrics: task success, tool selection, unnecessary tool calls,
expected-number recall, and failure rate. They work for single-agent
(AgentState) and multi-agent (MultiAgentResult) runs through small adapter
arguments rather than two parallel metric modules."""

import re

from ..agent.llm_judge import llm_judge_task_success
from ..agent.state import AgentState, TerminationReason
from ..llm.base import LLMClient
from .dataset import EvalQuery


async def score_task_success(llm: LLMClient, query: EvalQuery, actual_answer: str) -> bool:
    verdict = await llm_judge_task_success(llm, query.question, query.expected_answer, actual_answer)
    return verdict.success


def score_tool_selection(expected_tool: str | None, tool_names_called: list[str]) -> bool | None:
    """None when the question didn't call for any particular tool (pure
    document lookup, out-of-corpus) - there's no 'correct' tool to check
    against, so this query shouldn't move the tool-selection-accuracy
    average in either direction."""
    if expected_tool is None:
        return None
    return expected_tool in tool_names_called


def count_unnecessary_tool_calls(expected_tool: str | None, tool_names_called: list[str]) -> int:
    """A tool call is 'unnecessary' when the question didn't need any tool
    at all (expected_tool is None) but the agent made one anyway. When a
    tool genuinely was expected, calling it isn't unnecessary even if the
    agent also tried something else first - that's a different failure
    mode (inefficiency), not this metric."""
    if expected_tool is not None:
        return 0
    return len(tool_names_called)


# A number as written in prose. The lookbehind skips digits glued to an
# identifier or word ("ACCT-102", "Q3", "INV-2026-004"), which are labels,
# not figures.
_WRITTEN_NUMBER = re.compile(r"(?<![A-Za-z0-9_.\-])(\d[\d,]*(?:\.\d+)?)")


def _decimals(number_text: str) -> int:
    return len(number_text.split(".", 1)[1]) if "." in number_text else 0


def score_expected_numbers(expected_numbers: list[float], answer: str) -> float | None:
    """Fraction of the benchmark's expected figures that appear in the answer.

    This is what "numerical correctness" means in the evaluation reports, and
    it replaced a metric that did not measure correctness at all: the old
    harnesses reported `verify_numerical_grounding(answer).valid`, i.e.
    whether the answer's numbers exist somewhere in evidence. An answer that
    quotes the wrong (but real) row scores perfectly on that; it does not
    here.

    A figure matches a number written in the answer at the coarser of the
    two precisions: an expected 28.0 matches "28%", an expected 2.9 matches
    "2.93%", but an expected 0.015 does not match "0.014". None when the
    question has no expected figures (excluded from averages, not scored 1.0).
    """
    if not expected_numbers:
        return None
    written = [(float(text.replace(",", "")), _decimals(text)) for text in _WRITTEN_NUMBER.findall(answer)]

    def present(target: float) -> bool:
        target_decimals = _decimals(f"{target:.10f}".rstrip("0").rstrip("."))
        return any(
            abs(value - target) <= 0.5 * 10 ** -min(decimals, target_decimals) + 1e-9 for value, decimals in written
        )

    return sum(1 for target in expected_numbers if present(target)) / len(expected_numbers)


def single_agent_failed(state: AgentState) -> bool:
    """A run failed if it did not end with a final answer, or ended with one
    that did not pass verification (the loop's last-step best-effort
    acceptance). Counting only the termination reason used to score those
    unverified answers as successes."""
    return state.termination_reason != TerminationReason.FINAL_ANSWER or state.answer_verified is False


def single_agent_tool_names(state: AgentState) -> list[str]:
    return [step.decision.tool_name for step in state.tool_calls() if step.decision.tool_name]
