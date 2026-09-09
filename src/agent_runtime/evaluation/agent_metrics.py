"""Agent-side metrics: task success, tool selection accuracy, unnecessary
tool calls, and failure rate. Works against either single-agent (AgentState)
or multi-agent (MultiAgentResult) results via small adapter arguments rather
than two parallel metric modules."""

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


def single_agent_failed(state: AgentState) -> bool:
    return state.termination_reason != TerminationReason.FINAL_ANSWER


def single_agent_tool_names(state: AgentState) -> list[str]:
    return [step.decision.tool_name for step in state.tool_calls() if step.decision.tool_name]
