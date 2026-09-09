"""Human-readable execution traces, for debugging and for explaining the
system in an interview - not a production observability stack (no spans,
no exporters, just a clean sequential print of decisions and observations).
"""

from rich.console import Console

from .agent.state import AgentState
from .multi_agent.supervisor import MultiAgentResult

console = Console()


def _truncate(text: str, limit: int = 280) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def print_single_agent_trace(state: AgentState) -> None:
    console.rule(f"[bold]Query[/bold]: {state.query}")
    for step in state.steps:
        decision = step.decision
        header = f"[bold cyan]Step {step.index}[/bold cyan] action={decision.action.value} status={step.status.value}"
        console.print(header)
        console.print(f"  thought: {_truncate(decision.thought, 200)}")
        if decision.action.value == "retrieve":
            console.print(f"  query: {decision.query!r}")
        elif decision.action.value == "call_tool":
            console.print(f"  tool: {decision.tool_name}({decision.tool_arguments})")
        console.print(f"  observation: {_truncate(step.observation)}")
    console.rule("[bold green]Result[/bold green]")
    console.print(f"termination: {state.termination_reason}")
    console.print(f"citations: {state.citations}")
    console.print(state.final_answer)


def print_multi_agent_trace(result: MultiAgentResult) -> None:
    console.rule(f"[bold]Query[/bold]: {result.query}")
    console.print(f"[bold]plan[/bold]: needs_retrieval={result.plan.needs_retrieval} needs_tools={result.plan.needs_tools}")
    console.print(f"  reasoning: {_truncate(result.plan.thought, 200)}")
    for entry in result.worker_log:
        console.print(f"[bold cyan]{entry.worker}[/bold cyan]: {entry.summary}")
    console.rule("[bold green]Result[/bold green]")
    console.print(f"grounded: {result.grounded} (repair attempts: {result.repair_attempts})")
    if result.verification_issues:
        console.print(f"issues: {result.verification_issues}")
    console.print(f"citations: {result.citations}")
    console.print(result.final_answer)
