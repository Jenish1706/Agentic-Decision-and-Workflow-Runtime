"""Product Analytics Action-Space & Budgeting Experiment.

QUESTION
    The retrieval-budget experiment (budget_experiment.py) found that capping
    retrieval changed task success and cut wasted exploration on document
    questions. Does the same kind of action-space constraint carry over to
    business tasks that mix SQL, statistics and policy decisions?

REGIMES (see agent/budget.py)
    unconstrained  generous ceilings only
    fixed          the same moderate caps for every question
    adaptive       caps chosen from each question's classified complexity

SETUP
    The 12 questions in domains/product_analytics/benchmark.json, answered by
    one single-agent runtime connected to the built-in knowledge base
    (retrieval + records) AND the product analytics MCP server, governed by
    that domain's policy engine. The knowledge base cannot answer these
    questions; it stays connected on purpose, because retrieval is part of
    the action space the budgets constrain (this matches the original design
    of the experiment).

MEASUREMENT FIXES - results from earlier versions are not comparable
    - "Numerical correctness" was `verify_numerical_grounding(answer).valid`
      - whether the answer's numbers exist somewhere in evidence, which an
      answer quoting the wrong real number passes. It is now the share of the
      benchmark's expected figures present in the answer
      (agent_metrics.score_expected_numbers). Grounding is reported
      separately as `answer_verified`, the loop's own verification result.
    - A judge call that failed to parse counted as task FAILURE and as
      GROUNDED. Unjudgeable rows are now excluded and counted instead.
    - Policy alignment ignored questions where the agent never proposed a
      decision, inflating the score; a missing evaluation now counts as
      misaligned.
    - "Abstention precision" was in fact recall (share of unanswerable
      questions the agent declined) and defaulted to 1.0 with no such
      questions. It is now named `abstention_recall` and is None then.
    - The rollout policy lived in this file, pooled p-values across
      experiments and ignored the direction of the effect; the domain's own
      policies.py is used now.

Usage:
    uv run python scripts/run_product_analytics_experiment.py
"""

import hashlib
import json
import statistics
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from ..agent.budget import BudgetMode, allocate_budget, classify_task_complexity
from ..agent.state import AgentState
from ..config import REPO_ROOT, settings
from ..domains import load_domain
from ..llm.ollama_client import OllamaClient
from ..mcp_layer.client import DEFAULT_SERVER_MODULES
from ..runtime_factory import build_single_agent_runtime
from .agent_metrics import score_expected_numbers, score_task_success, single_agent_failed, single_agent_tool_names
from .dataset import EvalQuery, load_benchmark
from .rag_metrics import score_groundedness
from .run_eval import _judged

console = Console()

DOMAIN = "product_analytics"
REGIMES: tuple[BudgetMode, ...] = (BudgetMode.UNCONSTRAINED, BudgetMode.FIXED, BudgetMode.ADAPTIVE)
REPORT_PATH = REPO_ROOT / "data" / "product_analytics_experiment_report.json"

_ANALYTICS_TOOLS = frozenset({"execute_sql", "describe_tables", "test_significance", "calculator"})
_ABSTENTION_CUES = (
    "not available",
    "insufficient",
    "cannot be determined",
    "does not contain",
    "not recorded",
    "does not have",
    "no data",
    "does not record",
)


def abstained(answer: str) -> bool:
    """Whether an answer declines or reports missing data (a phrase check)."""
    lowered = answer.lower()
    return any(cue in lowered for cue in _ABSTENTION_CUES)


def count_unnecessary_tool_calls(query: EvalQuery, state: AgentState) -> int:
    """Tool calls that repeat an identical earlier call, plus calls to tools
    outside the analytics family when the question expects an analytics
    tool. Legitimate detours - describe_tables before a query, the
    calculator to check a difference - are not counted."""
    unnecessary = 0
    seen: set[str] = set()
    for step in state.tool_calls():
        name = step.decision.tool_name or ""
        signature = f"{name}:{json.dumps(step.decision.tool_arguments, sort_keys=True, default=str)}"
        if signature in seen:
            unnecessary += 1
            continue
        seen.add(signature)
        if query.expected_tool in _ANALYTICS_TOOLS and name not in _ANALYTICS_TOOLS:
            unnecessary += 1
    return unnecessary


def evidence_fingerprint(item) -> str:
    """Content identity of an evidence item (source, kind and payload), so
    two tool calls that returned the same data count as duplicate evidence
    even if their arguments were spelled differently."""
    payload = json.dumps(item.to_dict()["data"], sort_keys=True, default=str)
    return hashlib.sha256(f"{item.source.value}:{item.kind.value}:{payload}".encode()).hexdigest()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float | None:
        return _mean([1.0 if row[key] else 0.0 for row in rows if row[key] is not None])

    latencies = [row["latency_seconds"] for row in rows]
    return {
        "task_success": rate("task_success"),
        "groundedness": rate("grounded"),
        "unjudged_rows": sum(1 for row in rows if row["task_success"] is None or row["grounded"] is None),
        "numerical_correctness": _mean([row["numerical_correctness"] for row in rows if row["numerical_correctness"] is not None]),
        "answer_verified": rate("answer_verified"),
        "policy_alignment": rate("policy_aligned"),
        "abstention_recall": rate("abstained"),
        "avg_steps": _mean([float(row["step_count"]) for row in rows]),
        "sql_queries_total": sum(row["sql_queries"] for row in rows),
        "statistical_tests_total": sum(row["statistical_tests"] for row in rows),
        "retrieval_calls_total": sum(row["retrieval_calls"] for row in rows),
        "tool_calls_total": sum(row["tool_calls"] for row in rows),
        "unnecessary_tool_calls_total": sum(row["unnecessary_tool_calls"] for row in rows),
        "evidence_efficiency": _mean([row["evidence_efficiency"] for row in rows]),
        "budget_utilization": _mean([row["budget_utilization"] for row in rows]),
        "failure_rate": rate("failed"),
        "latency_mean": _mean(latencies),
        "latency_median": statistics.median(latencies) if latencies else None,
        "latency_stdev": statistics.stdev(latencies) if len(latencies) > 1 else None,
        "n": len(rows),
    }


async def run_regime(queries: list[EvalQuery], judge_llm: OllamaClient, regime: BudgetMode) -> tuple[dict, list[dict]]:
    domain = load_domain(DOMAIN)
    rows: list[dict[str, Any]] = []
    async with build_single_agent_runtime(
        with_memory=False,
        policy_engine=domain.create_policy_engine(),
        server_modules={**DEFAULT_SERVER_MODULES, **domain.server_modules},
    ) as runtime:
        for position, query in enumerate(queries, start=1):
            console.print(f"  [{regime.value} {position}/{len(queries)}] {query.id}: {query.question[:60]}...")
            # Independent questions: no conversation carried over between them.
            runtime.reset_session()
            budget = allocate_budget(classify_task_complexity(query.question), mode=regime)

            start = time.perf_counter()
            state = await runtime.run(query.question, budget=budget)
            latency = time.perf_counter() - start

            answer = state.final_answer or ""
            evidence_text = "\n\n".join(step.observation for step in state.steps if step.decision.action.value != "final")
            task_success = await _judged(score_task_success(judge_llm, query, answer), default=None)
            grounded, _ = await _judged(score_groundedness(judge_llm, query, answer, evidence_text), default=(None, []))

            items = state.evidence_ledger.all()
            unique_items = len({evidence_fingerprint(item) for item in items})
            tool_names = single_agent_tool_names(state)
            evaluation = state.policy_evaluation

            rows.append({
                "id": query.id,
                "category": query.category,
                "difficulty": query.difficulty,
                "task_success": task_success,
                "grounded": grounded,
                "numerical_correctness": score_expected_numbers(query.expected_numbers, answer),
                "answer_verified": state.answer_verified,
                # None only when the benchmark expects no decision; a missing
                # evaluation for a question that expects one is a miss.
                "policy_aligned": (
                    None
                    if query.expected_policy_status is None
                    else evaluation is not None and evaluation.final_status.value == query.expected_policy_status
                ),
                "policy_status": evaluation.final_status.value if evaluation is not None else None,
                "abstained": abstained(answer) if query.category == "insufficient_evidence" else None,
                "step_count": len(state.steps),
                "sql_queries": state.budget_usage.get("sql_queries_used", 0),
                "statistical_tests": tool_names.count("test_significance"),
                "retrieval_calls": len(state.retrieval_queries),
                "tool_calls": len(tool_names),
                "unnecessary_tool_calls": count_unnecessary_tool_calls(query, state),
                "evidence_items": len(items),
                "evidence_efficiency": unique_items / len(items) if items else 1.0,
                "budget": budget.to_dict(),
                "budget_utilization": len(state.steps) / budget.max_steps,
                "failed": single_agent_failed(state),
                "latency_seconds": round(latency, 2),
            })
    return _aggregate(rows), rows


def print_report(summaries: dict[str, dict[str, Any]]) -> None:
    table = Table(title="Product Analytics: Action-Space Budgeting")
    table.add_column("Metric", style="cyan")
    for regime in summaries:
        table.add_column(regime, justify="right")
    for key in (
        "task_success", "numerical_correctness", "answer_verified", "policy_alignment", "abstention_recall",
        "groundedness", "unjudged_rows", "avg_steps", "sql_queries_total", "retrieval_calls_total",
        "unnecessary_tool_calls_total", "evidence_efficiency", "budget_utilization", "failure_rate",
        "latency_mean", "latency_median", "latency_stdev",
    ):
        values = [summaries[regime].get(key) for regime in summaries]
        table.add_row(key, *("-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v) for v in values))
    console.print(table)


async def main() -> None:
    queries = load_benchmark(load_domain(DOMAIN).benchmark_path)
    judge_llm = OllamaClient(model=settings.ollama_model, host=settings.ollama_host)
    console.print(f"[bold]{len(queries)} questions x {len(REGIMES)} regimes[/bold]")

    summaries: dict[str, dict[str, Any]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    for regime in REGIMES:
        console.print(f"\n[bold green]Regime: {regime.value}[/bold green]")
        summaries[regime.value], details[regime.value] = await run_regime(queries, judge_llm, regime)
        # Written after every regime so an interrupted run keeps what it measured.
        REPORT_PATH.write_text(
            json.dumps({"model": settings.ollama_model, "timestamp": time.time(), "summaries": summaries, "details": details}, indent=2),
            encoding="utf-8",
        )

    print_report(summaries)
    console.print(f"[green]Report saved to {REPORT_PATH}[/green]")
