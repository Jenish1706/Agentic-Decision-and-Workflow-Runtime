"""Retrieval Budget & Search Efficiency experiment.

Question: when a single agent is allowed to search more freely, does the
extra searching buy answer quality - or does it mostly buy latency,
redundant retrieval, and a higher chance of exhausting the step budget
before answering at all?

The sharper version, which this experiment is actually designed to test:
the multi-agent path's RetrieverWorker searches EXACTLY ONCE per query, by
construction (see multi_agent/workers.py). So multi-agent already runs
under an implicit retrieval budget of 1. If a single agent capped at
`max_retrievals=1` performs like the multi-agent path, then the
multi-agent advantage measured in EVALUATION.md is substantially explained
by *retrieval constraint* rather than by orchestration or specialization.
That is a falsifiable claim, and this script is the falsification attempt.

Conditions: budget = 1, 2, 3, and None (unlimited, i.e. bounded only by
max_steps - the configuration this project originally shipped). Every
condition runs the identical benchmark and identical judge.

Run: uv run python scripts/run_budget_experiment.py
"""

import json
import time

from rich.console import Console
from rich.table import Table

from ..config import settings
from ..llm.ollama_client import OllamaClient
from ..runtime_factory import build_single_agent_runtime
from .agent_metrics import single_agent_failed, single_agent_tool_names
from .dataset import EvalQuery, load_benchmark
from .rag_metrics import score_citation_correctness, score_groundedness
from .retrieval_metrics import score_search_efficiency
from .run_eval import _aggregate, _judged
from .agent_metrics import count_unnecessary_tool_calls, score_task_success, score_tool_selection

console = Console()

BUDGETS: list[int | None] = [1, 2, 3, None]


def _label(budget: int | None) -> str:
    return "unlimited" if budget is None else str(budget)


async def run_condition(
    queries: list[EvalQuery], judge_llm: OllamaClient, budget: int | None
) -> tuple[dict, list[dict]]:
    per_query: list[dict] = []
    async with build_single_agent_runtime(with_memory=False, max_retrievals=budget) as runtime:
        for position, q in enumerate(queries, start=1):
            console.print(f"  [budget={_label(budget)} {position}/{len(queries)}] {q.id}")
            start = time.perf_counter()
            state = await runtime.run(q.question)
            latency = time.perf_counter() - start

            answer = state.final_answer or ""
            evidence_text = "\n\n".join(
                s.observation for s in state.steps if s.decision.action.value in ("retrieve", "call_tool")
            )
            task_success = await _judged(score_task_success(judge_llm, q, answer), default=None)
            grounded_result = await _judged(
                score_groundedness(judge_llm, q, answer, evidence_text), default=(None, [])
            )
            tool_names = single_agent_tool_names(state)
            efficiency = score_search_efficiency(state.retrieval_queries, state.retrieved_chunk_ids)

            per_query.append(
                {
                    "id": q.id,
                    "category": q.category,
                    "task_success": task_success,
                    "grounded": grounded_result[0],
                    "citation_correctness": score_citation_correctness(
                        state.citations, state.citation_sources, q.expected_doc_ids
                    ),
                    "tool_selection_correct": score_tool_selection(q.expected_tool, tool_names),
                    "unnecessary_tool_calls": count_unnecessary_tool_calls(q.expected_tool, tool_names),
                    "retrieval_calls": efficiency.retrieval_calls,
                    "unique_retrieval_queries": efficiency.unique_retrieval_queries,
                    "retrieval_chunks_consumed": efficiency.retrieval_chunks_consumed,
                    "retrieval_redundancy": efficiency.retrieval_redundancy,
                    "repeated_query_calls": efficiency.repeated_query_calls,
                    "step_count": len(state.steps),
                    "failed": single_agent_failed(state),
                    "latency_seconds": latency,
                }
            )
    return _aggregate(per_query), per_query


def print_report(results: dict[str, dict]) -> None:
    def fmt(v) -> str:
        if v is None:
            return "-"
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    table = Table(title="Retrieval Budget & Search Efficiency (single-agent)")
    table.add_column("Metric")
    for label in results:
        table.add_column(f"budget={label}")

    for key in (
        "task_success_rate",
        "groundedness_rate",
        "citation_correctness",
        "avg_retrieval_calls",
        "avg_unique_retrieval_queries",
        "avg_chunks_consumed",
        "avg_retrieval_redundancy",
        "total_repeated_query_calls",
        "avg_unnecessary_tool_calls",
        "avg_step_count",
        "failure_rate",
        "avg_latency_seconds",
    ):
        table.add_row(key, *[fmt(results[label].get(key)) for label in results])
    console.print(table)


async def main() -> None:
    queries = load_benchmark(settings.benchmark_path)
    judge_llm = OllamaClient(model=settings.ollama_model, host=settings.ollama_host)

    console.print(f"Retrieval budget experiment: {len(queries)} queries x {len(BUDGETS)} conditions")
    summaries: dict[str, dict] = {}
    detail: dict[str, list[dict]] = {}
    for budget in BUDGETS:
        label = _label(budget)
        console.print(f"\n[bold]Condition: retrieval budget = {label}[/bold]")
        summaries[label], detail[label] = await run_condition(queries, judge_llm, budget)

    print_report(summaries)

    out_path = settings.benchmark_path.parent / "budget_experiment_report.json"
    out_path.write_text(
        json.dumps({"summaries": summaries, "per_query": detail}, indent=2), encoding="utf-8"
    )
    console.print(f"\nFull report written to {out_path}")
