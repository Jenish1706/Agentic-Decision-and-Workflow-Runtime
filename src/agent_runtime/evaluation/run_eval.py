"""Runs the benchmark (data/benchmark.json) through single-agent and
multi-agent, plus a standalone RAG-only retrieval pass, and reports:
retrieval quality (naive vs reranked), groundedness, citation correctness,
task success, tool selection accuracy, unnecessary tool calls, step count,
latency, and failure rate - single-agent vs multi-agent side by side.

Requires Ollama running locally. Takes a few minutes for the full ~20-query
benchmark across both modes, since every row makes several real model calls.
"""

import asyncio
import json
import time
from statistics import mean

from rich.console import Console
from rich.table import Table

from ..config import settings
from ..llm.base import LLMError
from ..llm.ollama_client import OllamaClient
from ..rag.ingest import build_rag_pipeline
from ..runtime_factory import build_multi_agent_supervisor, build_single_agent_runtime
from .agent_metrics import (
    count_unnecessary_tool_calls,
    score_task_success,
    score_tool_selection,
    single_agent_failed,
    single_agent_tool_names,
)
from .dataset import EvalQuery, load_benchmark
from .rag_metrics import (
    score_citation_correctness,
    score_groundedness,
    score_retrieval,
    score_retrieval_naive,
)
from .retrieval_metrics import score_search_efficiency

console = Console()


def _avg(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


async def _judged(coro, *, default):
    """Runs one LLM-judge call, surviving a judge that can't produce
    schema-valid output. A full benchmark run is a few hundred model calls;
    letting a single unparseable judgement abort it would throw away every
    row already measured. `None` (not `False`) is the default for scores so
    an unjudgeable row is excluded from the average rather than silently
    counted as a failure - the agent didn't fail, the judge did.
    """
    try:
        return await coro
    except LLMError:
        return default


def run_retrieval_metrics(queries: list[EvalQuery]) -> dict:
    rag = build_rag_pipeline()
    reranked = [score_retrieval(rag, q) for q in queries]
    naive = [score_retrieval_naive(rag, q) for q in queries]
    return {
        "reranked_recall_at_k": _avg([r.recall_at_k for r in reranked]),
        "reranked_mrr": _avg([r.reciprocal_rank for r in reranked]),
        "naive_recall_at_k": _avg([r.recall_at_k for r in naive]),
        "naive_mrr": _avg([r.reciprocal_rank for r in naive]),
    }


def _aggregate(rows: list[dict]) -> dict:
    return {
        # `is not None` guards below matter: a row the judge couldn't score is
        # excluded from the average, never counted as a 0 (see `_judged`).
        "task_success_rate": _avg([1.0 if r["task_success"] else 0.0 for r in rows if r["task_success"] is not None]),
        "groundedness_rate": _avg([1.0 if r["grounded"] else 0.0 for r in rows if r["grounded"] is not None]),
        "unjudged_rows": sum(1 for r in rows if r["task_success"] is None or r["grounded"] is None),
        "citation_correctness": _avg([r["citation_correctness"] for r in rows]),
        "tool_selection_accuracy": _avg(
            [1.0 if r["tool_selection_correct"] else 0.0 for r in rows if r["tool_selection_correct"] is not None]
        ),
        "avg_unnecessary_tool_calls": _avg([float(r["unnecessary_tool_calls"]) for r in rows]),
        "avg_step_count": _avg([float(r["step_count"]) for r in rows]),
        # Search efficiency (see evaluation/retrieval_metrics.py)
        "avg_retrieval_calls": _avg([float(r["retrieval_calls"]) for r in rows]),
        "avg_unique_retrieval_queries": _avg([float(r["unique_retrieval_queries"]) for r in rows]),
        "avg_chunks_consumed": _avg([float(r["retrieval_chunks_consumed"]) for r in rows]),
        "avg_retrieval_redundancy": _avg([r["retrieval_redundancy"] for r in rows]),
        "total_repeated_query_calls": sum(r["repeated_query_calls"] for r in rows),
        "failure_rate": _avg([1.0 if r["failed"] else 0.0 for r in rows]),
        "avg_latency_seconds": _avg([r["latency_seconds"] for r in rows]),
        "n": len(rows),
    }


async def run_single_agent(queries: list[EvalQuery], judge_llm: OllamaClient) -> tuple[dict, list[dict]]:
    per_query = []
    async with build_single_agent_runtime(with_memory=False) as runtime:
        for position, q in enumerate(queries, start=1):
            console.print(f"  [single-agent {position}/{len(queries)}] {q.id}: {q.question[:60]}...")
            # Every benchmark question is independent. Without this reset the
            # runtime's short-term memory put earlier questions and answers
            # into every later prompt, so results published before this fix
            # were measured on contaminated inputs.
            runtime.reset_session()
            start = time.perf_counter()
            state = await runtime.run(q.question)
            latency = time.perf_counter() - start

            answer = state.final_answer or ""
            evidence_text = "\n\n".join(
                s.observation for s in state.steps if s.decision.action.value in ("retrieve", "call_tool")
            )
            task_success = await _judged(score_task_success(judge_llm, q, answer), default=None)
            if task_success and not state.answer_verified:
                task_success = False
            grounded_result = await _judged(
                score_groundedness(judge_llm, q, answer, evidence_text), default=(None, [])
            )
            grounded = grounded_result[0]
            tool_names = single_agent_tool_names(state)
            efficiency = score_search_efficiency(state.retrieval_queries, state.retrieved_chunk_ids)

            per_query.append(
                {
                    "id": q.id,
                    "category": q.category,
                    "retrieval_calls": efficiency.retrieval_calls,
                    "unique_retrieval_queries": efficiency.unique_retrieval_queries,
                    "retrieval_chunks_consumed": efficiency.retrieval_chunks_consumed,
                    "retrieval_redundancy": efficiency.retrieval_redundancy,
                    "repeated_query_calls": efficiency.repeated_query_calls,
                    "task_success": task_success,
                    "grounded": grounded,
                    "citation_correctness": score_citation_correctness(
                        state.citations, state.citation_sources, q.expected_doc_ids
                    ),
                    "tool_selection_correct": score_tool_selection(q.expected_tool, tool_names),
                    "unnecessary_tool_calls": count_unnecessary_tool_calls(q.expected_tool, tool_names),
                    "step_count": len(state.steps),
                    "failed": single_agent_failed(state),
                    "latency_seconds": latency,
                }
            )
    return _aggregate(per_query), per_query


async def run_multi_agent(queries: list[EvalQuery], judge_llm: OllamaClient) -> tuple[dict, list[dict]]:
    per_query = []
    async with build_multi_agent_supervisor() as supervisor:
        for position, q in enumerate(queries, start=1):
            console.print(f"  [multi-agent {position}/{len(queries)}] {q.id}: {q.question[:60]}...")
            start = time.perf_counter()
            result = await supervisor.run(q.question)
            latency = time.perf_counter() - start

            task_success = await _judged(score_task_success(judge_llm, q, result.final_answer), default=None)
            if task_success and not result.grounded:
                task_success = False
            efficiency = score_search_efficiency(result.retrieval_queries, result.retrieved_chunk_ids)
            per_query.append(
                {
                    "id": q.id,
                    "category": q.category,
                    "retrieval_calls": efficiency.retrieval_calls,
                    "unique_retrieval_queries": efficiency.unique_retrieval_queries,
                    "retrieval_chunks_consumed": efficiency.retrieval_chunks_consumed,
                    "retrieval_redundancy": efficiency.retrieval_redundancy,
                    "repeated_query_calls": efficiency.repeated_query_calls,
                    "task_success": task_success,
                    "grounded": result.grounded,
                    "citation_correctness": score_citation_correctness(
                        result.citations, result.citation_sources, q.expected_doc_ids
                    ),
                    "tool_selection_correct": score_tool_selection(q.expected_tool, result.tools_called),
                    "unnecessary_tool_calls": count_unnecessary_tool_calls(q.expected_tool, result.tools_called),
                    "step_count": len(result.worker_log),
                    "failed": not result.grounded,
                    "latency_seconds": latency,
                }
            )
    return _aggregate(per_query), per_query


def print_report(retrieval: dict, single: dict, multi: dict) -> None:
    def fmt(v: float | None) -> str:
        return "-" if v is None else f"{v:.2f}"

    table = Table(title="Agentic RAG Runtime - Evaluation Report")
    table.add_column("Metric")
    table.add_column("Single-agent")
    table.add_column("Multi-agent")
    for key in (
        "task_success_rate",
        "groundedness_rate",
        "citation_correctness",
        "tool_selection_accuracy",
        "avg_unnecessary_tool_calls",
        "avg_step_count",
        "avg_retrieval_calls",
        "avg_unique_retrieval_queries",
        "avg_chunks_consumed",
        "avg_retrieval_redundancy",
        "total_repeated_query_calls",
        "failure_rate",
        "avg_latency_seconds",
        "unjudged_rows",
    ):
        table.add_row(key, fmt(single.get(key)), fmt(multi.get(key)))
    console.print(table)

    console.print(
        f"\n[bold]Retrieval[/bold] (naive -> reranked) - "
        f"Recall@k: {fmt(retrieval['naive_recall_at_k'])} -> {fmt(retrieval['reranked_recall_at_k'])}   "
        f"MRR: {fmt(retrieval['naive_mrr'])} -> {fmt(retrieval['reranked_mrr'])}"
    )


async def main() -> None:
    queries = load_benchmark(settings.benchmark_path)
    judge_llm = OllamaClient(model=settings.ollama_model, host=settings.ollama_host)

    console.print(f"Loaded {len(queries)} benchmark queries. Scoring retrieval...")
    retrieval = run_retrieval_metrics(queries)

    console.print("Running single-agent mode over the benchmark...")
    single_summary, single_rows = await run_single_agent(queries, judge_llm)

    console.print("Running multi-agent mode over the benchmark...")
    multi_summary, multi_rows = await run_multi_agent(queries, judge_llm)

    print_report(retrieval, single_summary, multi_summary)

    report = {
        "retrieval": retrieval,
        "single_agent": {"summary": single_summary, "per_query": single_rows},
        "multi_agent": {"summary": multi_summary, "per_query": multi_rows},
    }
    out_path = settings.benchmark_path.parent / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
