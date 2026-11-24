"""Cross-domain evaluation: one runtime, four business domains.

The same single-agent runtime (evidence ledger, numerical verification,
policy engine, audit trace) answers every benchmark question of every domain
under domains/. Each domain contributes only its MCP server and its policy
engine, both loaded through agent_runtime.domains.

Per domain it reports: answer verification rate, expected-number recall,
policy alignment (for questions that expect a decision), LLM-judged
groundedness, steps, evidence items and latency.

What changed from the earlier version of this script:
    - It imported `domains.*` modules, which only resolves when the
      repository root is on sys.path, so `python scripts/...` failed; domains
      are now loaded by path through the domain loader.
    - It ran a hand-picked 4-question sample per domain by default. The full
      benchmark runs by default now; use --sample N to limit it.
    - It reset memory and attached budgets by assigning private runtime
      attributes; it uses the public API (reset_session, run(budget=...)).
    - Policy alignment was averaged over ALL questions, counting every
      question that expects no decision as "aligned", which inflated the
      percentage. It is averaged over decision questions only now.
    - A judge call that failed to parse counted as grounded; such rows are
      now excluded and counted as unjudged.
    - "Numerical grounding" re-ran verification without the question's own
      numbers or the policy thresholds, so it disagreed with the loop's
      verdict. The loop's `answer_verified` is reported instead, alongside
      expected-number recall.

Usage:
    uv run python scripts/experiments/run_cross_domain_eval.py [--domain NAME] [--sample N]
"""

import argparse
import asyncio
import json
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from agent_runtime.agent.budget import allocate_budget, classify_task_complexity
from agent_runtime.config import REPO_ROOT, settings
from agent_runtime.domains import list_domains, load_domain
from agent_runtime.evaluation.agent_metrics import score_expected_numbers, single_agent_failed, single_agent_tool_names
from agent_runtime.evaluation.dataset import load_benchmark
from agent_runtime.evaluation.rag_metrics import score_groundedness
from agent_runtime.evaluation.run_eval import _judged
from agent_runtime.llm.ollama_client import OllamaClient
from agent_runtime.runtime_factory import build_single_agent_runtime

console = Console()
REPORT_PATH = REPO_ROOT / "data" / "cross_domain_evaluation_report.json"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


async def evaluate_domain(name: str, judge_llm: OllamaClient, sample: int | None) -> dict[str, Any]:
    domain = load_domain(name)
    queries = load_benchmark(domain.benchmark_path)[:sample]
    console.print(f"\n[bold blue]{domain.title}[/bold blue] ({len(queries)} questions)")

    rows: list[dict[str, Any]] = []
    async with build_single_agent_runtime(
        with_memory=False,
        policy_engine=domain.create_policy_engine(),
        server_modules=domain.server_modules,
        include_rag=False,
    ) as runtime:
        for position, query in enumerate(queries, start=1):
            console.print(f"  [{position}/{len(queries)}] {query.id}: {query.question[:65]}...")
            runtime.reset_session()
            budget = allocate_budget(classify_task_complexity(query.question), mode="adaptive")

            start = time.perf_counter()
            state = await runtime.run(query.question, budget=budget)
            latency = time.perf_counter() - start

            answer = state.final_answer or ""
            evidence_text = "\n\n".join(step.observation for step in state.steps if step.decision.action.value != "final")
            grounded, _ = await _judged(score_groundedness(judge_llm, query, answer, evidence_text), default=(None, []))
            evaluation = state.policy_evaluation

            rows.append({
                "id": query.id,
                "category": query.category,
                "steps": len(state.steps),
                "evidence_count": len(state.evidence_ledger),
                "evidence_sources": sorted({item.source.value for item in state.evidence_ledger.all()}),
                "answer_verified": state.answer_verified,
                "numerical_correctness": score_expected_numbers(query.expected_numbers, answer),
                "grounded": grounded,
                "policy_status": evaluation.final_status.value if evaluation is not None else None,
                "expected_policy_status": query.expected_policy_status,
                "policy_aligned": (
                    None
                    if query.expected_policy_status is None
                    else evaluation is not None and evaluation.final_status.value == query.expected_policy_status
                ),
                "tools_used": single_agent_tool_names(state),
                "failed": single_agent_failed(state),
                "latency_seconds": round(latency, 2),
                "final_answer": answer,
            })

    def rate(key: str) -> float | None:
        return _mean([1.0 if row[key] else 0.0 for row in rows if row[key] is not None])

    return {
        "title": domain.title,
        "questions": len(rows),
        "answer_verified_rate": rate("answer_verified"),
        "numerical_correctness": _mean([row["numerical_correctness"] for row in rows if row["numerical_correctness"] is not None]),
        "policy_alignment": rate("policy_aligned"),
        "policy_questions": sum(1 for row in rows if row["policy_aligned"] is not None),
        "groundedness": rate("grounded"),
        "unjudged_rows": sum(1 for row in rows if row["grounded"] is None),
        "failure_rate": rate("failed"),
        "avg_steps": _mean([float(row["steps"]) for row in rows]),
        "avg_evidence_items": _mean([float(row["evidence_count"]) for row in rows]),
        "avg_latency_seconds": _mean([row["latency_seconds"] for row in rows]),
        "rows": rows,
    }


def print_matrix(results: dict[str, dict[str, Any]]) -> None:
    table = Table(title="Cross-Domain Evaluation")
    columns = (
        ("Domain", "title"), ("Qs", "questions"), ("Verified", "answer_verified_rate"),
        ("Num. correct", "numerical_correctness"), ("Policy aligned", "policy_alignment"),
        ("Grounded", "groundedness"), ("Failures", "failure_rate"), ("Avg steps", "avg_steps"),
        ("Avg latency (s)", "avg_latency_seconds"),
    )
    for header, _ in columns:
        table.add_column(header, justify="left" if header == "Domain" else "right")
    for result in results.values():
        table.add_row(*(
            "-" if result[key] is None else f"{result[key]:.2f}" if isinstance(result[key], float) else str(result[key])
            for _, key in columns
        ))
    console.print(table)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", choices=list_domains(), help="evaluate one domain only")
    parser.add_argument("--sample", type=int, default=None, help="only the first N questions per domain")
    args = parser.parse_args()

    judge_llm = OllamaClient(model=settings.ollama_model, host=settings.ollama_host)
    results = {name: await evaluate_domain(name, judge_llm, args.sample) for name in ([args.domain] if args.domain else list_domains())}

    print_matrix(results)
    REPORT_PATH.write_text(
        json.dumps({"model": settings.ollama_model, "timestamp": time.time(), "domains": results}, indent=2), encoding="utf-8"
    )
    console.print(f"[green]Report written to {REPORT_PATH}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
