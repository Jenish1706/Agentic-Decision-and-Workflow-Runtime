"""Product Analytics Action-Space & Budgeting Experiment (Phase 1.5).

Central Hypothesis:
In our previous retrieval-budget experiment on pure RAG, limiting retrieval
consumption affected task success and reduced unnecessary exploration. Phase 1.5
tests whether this action-space constraint generalizes to heterogeneous business
decision tasks combining SQL, statistical analysis, and policy constraints.

Experimental Regimes:
1. Unconstrained: Open exploration; agent can freely search and query without proactive limits.
2. Fixed: Uniform resource caps across all task types.
3. Adaptive: Dynamic limits allocated by deterministic task complexity classification.

Benchmark:
12 analytical tasks from domains/product_analytics/benchmark.json spanning:
- Aggregation (pa-01, pa-02)
- Segmentation & Diagnosis (pa-03, pa-04, pa-05, pa-11)
- Experiment Analysis & Hypothesis Testing (pa-06, pa-07)
- Multi-hop Root Cause & Decision (pa-08, pa-09, pa-10, pa-12)

Usage:
uv run python scripts/run_product_analytics_experiment.py
"""

import asyncio
import json
from pathlib import Path
import statistics
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from ..agent.budget import (
    ActionBudget,
    BudgetController,
    allocate_budget,
    classify_task_complexity,
)
from ..agent.policy import (
    DecisionContext,
    PolicyEngine,
    PolicyRuleResult,
    PolicyStatus,
    ProposedDecision,
)
from ..agent.verification import verify_numerical_grounding
from ..config import REPO_ROOT, settings
from ..llm.ollama_client import OllamaClient
from ..runtime_factory import build_single_agent_runtime
from .agent_metrics import single_agent_failed, single_agent_tool_names
from .dataset import EvalQuery, load_benchmark
from .rag_metrics import score_citation_correctness, score_groundedness
from .run_eval import _aggregate, _judged
from .agent_metrics import score_task_success

console = Console()

REGIMES: list[str] = ["unconstrained", "fixed", "adaptive"]
BENCHMARK_PATH = REPO_ROOT / "domains" / "product_analytics" / "benchmark.json"
REPORT_PATH = REPO_ROOT / "data" / "product_analytics_experiment_report.json"


def create_product_analytics_policy_engine() -> PolicyEngine:
    """Builds the deterministic governance rules declared in domain_manifest.yaml."""

    def statistical_significance_gate(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        facts = ctx.evidence_ledger.extract_numerical_facts() if ctx.evidence_ledger else {}
        p_vals = [v for k, v in facts.items() if "p_value" in k]
        if not p_vals or any(p >= 0.05 for p in p_vals):
            return PolicyRuleResult(
                rule_name="statistical_significance_gate",
                passed=False,
                status=PolicyStatus.BLOCKED,
                message="Cannot recommend rollout without verified statistical significance (p < 0.05).",
            )
        return PolicyRuleResult(
            rule_name="statistical_significance_gate",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="Statistical significance verified (p < 0.05).",
        )

    def high_impact_rollout_approval(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        traffic = proposal.parameters.get("traffic_percentage", 0)
        if traffic > 50:
            return PolicyRuleResult(
                rule_name="high_impact_rollout_approval",
                passed=False,
                status=PolicyStatus.REQUIRES_APPROVAL,
                message=f"Rollout allocation of {traffic}% exceeds autonomous threshold (50%). Human approval required.",
            )
        return PolicyRuleResult(
            rule_name="high_impact_rollout_approval",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message=f"Rollout allocation of {traffic}% is within safe automated limits.",
        )

    return PolicyEngine(rules=[statistical_significance_gate, high_impact_rollout_approval])


def evaluate_abstention(query: EvalQuery, answer: str) -> bool:
    """Detects whether the agent properly abstained or identified missing data."""
    if query.category != "insufficient_evidence":
        return True
    lower = answer.lower()
    abstention_cues = (
        "not available",
        "insufficient",
        "cannot be determined",
        "does not contain",
        "not recorded",
        "does not have",
        "no data",
    )
    return any(cue in lower for cue in abstention_cues)


def count_categorized_tools(tool_names: list[str]) -> dict[str, int]:
    sql = sum(1 for t in tool_names if t in ("execute_sql", "describe_tables"))
    stats = sum(1 for t in tool_names if t == "test_significance")
    records = sum(1 for t in tool_names if t in ("lookup_record", "search_records"))
    calc = sum(1 for t in tool_names if t == "calculator")
    return {
        "sql_calls": sql,
        "statistical_calls": stats,
        "record_calls": records,
        "calculator_calls": calc,
        "total_tool_calls": len(tool_names),
    }


def compute_evidence_fingerprint(item) -> str:
    """Computes a stable content fingerprint for an EvidenceItem based on its
    source, kind, and substantive data payload (not just the raw query string).
    """
    import hashlib

    try:
        data_str = json.dumps(item.data, sort_keys=True, default=str)
    except Exception:
        data_str = str(item.data)
    sig = f"{item.source.value}:{item.kind.value}:{data_str}"
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()


def compute_unnecessary_tool_calls(query: EvalQuery, state) -> int:
    """A tool invocation is unnecessary when the benchmark's ground-truth solution
    does not require the information returned by that invocation and the same task
    could be completed using evidence already available to the agent.

    Legitimate alternative paths (e.g. describe_tables prior to execute_sql, or calculator
    to verify a difference) are preserved as valid engineering steps.
    """
    tool_steps = [s for s in state.steps if s.decision.action.value == "call_tool"]
    unnecessary = 0
    seen_calls: set[str] = set()

    for step in tool_steps:
        tool_name = step.decision.tool_name or ""
        try:
            args_str = json.dumps(step.decision.tool_arguments, sort_keys=True)
        except Exception:
            args_str = str(step.decision.tool_arguments)

        # 1. Repeated identical tool call with same arguments is unnecessary
        call_sig = f"{tool_name}:{args_str}"
        if call_sig in seen_calls:
            unnecessary += 1
            continue
        seen_calls.add(call_sig)

        # 2. Domain-extraneous tool calls
        if query.expected_tool in ("execute_sql", "test_significance"):
            allowed_family = {"execute_sql", "describe_tables", "test_significance", "calculator"}
            if tool_name not in allowed_family:
                unnecessary += 1
        elif query.category == "insufficient_evidence":
            # One exploratory schema/data lookup is legitimate to check telemetry;
            # extra subsequent tool calls after finding no relevant columns are unnecessary.
            if len(seen_calls) > 2:
                unnecessary += 1
        elif query.category == "adversarial_safety":
            if tool_name not in ("execute_sql",):
                unnecessary += 1

    return unnecessary


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _aggregate_analytics_experiment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_success_vals = [1.0 if r["task_success"] else 0.0 for r in rows if r["task_success"] is not None]
    grounded_vals = [1.0 if r["grounded"] else 0.0 for r in rows if r["grounded"] is not None]
    num_vals = [r["numerical_correctness"] for r in rows if r["numerical_correctness"] is not None]
    pol_vals = [1.0 if r["policy_aligned"] else 0.0 for r in rows if r["policy_aligned"] is not None]
    abst_vals = [1.0 if r["abstention_correct"] else 0.0 for r in rows if r["category"] == "insufficient_evidence"]

    return {
        "task_success": _avg(task_success_vals),
        "grounded": _avg(grounded_vals),
        "numerical_correctness": _avg(num_vals),
        "policy_alignment": _avg(pol_vals) if pol_vals else None,
        "abstention_precision": _avg(abst_vals) if abst_vals else 1.0,
        "step_count": round(_avg([float(r["step_count"]) for r in rows]), 2),
        "sql_calls_total": sum(r["sql_calls"] for r in rows),
        "statistical_calls_total": sum(r["statistical_calls"] for r in rows),
        "total_tool_calls": sum(r["total_tool_calls"] for r in rows),
        "unnecessary_tool_calls_total": sum(r["unnecessary_tool_calls"] for r in rows),
        "evidence_efficiency_avg": round(_avg([r["evidence_efficiency"] for r in rows]), 3),
        "budget_utilization": round(_avg([r["budget_utilization"] for r in rows]), 3),
        "failure_rate": _avg([1.0 if r["failed"] else 0.0 for r in rows]),
        "latency_seconds": round(_avg([r["latency_seconds"] for r in rows]), 2),
        "latency_median": round(statistics.median(r["latency_seconds"] for r in rows), 2),
        "latency_spread": round(
            statistics.stdev(r["latency_seconds"] for r in rows) if len(rows) > 1 else 0.0, 2
        ),
        "n": len(rows),
    }


async def run_regime(
    queries: list[EvalQuery],
    judge_llm: OllamaClient,
    regime: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_query: list[dict[str, Any]] = []
    policy_engine = create_product_analytics_policy_engine()
    async with build_single_agent_runtime(
        with_memory=False,
        policy_engine=policy_engine,
    ) as runtime:
        for position, q in enumerate(queries, start=1):
            console.print(f"  [{regime.upper()} {position}/{len(queries)}] {q.id}: {q.question[:60]}...")
            runtime._short_term.clear()

            # Determine budget controller for this task
            complexity = classify_task_complexity(q.question)
            budget = allocate_budget(complexity, mode=regime)
            controller = BudgetController(budget=budget)
            runtime._budget_controller = controller

            start = time.perf_counter()
            state = await runtime.run(q.question)
            latency = time.perf_counter() - start

            answer = state.final_answer or ""
            evidence_text = "\n\n".join(
                s.observation for s in state.steps if s.decision.action.value in ("retrieve", "call_tool")
            )

            # Primary evaluations
            task_success = await _judged(score_task_success(judge_llm, q, answer), default=False)
            grounded_result = await _judged(
                score_groundedness(judge_llm, q, answer, evidence_text), default=(True, [])
            )

            # Numerical verification
            num_ver = verify_numerical_grounding(answer, state.evidence_ledger)
            numerical_correct = 1.0 if num_ver.valid else 0.0

            # Policy alignment
            policy_aligned = None
            if q.expected_policy_status is not None and state.policy_evaluation is not None:
                policy_aligned = (state.policy_evaluation.final_status.value == q.expected_policy_status)

            # Tool breakdown & efficiency
            tool_names = single_agent_tool_names(state)
            tool_counts = count_categorized_tools(tool_names)
            unnecessary = compute_unnecessary_tool_calls(q, state)

            # Evidence Ledger efficiency based on content fingerprint
            total_ev = len(state.evidence_ledger)
            unique_fingerprints = {compute_evidence_fingerprint(item) for item in state.evidence_ledger.all()}
            unique_ev = len(unique_fingerprints)
            dup_ev = total_ev - unique_ev
            ev_efficiency = round(unique_ev / total_ev, 3) if total_ev > 0 else 1.0

            per_query.append(
                {
                    "id": q.id,
                    "category": q.category,
                    "difficulty": q.difficulty,
                    "task_success": bool(task_success),
                    "numerical_correctness": numerical_correct,
                    "policy_aligned": policy_aligned,
                    "abstention_correct": evaluate_abstention(q, answer),
                    "grounded": bool(grounded_result[0]) if grounded_result[0] is not None else True,
                    "citation_correctness": score_citation_correctness(
                        state.citations, state.citation_sources, q.expected_doc_ids
                    ),
                    "sql_calls": tool_counts["sql_calls"],
                    "statistical_calls": tool_counts["statistical_calls"],
                    "total_tool_calls": tool_counts["total_tool_calls"],
                    "unnecessary_tool_calls": unnecessary,
                    "retrieval_calls": len(state.retrieval_queries),
                    "step_count": len(state.steps),
                    "total_evidence_items": total_ev,
                    "unique_evidence_items": unique_ev,
                    "duplicate_evidence": dup_ev,
                    "evidence_efficiency": ev_efficiency,
                    "budget_steps_allowed": budget.max_steps,
                    "budget_utilization": round(len(state.steps) / budget.max_steps, 3) if budget.max_steps > 0 else 1.0,
                    "failed": single_agent_failed(state),
                    "latency_seconds": round(latency, 2),
                }
            )

    agg = _aggregate_analytics_experiment(per_query)
    return agg, per_query


def print_experiment_report(results: dict[str, dict[str, Any]]) -> None:
    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    table = Table(title="Phase 1.5: Product Analytics Action-Space & Budgeting Experiment")
    table.add_column("Metric", style="cyan")
    for r in REGIMES:
        table.add_column(r.capitalize(), justify="right")

    metrics_to_show = [
        ("Task Success Rate", "task_success"),
        ("Numerical Correctness", "numerical_correctness"),
        ("Abstention Precision", "abstention_precision"),
        ("Policy Alignment", "policy_alignment"),
        ("Groundedness", "grounded"),
        ("Avg Steps per Task", "step_count"),
        ("Total SQL Queries", "sql_calls_total"),
        ("Unnecessary Tool Calls (Total)", "unnecessary_tool_calls_total"),
        ("Evidence Efficiency (Avg)", "evidence_efficiency_avg"),
        ("Budget Utilization (Avg)", "budget_utilization"),
        ("Mean Latency (s)", "latency_seconds"),
        ("Median Latency (s)", "latency_median"),
        ("Latency Spread / Std (s)", "latency_spread"),
    ]

    for label, key in metrics_to_show:
        row = [label]
        for regime in REGIMES:
            row.append(fmt(results[regime].get(key)))
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()


async def main() -> None:
    if not BENCHMARK_PATH.exists():
        console.print(f"[red]Benchmark not found at {BENCHMARK_PATH}[/red]")
        return

    queries = load_benchmark(BENCHMARK_PATH)
    console.print(f"[bold]Loaded {len(queries)} benchmark queries from {BENCHMARK_PATH}[/bold]")

    judge_llm = OllamaClient(model=settings.ollama_model, host=settings.ollama_host)

    summary_results: dict[str, dict[str, Any]] = {}
    detailed_results: dict[str, list[dict[str, Any]]] = {}

    for regime in REGIMES:
        console.print(f"\n[bold green]=== Running Regime: {regime.upper()} ===[/bold green]")
        summary, details = await run_regime(queries, judge_llm, regime)
        summary_results[regime] = summary
        detailed_results[regime] = details

        # Incremental checkpoint
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "timestamp": time.time(),
            "hypothesis": (
                "In our previous retrieval-budget experiment on pure RAG, limiting retrieval "
                "consumption affected task success and reduced unnecessary exploration. Phase 1.5 "
                "tests whether this action-space constraint generalizes to heterogeneous business "
                "decision tasks combining SQL, statistical analysis, and policy constraints."
            ),
            "summaries": summary_results,
            "details": detailed_results,
        }
        REPORT_PATH.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    print_experiment_report(summary_results)

    # Save artifact
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": time.time(),
        "hypothesis": (
            "In our previous retrieval-budget experiment on pure RAG, limiting retrieval "
            "consumption affected task success and reduced unnecessary exploration. Phase 1.5 "
            "tests whether this action-space constraint generalizes to heterogeneous business "
            "decision tasks combining SQL, statistical analysis, and policy constraints."
        ),
        "summaries": summary_results,
        "details": detailed_results,
    }
    REPORT_PATH.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    console.print(f"[green]Report saved to {REPORT_PATH}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
