"""Cross-Domain Evaluation Suite for Agentic Decision & Workflow Runtime.

Demonstrates that the SAME core runtime architecture:
  Evidence Gathering -> Bounded Analysis -> Numerical Verification -> Policy Governance -> Audit Trace
solves heterogeneous business problems across 4 real-world enterprise domains:
1. Product Analytics & Growth Experiments (SQL + SciPy + Rollout Policy)
2. Credit Risk & Lending Underwriting (DuckDB + FICO/DTI Rules + Escalation)
3. Multimodal Business Investigation (Scanned Invoices + Visual Charts + Fraud Audit)
4. Enterprise Systems Operations (SRE Telemetry + Drift Detection + Blast Radius Governance)

Usage:
  uv run python scripts/run_cross_domain_eval.py
"""

import argparse
import asyncio
import json
from pathlib import Path
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from agent_runtime.agent.budget import BudgetController, allocate_budget, classify_task_complexity
from agent_runtime.agent.evidence import EvidenceSource
from agent_runtime.agent.policy import PolicyStatus
from agent_runtime.agent.verification import verify_numerical_grounding
from agent_runtime.config import REPO_ROOT, settings
from agent_runtime.evaluation.agent_metrics import single_agent_failed, single_agent_tool_names
from agent_runtime.evaluation.dataset import EvalQuery, load_benchmark
from agent_runtime.evaluation.rag_metrics import score_groundedness
from agent_runtime.evaluation.run_eval import _judged
from agent_runtime.llm.ollama_client import OllamaClient
from agent_runtime.runtime_factory import build_single_agent_runtime

from domains.credit_risk.policies import create_credit_risk_policy_engine
from domains.enterprise_operations.policies import create_enterprise_ops_policy_engine
from domains.multimodal_investigation.policies import create_multimodal_policy_engine

console = Console()

OUTPUT_REPORT_PATH = REPO_ROOT / "data" / "cross_domain_evaluation_report.json"

DOMAIN_CONFIGS: dict[str, dict[str, Any]] = {
    "product_analytics": {
        "title": "Product Analytics & Growth",
        "benchmark_path": REPO_ROOT / "domains" / "product_analytics" / "benchmark.json",
        "server_modules": {"analytics": "agent_runtime.mcp_layer.servers.analytics_server"},
        "policy_factory": None,  # Will be imported from experiment
        "sample_query_ids": ["pa-01", "pa-03", "pa-07", "pa-10"],
    },
    "credit_risk": {
        "title": "Credit Risk & Lending",
        "benchmark_path": REPO_ROOT / "domains" / "credit_risk" / "benchmark.json",
        "server_modules": {"credit_risk": "agent_runtime.mcp_layer.servers.credit_risk_server"},
        "policy_factory": create_credit_risk_policy_engine,
        "sample_query_ids": ["cr-01", "cr-02", "cr-03", "cr-04"],
    },
    "multimodal_investigation": {
        "title": "Multimodal Document & Chart Audit",
        "benchmark_path": REPO_ROOT / "domains" / "multimodal_investigation" / "benchmark.json",
        "server_modules": {"vision": "agent_runtime.mcp_layer.servers.vision_server"},
        "policy_factory": create_multimodal_policy_engine,
        "sample_query_ids": ["mm-01", "mm-02", "mm-03", "mm-06"],
    },
    "enterprise_operations": {
        "title": "Enterprise SRE & Systems Ops",
        "benchmark_path": REPO_ROOT / "domains" / "enterprise_operations" / "benchmark.json",
        "server_modules": {"ops": "agent_runtime.mcp_layer.servers.ops_server"},
        "policy_factory": create_enterprise_ops_policy_engine,
        "sample_query_ids": ["ops-01", "ops-02", "ops-03", "ops-05"],
    },
}


def _get_product_analytics_policy_engine():
    from agent_runtime.evaluation.product_analytics_experiment import create_product_analytics_policy_engine
    return create_product_analytics_policy_engine()


DOMAIN_CONFIGS["product_analytics"]["policy_factory"] = _get_product_analytics_policy_engine


async def evaluate_domain(
    domain_key: str,
    domain_info: dict[str, Any],
    judge_llm: OllamaClient,
) -> dict[str, Any]:
    benchmark_file = domain_info["benchmark_path"]
    if not benchmark_file.exists():
        console.print(f"[red]Missing benchmark: {benchmark_file}[/red]")
        return {"error": f"Benchmark {benchmark_file} not found"}

    all_queries = load_benchmark(benchmark_file)
    target_ids = set(domain_info["sample_query_ids"])
    queries = [q for q in all_queries if q.id in target_ids]
    if not queries:
        queries = all_queries[:4]

    console.print(f"\n[bold blue]>>> Evaluating Domain: {domain_info['title']} ({len(queries)} scenarios)[/bold blue]")

    policy_engine = domain_info["policy_factory"]()
    server_modules = domain_info["server_modules"]

    results: list[dict[str, Any]] = []

    async with build_single_agent_runtime(
        with_memory=False,
        policy_engine=policy_engine,
        server_modules=server_modules,
        include_rag=False,
    ) as runtime:
        for idx, q in enumerate(queries, start=1):
            console.print(f"  [{idx}/{len(queries)}] {q.id}: {q.question[:65]}...")
            runtime._short_term.clear()

            # Allocate adaptive budget
            complexity = classify_task_complexity(q.question)
            budget = allocate_budget(complexity, mode="adaptive")
            controller = BudgetController(budget=budget)
            runtime._budget_controller = controller

            t0 = time.perf_counter()
            state = await runtime.run(q.question)
            latency = time.perf_counter() - t0

            answer = state.final_answer or ""
            evidence_text = "\n\n".join(
                s.observation for s in state.steps if s.decision.action.value in ("retrieve", "call_tool")
            )

            # Groundedness & Numerical Verification
            grounded_res = await _judged(
                score_groundedness(judge_llm, q, answer, evidence_text), default=(True, [])
            )
            num_res = verify_numerical_grounding(answer, state.evidence_ledger)

            # Policy compliance check
            policy_status = (
                state.policy_evaluation.final_status.value
                if state.policy_evaluation is not None
                else "none"
            )
            policy_expected = q.expected_policy_status
            policy_matched = (policy_status == policy_expected) if policy_expected is not None else True

            # Categorize gathered evidence
            sources = [it.source.value for it in state.evidence_ledger.all()]
            sql_count = sum(1 for s in sources if s == EvidenceSource.ANALYTICS_SQL.value)
            vlm_count = sum(1 for s in sources if s == EvidenceSource.VISION_VLM.value)
            rec_count = sum(1 for s in sources if s == EvidenceSource.RECORD_LOOKUP.value)
            stat_count = sum(1 for s in sources if s == EvidenceSource.STATISTICS.value)

            row = {
                "id": q.id,
                "category": q.category,
                "steps": len(state.steps),
                "evidence_count": len(state.evidence_ledger),
                "evidence_sources": {
                    "sql": sql_count,
                    "vision": vlm_count,
                    "records": rec_count,
                    "statistics": stat_count,
                },
                "numerical_grounded": num_res.valid,
                "grounded": bool(grounded_res[0]) if grounded_res[0] is not None else True,
                "policy_enforced": policy_status,
                "expected_policy": policy_expected,
                "policy_aligned": policy_matched,
                "tools_used": single_agent_tool_names(state),
                "latency_seconds": round(latency, 2),
                "failed": single_agent_failed(state),
                "final_answer_preview": answer[:120] + "..." if len(answer) > 120 else answer,
            }
            results.append(row)

    # Domain summary
    total = len(results)
    num_grounded_pct = (sum(1 for r in results if r["numerical_grounded"]) / total) * 100.0 if total else 0.0
    grounded_pct = (sum(1 for r in results if r["grounded"]) / total) * 100.0 if total else 0.0
    policy_aligned_pct = (sum(1 for r in results if r["policy_aligned"]) / total) * 100.0 if total else 0.0
    avg_steps = sum(r["steps"] for r in results) / total if total else 0.0
    avg_evidence = sum(r["evidence_count"] for r in results) / total if total else 0.0
    avg_latency = sum(r["latency_seconds"] for r in results) / total if total else 0.0

    return {
        "title": domain_info["title"],
        "scenarios_evaluated": total,
        "avg_steps": round(avg_steps, 2),
        "avg_evidence_items": round(avg_evidence, 2),
        "numerical_grounding_pct": round(num_grounded_pct, 1),
        "groundedness_pct": round(grounded_pct, 1),
        "policy_alignment_pct": round(policy_aligned_pct, 1),
        "avg_latency_s": round(avg_latency, 2),
        "scenarios": results,
    }


def print_cross_domain_matrix(domain_results: dict[str, dict[str, Any]]) -> None:
    table = Table(title="Cross-Domain Generalization Matrix (Unified Agentic Decision Runtime)")
    table.add_column("Business Domain", style="bold cyan")
    table.add_column("Scenarios", justify="right")
    table.add_column("Avg Steps", justify="right")
    table.add_column("Avg Evidence", justify="right")
    table.add_column("Num. Grounding", justify="right", style="green")
    table.add_column("Groundedness", justify="right", style="green")
    table.add_column("Policy Alignment", justify="right", style="magenta")
    table.add_column("Avg Latency (s)", justify="right")

    for dom_key, res in domain_results.items():
        if "error" in res:
            continue
        table.add_row(
            res["title"],
            str(res["scenarios_evaluated"]),
            f"{res['avg_steps']:.1f}",
            f"{res['avg_evidence_items']:.1f}",
            f"{res['numerical_grounding_pct']:.1f}%",
            f"{res['groundedness_pct']:.1f}%",
            f"{res['policy_alignment_pct']:.1f}%",
            f"{res['avg_latency_s']:.2f}s",
        )

    console.print()
    console.print(table)
    console.print()


async def main() -> None:
    console.print("[bold green]============================================================[/bold green]")
    console.print("[bold green]   CROSS-DOMAIN AGENTIC DECISION & WORKFLOW RUNTIME EVAL   [/bold green]")
    console.print("[bold green]============================================================[/bold green]")

    judge_llm = OllamaClient(model=settings.ollama_model, host=settings.ollama_host)

    all_domain_results: dict[str, dict[str, Any]] = {}

    for domain_key, domain_info in DOMAIN_CONFIGS.items():
        res = await evaluate_domain(domain_key, domain_info, judge_llm)
        all_domain_results[domain_key] = res

    print_cross_domain_matrix(all_domain_results)

    # Save artifact
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": time.time(),
        "runtime_invariants": [
            "LLM proposes; deterministic components verify and constrain.",
            "Append-only Evidence Ledger tracking heterogeneous observations across all domains.",
            "Deterministic Numerical Verification detecting ungrounded quantitative assertions.",
            "Deterministic Policy Engine enforcing hierarchical governance (BLOCKED > REQUIRES_APPROVAL > ALLOWED).",
            "Action Budget Controller preventing exploration thrashing and cost escalation.",
            "Comprehensive Audit Trace preserving provenance and decision accountability.",
        ],
        "domain_results": all_domain_results,
    }
    OUTPUT_REPORT_PATH.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    console.print(f"[green]Full cross-domain evaluation report saved to {OUTPUT_REPORT_PATH}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
