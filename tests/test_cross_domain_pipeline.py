"""One runtime, four domains.

The same AgentRuntime - evidence ledger, numerical verification, policy
engine, audit trace - runs a task from every domain. Only the tools and the
policy engine change, and both come from the domain loader.
"""

import pytest

from agent_runtime.agent.actions import ActionDecision, ActionType, DecisionExtraction
from agent_runtime.agent.evidence import EvidenceSource
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.policy import PolicyStatus, ProposedDecision
from agent_runtime.agent.tools.base import ToolCategory
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.domains import list_domains, load_domain
from domains.credit_risk.server import get_account_profile
from domains.enterprise_operations.server import list_service_incidents
from domains.multimodal_investigation.server import extract_document_metrics
from domains.product_analytics.server import execute_sql as analytics_sql

from .fakes import FakeLLMClient, FunctionTool


def _run_domain(
    domain: str,
    tool: FunctionTool,
    call_arguments: dict,
    answer: str,
    proposal: ProposedDecision | None,
    query: str | None = None,
):
    tools = ToolRegistry()
    tools.register(tool)
    decisions = [
        ActionDecision(thought="gather", action=ActionType.CALL_TOOL, tool_name=tool.name, tool_arguments=call_arguments),
        ActionDecision(thought="answer", action=ActionType.FINAL, answer=answer, proposed_decision=proposal),
    ]
    if proposal is None:
        # With a policy engine configured, an answer without an attached
        # decision gets one extraction call; this question asks for none.
        decisions.append(
            DecisionExtraction(asks_for_decision=False, recommended_action="", parameters={}, reasoning="", confidence=0.0)
        )
    runtime = AgentRuntime(
        llm=FakeLLMClient(decisions),
        tools=tools,
        policy_engine=load_domain(domain).create_policy_engine(),
        max_steps=4,
    )
    return runtime.run(query or f"{domain} task")


def _proposal(action: str, **parameters) -> ProposedDecision:
    return ProposedDecision(intent=action, recommended_action=action, parameters=parameters, reasoning="evidence", confidence=0.9)


def test_every_domain_folder_loads():
    assert list_domains() == ["credit_risk", "enterprise_operations", "multimodal_investigation", "product_analytics"]


@pytest.mark.asyncio
async def test_product_analytics_through_runtime():
    state = await _run_domain(
        "product_analytics",
        FunctionTool(analytics_sql, ToolCategory.SQL),
        {"query": "SELECT SUM(checkouts_started) AS total FROM daily_checkout_metrics WHERE period = 'baseline_week'"},
        "Total checkouts started in baseline_week: 10,400.",
        None,
    )
    assert state.answer_verified is True
    assert state.evidence_ledger.all()[0].source == EvidenceSource.ANALYTICS_SQL
    assert state.policy_evaluation is None  # no decision was proposed


@pytest.mark.asyncio
async def test_credit_risk_through_runtime():
    state = await _run_domain(
        "credit_risk",
        FunctionTool(get_account_profile, ToolCategory.RECORD),
        {"account_id": "ACCT-101"},
        "Approve ACCT-101's increase to $8,000; credit score 740.",
        _proposal("approve_increase", account_id="ACCT-101", increase_amount=3000),
        query="Evaluate credit limit increase for ACCT-101.",
    )
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED


@pytest.mark.asyncio
async def test_multimodal_through_runtime():
    state = await _run_domain(
        "multimodal_investigation",
        FunctionTool(extract_document_metrics, ToolCategory.VISION),
        {"document_path": "inv-2026-004.json"},
        "Approve payment of $453.60 for INV-2026-004.",
        _proposal("approve_payment", document_id="INV-2026-004"),
        query="Audit invoice INV-2026-004.",
    )
    assert state.evidence_ledger.all()[0].source == EvidenceSource.VISION_VLM
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED


@pytest.mark.asyncio
async def test_enterprise_operations_through_runtime():
    state = await _run_domain(
        "enterprise_operations",
        FunctionTool(list_service_incidents, ToolCategory.RECORD),
        {"service_name": "recommendation-worker"},
        "Scale recommendation-worker for INC-903 (10.0% blast radius).",
        _proposal("scale_workers", incident_id="INC-903", blast_radius_pct=10),
        query="Scale recommendation-worker for INC-903 (10.0% blast radius).",
    )
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED
    assert state.audit_trace["evidence_count"] == 1
