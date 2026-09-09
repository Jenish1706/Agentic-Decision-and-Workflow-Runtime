"""Unified Cross-Domain Pipeline Test.

Verifies that the SAME invariant AgentRuntime loop:
  Query -> Tools/Evidence -> Numerical Verification -> Policy Engine -> Audit Trace
handles tasks from all 4 enterprise domains:
1. Product Analytics
2. Credit Risk Underwriting
3. Multimodal Document & Chart Audit
4. Enterprise SRE & Systems Operations
"""

import pytest

from agent_runtime.agent.actions import ActionDecision, ActionType
from agent_runtime.agent.evidence import EvidenceKind, EvidenceSource
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.policy import PolicyStatus, ProposedDecision
from agent_runtime.agent.state import TerminationReason
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.mcp_layer.servers.analytics_server import execute_sql as run_analytics_sql
from agent_runtime.mcp_layer.servers.credit_risk_server import get_account_profile
from agent_runtime.mcp_layer.servers.ops_server import get_incident_telemetry
from agent_runtime.mcp_layer.servers.vision_server import extract_document_metrics
from agent_runtime.agent.tools.base import Tool, ToolResult

from domains.credit_risk.policies import create_credit_risk_policy_engine
from domains.enterprise_operations.policies import create_enterprise_ops_policy_engine
from domains.multimodal_investigation.policies import create_multimodal_policy_engine


class _MockTool(Tool):
    def __init__(self, name: str, fn, schema: dict):
        self.name = name
        self.description = f"Mock wrapper for {name}"
        self.parameters_schema = schema
        self.origin = "local"
        self._fn = fn

    async def run(self, **kwargs) -> ToolResult:
        res = self._fn(**kwargs)
        if hasattr(res, "ok") and not res.ok:
            return ToolResult(ok=False, content="", error=getattr(res, "error", "error"))
        summary = getattr(res, "summary", str(res))
        data = res.model_dump() if hasattr(res, "model_dump") else (res.to_dict() if hasattr(res, "to_dict") else {"res": str(res)})
        return ToolResult(ok=True, content=summary, data=data)


class FakeLLM:
    def __init__(self, decisions: list[ActionDecision]):
        self._decisions = list(decisions)

    async def decide(self, messages, response_model, max_retries=2):
        if not self._decisions:
            raise RuntimeError("FakeLLM ran out of decisions")
        return self._decisions.pop(0)


@pytest.mark.asyncio
async def test_domain_1_product_analytics_through_runtime():
    tools = ToolRegistry()
    tools.register(_MockTool("execute_sql", run_analytics_sql, {"type": "object", "properties": {"query": {"type": "string"}}}))

    decisions = [
        ActionDecision(
            thought="Query checkout metrics for baseline_week",
            action=ActionType.CALL_TOOL,
            tool_name="execute_sql",
            tool_arguments={"query": "SELECT SUM(checkouts_started) as total FROM daily_checkout_metrics WHERE period = 'baseline_week'"},
        ),
        ActionDecision(
            thought="Summarize result",
            action=ActionType.FINAL,
            answer="Total checkouts started in baseline_week was 10,400.",
        ),
    ]

    runtime = AgentRuntime(llm=FakeLLM(decisions), tools=tools, max_steps=4)
    state = await runtime.run("What was the total number of checkouts?")

    assert state.done is True
    assert len(state.evidence_ledger) == 1
    assert state.evidence_ledger.all()[0].source == EvidenceSource.ANALYTICS_SQL


@pytest.mark.asyncio
async def test_domain_2_credit_risk_through_runtime():
    """Domain 2: Credit Risk Underwriting via AgentRuntime with PolicyEngine."""
    tools = ToolRegistry()
    tools.register(_MockTool("get_account_profile", get_account_profile, {"type": "object", "properties": {"account_id": {"type": "string"}}}))

    policy_engine = create_credit_risk_policy_engine()

    decisions = [
        ActionDecision(
            thought="Check borrower profile",
            action=ActionType.CALL_TOOL,
            tool_name="get_account_profile",
            tool_arguments={"account_id": "ACCT-101"},
        ),
        ActionDecision(
            thought="Approve borrower with 740 FICO",
            action=ActionType.FINAL,
            answer="Approve ACCT-101 credit limit increase to $8,000 with credit score 740.",
            proposed_decision=ProposedDecision(
                intent="Approve credit expansion",
                recommended_action="approve_increase",
                parameters={"account_id": "ACCT-101", "increase_amount": 3000.0, "credit_score": 740, "dti_ratio": 0.28},
                reasoning="Prime tier borrower.",
                confidence=0.95,
            ),
        ),
    ]

    runtime = AgentRuntime(llm=FakeLLM(decisions), tools=tools, policy_engine=policy_engine, max_steps=4)
    state = await runtime.run("Evaluate credit limit increase for ACCT-101.")

    assert state.done is True
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED


@pytest.mark.asyncio
async def test_domain_3_multimodal_through_runtime():
    """Domain 3: Multimodal Invoice Audit via AgentRuntime."""
    tools = ToolRegistry()
    tools.register(_MockTool("extract_document_metrics", extract_document_metrics, {"type": "object", "properties": {"document_path": {"type": "string"}}}))

    policy_engine = create_multimodal_policy_engine()

    decisions = [
        ActionDecision(
            thought="Extract invoice metrics",
            action=ActionType.CALL_TOOL,
            tool_name="extract_document_metrics",
            tool_arguments={"document_path": "inv-2026-004.json"},
        ),
        ActionDecision(
            thought="Disburse office supplies payment",
            action=ActionType.FINAL,
            answer="Approve payment of $453.60 for invoice INV-2026-004.",
            proposed_decision=ProposedDecision(
                intent="Approve invoice payment",
                recommended_action="approve_payment",
                parameters={"document_id": "INV-2026-004", "subtotal": 420.0, "tax_amount": 33.6, "total_amount": 453.6},
                reasoning="Small expense within $1,000 policy threshold.",
                confidence=0.99,
            ),
        ),
    ]

    runtime = AgentRuntime(llm=FakeLLM(decisions), tools=tools, policy_engine=policy_engine, max_steps=4)
    state = await runtime.run("Audit invoice INV-2026-004.")

    assert state.done is True
    assert len(state.evidence_ledger) == 1
    assert state.evidence_ledger.all()[0].source == EvidenceSource.VISION_VLM
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED


@pytest.mark.asyncio
async def test_domain_4_enterprise_ops_through_runtime():
    """Domain 4: Enterprise Operations & SRE Incident Response via AgentRuntime."""
    tools = ToolRegistry()
    tools.register(_MockTool("get_incident_telemetry", get_incident_telemetry, {"type": "object", "properties": {"incident_id": {"type": "string"}}}))

    policy_engine = create_enterprise_ops_policy_engine()

    decisions = [
        ActionDecision(
            thought="Inspect incident INC-903",
            action=ActionType.CALL_TOOL,
            tool_name="get_incident_telemetry",
            tool_arguments={"incident_id": "INC-903"},
        ),
        ActionDecision(
            thought="Worker scale-up is safe with 10% blast radius",
            action=ActionType.FINAL,
            answer="Approve scaling recommendation-worker for INC-903 with 10.0% blast radius.",
            proposed_decision=ProposedDecision(
                intent="Remediate worker lag",
                recommended_action="scale_workers",
                parameters={"incident_id": "INC-903", "severity": "P2", "blast_radius_pct": 10.0},
                reasoning="Safe remediation under 10% limit.",
                confidence=0.95,
            ),
        ),
    ]

    runtime = AgentRuntime(llm=FakeLLM(decisions), tools=tools, policy_engine=policy_engine, max_steps=4)
    state = await runtime.run("Remediate incident INC-903.")

    assert state.done is True
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED
    assert state.audit_trace["evidence_count"] == 1
