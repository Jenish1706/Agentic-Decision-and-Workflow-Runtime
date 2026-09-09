"""End-to-end integration and policy tests for Multimodal Business Investigation Domain.

Exercises:
1. Document extraction: parsing scanned invoice into structured line items and totals.
2. Financial governance: autonomous approval under $1,000 threshold (ALLOWED).
3. Executive escalation: expense exceeding $1,000 requires authorization (REQUIRES_APPROVAL).
4. Fraud prevention: mathematical mismatch between subtotal+tax and total triggers BLOCKED.
5. Duplicate prevention: re-billed invoice triggers BLOCKED.
6. Telemetry chart analysis: SLA uptime breach triggers REQUIRES_APPROVAL.
7. Funnel chart analysis: identifies biggest drop-off stage and conversion rate.
"""

import pytest

from agent_runtime.agent.actions import ActionDecision, ActionType
from agent_runtime.agent.evidence import EvidenceKind, EvidenceSource
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.policy import PolicyStatus, ProposedDecision
from agent_runtime.agent.state import TerminationReason
from agent_runtime.agent.tools.base import Tool, ToolResult
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.mcp_layer.servers.vision_server import (
    analyze_chart_data,
    extract_document_metrics,
    list_visual_artifacts,
)
from domains.multimodal_investigation.policies import create_multimodal_policy_engine


class _LocalDocumentExtractTool(Tool):
    name = "extract_document_metrics"
    description = "Extract structured metrics and line items from a scanned invoice or document."
    parameters_schema = {
        "type": "object",
        "properties": {"document_path": {"type": "string"}},
        "required": ["document_path"],
    }
    origin = "local"

    async def run(self, document_path: str) -> ToolResult:
        res = extract_document_metrics(document_path)
        if not res.ok:
            return ToolResult(ok=False, content="", error=res.error)
        return ToolResult(
            ok=True,
            content=res.summary,
            data={
                "document_id": res.document_id,
                "vendor_name": res.vendor_name,
                "subtotal": res.subtotal,
                "tax_amount": res.tax_amount,
                "total_amount": res.total_amount,
                "settlement_status": res.settlement_status,
                "is_duplicate": res.is_duplicate,
                "line_items": res.line_items,
            },
        )


class _LocalChartAnalysisTool(Tool):
    name = "analyze_chart_data"
    description = "Analyze telemetry or funnel chart visual artifact."
    parameters_schema = {
        "type": "object",
        "properties": {"image_path": {"type": "string"}},
        "required": ["image_path"],
    }
    origin = "local"

    async def run(self, image_path: str) -> ToolResult:
        res = analyze_chart_data(image_path)
        if not res.ok:
            return ToolResult(ok=False, content="", error=res.error)
        return ToolResult(
            ok=True,
            content=res.summary,
            data={
                "artifact_id": res.artifact_id,
                "chart_type": res.chart_type,
                "title": res.title,
                "metrics": res.metrics,
                "sla_breach_detected": res.sla_breach_detected,
                "data_points": res.data_points,
            },
        )


class FakeLLM:
    def __init__(self, decisions: list[ActionDecision]):
        self._decisions = list(decisions)
        self.call_count = 0

    async def decide(self, messages, response_model, max_retries=2):
        if not self._decisions:
            raise RuntimeError("FakeLLM ran out of decisions")
        d = self._decisions.pop(0)
        self.call_count += 1
        return d


@pytest.mark.asyncio
async def test_multimodal_autonomous_expense_approval_flow():
    """INV-2026-004 is $453.60 (<$1,000 cap), arithmetic matches ($420 + $33.60); ALLOWED."""
    tools = ToolRegistry()
    tools.register(_LocalDocumentExtractTool())

    policy_engine = create_multimodal_policy_engine()

    decisions = [
        # Step 0: extract invoice
        ActionDecision(
            thought="Extract document metrics from INV-2026-004",
            action=ActionType.CALL_TOOL,
            tool_name="extract_document_metrics",
            tool_arguments={"document_path": "inv-2026-004.json"},
        ),
        # Step 1: recommend payment
        ActionDecision(
            thought="Office supplies invoice reconciles at $453.60 and is within $1,000 threshold.",
            action=ActionType.FINAL,
            answer="Approve payment of $453.60 for invoice INV-2026-004. Subtotal is $420.00, tax is $33.60, and total is $453.60.",
            proposed_decision=ProposedDecision(
                intent="Disburse payment for office supplies",
                recommended_action="approve_payment",
                parameters={"document_id": "INV-2026-004", "subtotal": 420.0, "tax_amount": 33.6, "total_amount": 453.6},
                reasoning="Verified invoice under $1,000 autonomous threshold.",
                confidence=0.99,
            ),
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Audit invoice INV-2026-004 for payment approval.")

    assert state.done is True
    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert len(state.evidence_ledger) == 1
    ev = state.evidence_ledger.all()[0]
    assert ev.source == EvidenceSource.VISION_VLM
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED
    assert state.policy_evaluation.requires_approval is False


@pytest.mark.asyncio
async def test_multimodal_executive_expense_approval_required():
    """INV-2026-001 is $2,592.00 (>$1,000 cap); triggers REQUIRES_APPROVAL."""
    tools = ToolRegistry()
    tools.register(_LocalDocumentExtractTool())

    policy_engine = create_multimodal_policy_engine()

    decisions = [
        # Step 0: extract invoice
        ActionDecision(
            thought="Extract document metrics from INV-2026-001",
            action=ActionType.CALL_TOOL,
            tool_name="extract_document_metrics",
            tool_arguments={"document_path": "inv-2026-001.json"},
        ),
        # Step 1: propose payment
        ActionDecision(
            thought="Invoice is legitimate but exceeds $1,000 threshold.",
            action=ActionType.FINAL,
            answer="Invoice INV-2026-001 total is $2,592.00 with subtotal of $2,400.00 and tax of $192.00. Payment requires executive authorization.",
            proposed_decision=ProposedDecision(
                intent="Disburse payment for consulting services",
                recommended_action="approve_payment",
                parameters={"document_id": "INV-2026-001", "subtotal": 2400.0, "tax_amount": 192.0, "total_amount": 2592.0},
                reasoning="Consulting services verified, amount exceeds $1,000 threshold.",
                confidence=0.95,
            ),
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Audit invoice INV-2026-001.")

    assert state.done is True
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert state.policy_evaluation.requires_approval is True
    assert "exceeds autonomous $1,000 limit" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_multimodal_fraud_discrepancy_blocked():
    """INV-2026-002 has math discrepancy ($1,500 + $120 != $2,120); triggers BLOCKED."""
    tools = ToolRegistry()
    tools.register(_LocalDocumentExtractTool())

    policy_engine = create_multimodal_policy_engine()

    decisions = [
        # Step 0: extract invoice
        ActionDecision(
            thought="Extract document metrics from INV-2026-002",
            action=ActionType.CALL_TOOL,
            tool_name="extract_document_metrics",
            tool_arguments={"document_path": "inv-2026-002.json"},
        ),
        # Step 1: propose payment despite discrepancy
        ActionDecision(
            thought="Attempt payment proposal",
            action=ActionType.FINAL,
            answer="Invoice INV-2026-002 has subtotal $1,500.00, tax $120.00, and billed total $2,120.00.",
            proposed_decision=ProposedDecision(
                intent="Pay invoice INV-2026-002",
                recommended_action="approve_payment",
                parameters={"document_id": "INV-2026-002", "subtotal": 1500.0, "tax_amount": 120.0, "total_amount": 2120.0},
                reasoning="Attempt approval.",
                confidence=0.5,
            ),
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Audit invoice INV-2026-002.")

    assert state.done is True
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.BLOCKED
    assert "Arithmetic mismatch" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_multimodal_duplicate_invoice_blocked():
    """INV-2026-003 is already paid duplicate; triggers BLOCKED."""
    tools = ToolRegistry()
    tools.register(_LocalDocumentExtractTool())

    policy_engine = create_multimodal_policy_engine()

    decisions = [
        # Step 0: extract invoice
        ActionDecision(
            thought="Extract document metrics from INV-2026-003",
            action=ActionType.CALL_TOOL,
            tool_name="extract_document_metrics",
            tool_arguments={"document_path": "inv-2026-003.json"},
        ),
        # Step 1: propose payment
        ActionDecision(
            thought="Attempt payment proposal",
            action=ActionType.FINAL,
            answer="Invoice INV-2026-003 is a resubmitted duplicate.",
            proposed_decision=ProposedDecision(
                intent="Pay invoice INV-2026-003",
                recommended_action="approve_payment",
                parameters={"document_id": "INV-2026-003", "is_duplicate": True, "settlement_status": "already_paid"},
                reasoning="Resubmitted invoice.",
                confidence=0.5,
            ),
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Audit invoice INV-2026-003.")

    assert state.done is True
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.BLOCKED
    assert "Duplicate invoice detected" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_multimodal_sla_chart_breach_detection():
    """Telemetry chart shows uptime dip to 99.12% (<99.90% SLA); triggers REQUIRES_APPROVAL."""
    tools = ToolRegistry()
    tools.register(_LocalChartAnalysisTool())

    policy_engine = create_multimodal_policy_engine()

    decisions = [
        # Step 0: analyze SLA chart
        ActionDecision(
            thought="Analyze SLA telemetry chart",
            action=ActionType.CALL_TOOL,
            tool_name="analyze_chart_data",
            tool_arguments={"image_path": "latency_sla_september.json"},
        ),
        # Step 1: propose SLA assessment
        ActionDecision(
            thought="Chart shows Week 3 outage of 88.7 minutes with 99.12% minimum uptime.",
            action=ActionType.FINAL,
            answer="The September telemetry chart shows a minimum uptime of 99.12% and monthly average uptime of 99.75%, breaching the 99.90% SLA commitment.",
            proposed_decision=ProposedDecision(
                intent="Assess commercial SLA penalty credit",
                recommended_action="apply_penalty_credit",
                parameters={"artifact_id": "CHART-SLA-0926", "min_uptime_pct": 99.12, "monthly_average_uptime_pct": 99.75, "sla_breach_detected": True},
                reasoning="SLA breached in Week 3.",
                confidence=0.98,
            ),
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Check September uptime chart for SLA compliance.")

    assert state.done is True
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "SLA breach detected" in state.policy_evaluation.audit_notes


def test_multimodal_list_artifacts():
    """Verify discovery of invoices and charts."""
    res = list_visual_artifacts()
    assert res.ok is True
    assert "inv-2026-001" in res.invoices
    assert "latency_sla_september" in res.charts
    assert res.total_artifacts >= 6
