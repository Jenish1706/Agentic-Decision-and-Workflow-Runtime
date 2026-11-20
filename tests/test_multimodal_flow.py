"""Multimodal investigation domain through the real agent loop.

"Vision" here is simulated over JSON fixtures (see the server docstring);
what these tests exercise is everything downstream of extraction: evidence,
numerical verification and the payment / SLA policies.
"""

import pytest

from agent_runtime.agent.actions import ActionDecision, ActionType
from agent_runtime.agent.evidence import EvidenceSource
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.policy import PolicyStatus, ProposedDecision
from agent_runtime.agent.state import TerminationReason
from agent_runtime.agent.tools.base import ToolCategory
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.domains import load_domain
from domains.multimodal_investigation.server import (
    analyze_chart_data,
    extract_document_metrics,
    list_visual_artifacts,
)

from .fakes import FakeLLMClient, FunctionTool


def _runtime(decisions: list[ActionDecision]) -> AgentRuntime:
    tools = ToolRegistry()
    tools.register(FunctionTool(extract_document_metrics, ToolCategory.VISION))
    tools.register(FunctionTool(analyze_chart_data, ToolCategory.VISION))
    return AgentRuntime(
        llm=FakeLLMClient(decisions),
        tools=tools,
        policy_engine=load_domain("multimodal_investigation").create_policy_engine(),
        max_steps=5,
    )


def _call(tool_name: str, **arguments) -> ActionDecision:
    return ActionDecision(thought=f"Call {tool_name}", action=ActionType.CALL_TOOL, tool_name=tool_name, tool_arguments=arguments)


def _decide(answer: str, action: str, **parameters) -> ActionDecision:
    return ActionDecision(
        thought="Propose the decision",
        action=ActionType.FINAL,
        answer=answer,
        proposed_decision=ProposedDecision(
            intent="Payment or penalty decision",
            recommended_action=action,
            parameters=parameters,
            reasoning="Based on the extracted document.",
            confidence=0.9,
        ),
    )


def _statuses(state) -> dict[str, PolicyStatus]:
    return {result.rule_name: result.effective_status for result in state.policy_evaluation.rule_results}


@pytest.mark.asyncio
async def test_small_reconciled_invoice_is_approved():
    runtime = _runtime([
        _call("extract_document_metrics", document_path="inv-2026-004.json"),
        _decide(
            "Approve $453.60 for INV-2026-004: subtotal $420.00 plus tax $33.60.",
            "approve_payment",
            document_id="INV-2026-004",
        ),
    ])
    state = await runtime.run("Audit invoice INV-2026-004 for payment.")

    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert state.evidence_ledger.all()[0].source == EvidenceSource.VISION_VLM
    assert state.answer_verified is True
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED


@pytest.mark.asyncio
async def test_invoice_above_expense_limit_requires_executive_approval():
    runtime = _runtime([
        _call("extract_document_metrics", document_path="INV-2026-001"),
        _decide("INV-2026-001 totals $2,592.00.", "approve_payment", document_id="INV-2026-001"),
    ])
    state = await runtime.run("Audit invoice INV-2026-001.")

    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "exceeds autonomous $1,000 limit" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_arithmetic_mismatch_blocks_payment():
    runtime = _runtime([
        _call("extract_document_metrics", document_path="inv-2026-002.json"),
        _decide(
            "INV-2026-002 bills $2,120.00, but subtotal plus tax is $1,620.00: a $500.00 discrepancy.",
            "approve_payment",
            document_id="INV-2026-002",
        ),
    ])
    state = await runtime.run("Inspect INV-2026-002 before payment.")

    assert state.answer_verified is True  # 1,620 and 500 are computed by the tool, so they are grounded
    assert state.policy_evaluation.final_status == PolicyStatus.BLOCKED
    assert "Arithmetic mismatch" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_duplicate_invoice_blocks_payment():
    runtime = _runtime([
        _call("extract_document_metrics", document_path="inv-2026-003.json"),
        _decide("INV-2026-003 resubmits INV-2026-001.", "approve_payment", document_id="INV-2026-003"),
    ])
    state = await runtime.run("Evaluate INV-2026-003 for payment.")

    assert state.policy_evaluation.final_status == PolicyStatus.BLOCKED
    assert "Duplicate invoice detected" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_sla_breach_requires_penalty_review():
    runtime = _runtime([
        _call("analyze_chart_data", image_path="latency_sla_september"),
        _decide(
            "Monthly average uptime was 99.75%, below the 99.90% SLA; the worst week was 99.12%.",
            "apply_penalty_credit",
            artifact_id="latency_sla_september",
        ),
    ])
    state = await runtime.run("Did the September uptime chart breach the SLA?")

    statuses = _statuses(state)
    assert state.answer_verified is True
    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "SLA breach detected" in state.policy_evaluation.audit_notes
    # A penalty action is not a payment: the invoice rules must not run against it.
    assert statuses["invoice_math_reconciliation"] == PolicyStatus.ALLOWED


# ---------------------------------------------------------------- regressions


@pytest.mark.asyncio
async def test_proposal_cannot_clear_a_duplicate_flag():
    # Before: {"is_duplicate": false, "settlement_status": "unpaid"} in the
    # proposal overrode the document.
    runtime = _runtime([
        _call("extract_document_metrics", document_path="inv-2026-003.json"),
        _decide(
            "Pay INV-2026-003.",
            "approve_payment",
            document_id="INV-2026-003",
            is_duplicate=False,
            settlement_status="unpaid",
        ),
    ])
    state = await runtime.run("Pay INV-2026-003?")

    assert _statuses(state)["duplicate_invoice_prevention"] == PolicyStatus.BLOCKED


@pytest.mark.asyncio
async def test_rules_read_the_named_invoice_not_the_first_one_extracted():
    runtime = _runtime([
        _call("extract_document_metrics", document_path="inv-2026-004.json"),
        _call("extract_document_metrics", document_path="inv-2026-002.json"),
        _decide("Pay INV-2026-004.", "approve_payment", document_id="INV-2026-004"),
    ])
    state = await runtime.run("Pay INV-2026-004?")

    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED


@pytest.mark.asyncio
async def test_payment_without_extraction_is_not_allowed():
    runtime = _runtime([_decide("Pay INV-2026-004.", "approve_payment", document_id="INV-2026-004")])
    state = await runtime.run("Pay INV-2026-004?")

    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL


@pytest.mark.asyncio
async def test_target_binding_catches_document_path_swap_with_multiple_invoices_extracted():
    # User asks to pay INV-2026-001, but the proposal specifies document_path="invoices/inv-2026-004.json".
    # Even though both invoices are in evidence, target entity binding must require approval.
    runtime = _runtime([
        _call("extract_document_metrics", document_path="inv-2026-001.json"),
        _call("extract_document_metrics", document_path="inv-2026-004.json"),
        _decide("Pay INV-2026-004.", "approve_payment", document_path="invoices/inv-2026-004.json"),
    ])
    state = await runtime.run("Pay INV-2026-001?")

    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert any(r.rule_name == "target_entity_binding" for r in state.policy_evaluation.rule_results)


# ---------------------------------------------------------------- server


def test_paths_outside_the_data_folder_are_refused():
    for reference in ("../../pyproject.toml", "..\\..\\.env", "C:/Windows/win.ini", "*", "inv-*"):
        result = extract_document_metrics(reference)
        assert not result.ok, reference


def test_absent_fields_are_null_not_invented():
    invoice = extract_document_metrics("INV-2026-001")
    assert invoice.ok
    assert invoice.is_duplicate is False  # no duplicate flag in the document
    assert invoice.line_items_total == 2400.0 and invoice.computed_total == 2592.0 and invoice.total_discrepancy == 0.0


def test_chart_breach_is_computed_from_monthly_average():
    chart = analyze_chart_data("CHART-SLA-0926")
    assert chart.ok and chart.artifact_name == "latency_sla_september"
    assert chart.sla_breach_detected is True
    assert chart.metrics["monthly_average_uptime_pct"] == 99.75

    funnel = analyze_chart_data("conversion_funnel_redesign")
    assert funnel.sla_breach_detected is None  # no SLA on a funnel chart


def test_list_artifacts():
    listing = list_visual_artifacts()
    assert "inv-2026-001" in listing.invoices
    assert "latency_sla_september" in listing.charts
    assert listing.total_artifacts == 6
