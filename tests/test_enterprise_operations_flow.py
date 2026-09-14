"""Enterprise-operations domain through the real agent loop.

Real server functions feed real domain policies; only the LLM is scripted.
The regression tests cover the bypasses the old rules allowed: a model
granting itself an emergency override, the string "false" counting as an
authorized override, and a change being allowed because nothing was looked up.
"""

import json

import pytest

import domains.enterprise_operations.server as ops_server
from agent_runtime.agent.actions import ActionDecision, ActionType
from agent_runtime.agent.evidence import EvidenceLedger, EvidenceSource
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.policy import DecisionContext, PolicyStatus, ProposedDecision
from agent_runtime.agent.tools.base import ToolCategory
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.domains import load_domain
from domains.enterprise_operations.server import (
    get_incident_telemetry,
    inspect_service_config,
    list_service_incidents,
    simulate_remediation_action,
)

from .fakes import FakeLLMClient, FunctionTool


def _engine():
    return load_domain("enterprise_operations").create_policy_engine()


def _runtime(decisions: list[ActionDecision]) -> AgentRuntime:
    tools = ToolRegistry()
    tools.register(FunctionTool(get_incident_telemetry, ToolCategory.RECORD))
    tools.register(FunctionTool(list_service_incidents, ToolCategory.RECORD))
    tools.register(FunctionTool(inspect_service_config, ToolCategory.RECORD))
    tools.register(FunctionTool(simulate_remediation_action, ToolCategory.STATISTICS))
    return AgentRuntime(llm=FakeLLMClient(decisions), tools=tools, policy_engine=_engine(), max_steps=6)


def _call(tool_name: str, **arguments) -> ActionDecision:
    return ActionDecision(thought=f"Call {tool_name}", action=ActionType.CALL_TOOL, tool_name=tool_name, tool_arguments=arguments)


def _decide(answer: str, action: str, **parameters) -> ActionDecision:
    return ActionDecision(
        thought="Propose the remediation",
        action=ActionType.FINAL,
        answer=answer,
        proposed_decision=ProposedDecision(
            intent="Operational change",
            recommended_action=action,
            parameters=parameters,
            reasoning="Based on the gathered evidence.",
            confidence=0.9,
        ),
    )


def _statuses(evaluation) -> dict[str, PolicyStatus]:
    return {result.rule_name: result.effective_status for result in evaluation.rule_results}


@pytest.mark.asyncio
async def test_low_impact_remediation_with_complete_incident_listing_is_allowed():
    runtime = _runtime([
        _call("list_service_incidents", service_name="recommendation-worker"),
        _decide(
            "Scale recommendation-worker for INC-903: a P2 incident with a 10.0% blast radius.",
            "scale_workers",
            incident_id="INC-903",
            blast_radius_pct=10,
        ),
    ])
    state = await runtime.run("Can we scale recommendation-worker for INC-903 with a 10% blast radius?")

    assert state.answer_verified is True
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED


@pytest.mark.asyncio
async def test_single_incident_lookup_cannot_prove_the_service_is_not_frozen():
    runtime = _runtime([
        _call("get_incident_telemetry", incident_id="INC-903"),
        _decide("Scale recommendation-worker for INC-903.", "scale_workers", incident_id="INC-903", blast_radius_pct=10),
    ])
    state = await runtime.run("Scale recommendation-worker?")

    assert _statuses(state.policy_evaluation)["active_p0_incident_freeze"] == PolicyStatus.REQUIRES_APPROVAL
    assert "list_service_incidents" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_blast_radius_above_cap_requires_approval():
    runtime = _runtime([
        _call("simulate_remediation_action", service_name="checkout-api", action="rolling_restart", blast_radius_pct=40),
        _decide(
            "A rolling restart of checkout-api has a 40.0% blast radius.",
            "rolling_restart",
            service_name="checkout-api",
            blast_radius_pct=40,
        ),
    ])
    state = await runtime.run("Simulate a 40% restart of checkout-api.")

    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "exceeds 25.0% threshold" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_change_during_active_p0_is_frozen():
    runtime = _runtime([
        _call("get_incident_telemetry", incident_id="INC-901"),
        _decide("INC-901 is an active P0 on auth-service.", "apply_config_change", service_name="auth-service"),
    ])
    state = await runtime.run("Apply a config update to auth-service.")

    assert state.policy_evaluation.final_status == PolicyStatus.BLOCKED
    assert "Production freeze enforced" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_security_baseline_violation_blocks_deployment():
    runtime = _runtime([
        _call("inspect_service_config", service_name="legacy-billing"),
        _decide("legacy-billing runs TLS 1.1 with its debug port open.", "deploy_service", service_name="legacy-billing"),
    ])
    state = await runtime.run("Deploy legacy-billing.")

    assert state.policy_evaluation.final_status == PolicyStatus.BLOCKED
    assert "Security baseline violation" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_authorized_emergency_override_allows_failover():
    runtime = _runtime([
        _call("list_service_incidents", service_name="payments-core"),
        _decide(
            "Execute the gateway failover for INC-904 under its authorized emergency override.",
            "failover_secondary_gateway",
            incident_id="INC-904",
        ),
    ])
    state = await runtime.run("Execute failover for INC-904.")

    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED


# ---------------------------------------------------------------- regressions


@pytest.mark.asyncio
async def test_model_cannot_grant_itself_an_emergency_override():
    # Before: {"severity": "P2", "override_authorized": true} in the proposal
    # was read as fact, and the freeze on INC-901 was lifted.
    runtime = _runtime([
        _call("list_service_incidents", service_name="auth-service"),
        _decide(
            "Roll back auth-service.",
            "rollback_deployment",
            service_name="auth-service",
            severity="P2",
            override_authorized=True,
            blast_radius_pct=5,
        ),
    ])
    state = await runtime.run("Roll back auth-service.")

    assert _statuses(state.policy_evaluation)["active_p0_incident_freeze"] == PolicyStatus.BLOCKED


def test_string_false_override_is_not_authorized():
    # Before: bool("false") is True, so this record lifted the freeze.
    ledger = EvidenceLedger()
    ledger.add(
        source=EvidenceSource.RECORD_LOOKUP,
        query_or_input="{}",
        data={"incident_id": "INC-1", "service_name": "svc", "severity": "P0", "status": "active", "override_authorized": "false"},
        provenance="test",
        summary="",
    )
    proposal = ProposedDecision(
        intent="change", recommended_action="restart_service", parameters={"service_name": "svc"}, reasoning="r", confidence=0.5
    )
    evaluation = _engine().evaluate(proposal, DecisionContext(subject_or_entity="t", evidence_ledger=ledger))
    assert _statuses(evaluation)["active_p0_incident_freeze"] == PolicyStatus.BLOCKED


@pytest.mark.asyncio
async def test_change_with_no_evidence_is_not_allowed():
    runtime = _runtime([_decide("Restart checkout-api.", "rolling_restart", service_name="checkout-api")])
    state = await runtime.run("Restart checkout-api.")

    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL


@pytest.mark.asyncio
async def test_understated_blast_radius_uses_the_assessed_one():
    runtime = _runtime([
        _call("list_service_incidents", service_name="checkout-api"),
        _call("simulate_remediation_action", service_name="checkout-api", action="rolling_restart", blast_radius_pct=40),
        _decide("Restart checkout-api.", "rolling_restart", service_name="checkout-api", blast_radius_pct=5),
    ])
    state = await runtime.run("Restart checkout-api.")

    assert _statuses(state.policy_evaluation)["production_blast_radius_cap"] == PolicyStatus.REQUIRES_APPROVAL


@pytest.mark.asyncio
async def test_read_only_action_is_not_governed():
    runtime = _runtime([_decide("Investigate auth-service further.", "investigate_incident", service_name="auth-service")])
    state = await runtime.run("What next for auth-service?")

    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED


# ---------------------------------------------------------------- server


def test_incident_listing_is_complete_and_unknown_services_are_not_empty_listings():
    listing = list_service_incidents("Payments-Core")
    assert listing.ok and [incident.incident_id for incident in listing.incidents] == ["INC-904"]
    assert listing.incidents[0].override_authorized is True

    unknown = list_service_incidents("auth-servce")
    assert not unknown.ok and unknown.incidents is None


def test_incident_telemetry_reports_backlog_and_unknown_ids():
    assert get_incident_telemetry("inc-903").queue_backlog_items == 52000
    missing = get_incident_telemetry("INC-999")
    assert not missing.ok and missing.service_name is None


def test_simulation_uses_real_replica_count():
    result = simulate_remediation_action("checkout-api", "rolling_restart", 20)
    assert (result.current_replicas, result.affected_nodes, result.remaining_nodes) == (12, 3, 9)  # ceil(2.4)

    legacy = simulate_remediation_action("legacy-billing", "rolling_restart", 50)
    assert (legacy.current_replicas, legacy.affected_nodes) == (2, 1)

    assert not simulate_remediation_action("no-such-service", "restart", 10).ok
    with pytest.raises(ValueError):
        simulate_remediation_action("checkout-api", "restart", 150)


def test_corrupt_registry_fails_instead_of_looking_empty(tmp_path, monkeypatch):
    (tmp_path / "incidents.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "service_configs.json").write_text(json.dumps([]), encoding="utf-8")
    monkeypatch.setattr(ops_server, "DATA_DIR", tmp_path)
    with pytest.raises(json.JSONDecodeError):
        list_service_incidents("auth-service")
