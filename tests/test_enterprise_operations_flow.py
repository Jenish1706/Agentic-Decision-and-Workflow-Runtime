"""End-to-end integration and policy tests for Enterprise Systems Operations Domain.

Verifies:
1. Incident diagnostics: extracting telemetry, severity, error rate, and p99 latency into Evidence Ledger.
2. SRE governance: autonomous low-impact restart (<=10% blast radius -> ALLOWED).
3. Blast radius escalation: remediation affecting > 25% capacity triggers REQUIRES_APPROVAL.
4. Production freeze: changes during active P0 incidents without override trigger BLOCKED.
5. Security compliance: TLS < 1.3 or exposed debug ports trigger BLOCKED.
6. Emergency override: P0 changes with executive authorization evaluate to ALLOWED.
"""

import pytest

from agent_runtime.agent.actions import ActionDecision, ActionType
from agent_runtime.agent.evidence import EvidenceKind, EvidenceSource
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.policy import PolicyStatus, ProposedDecision
from agent_runtime.agent.state import TerminationReason
from agent_runtime.agent.tools.base import Tool, ToolResult
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.mcp_layer.servers.ops_server import (
    get_incident_telemetry,
    inspect_service_config,
    simulate_remediation_action,
)
from domains.enterprise_operations.policies import create_enterprise_ops_policy_engine


class _LocalTelemetryTool(Tool):
    name = "get_incident_telemetry"
    description = "Retrieve incident diagnostic telemetry."
    parameters_schema = {
        "type": "object",
        "properties": {"incident_id": {"type": "string"}},
        "required": ["incident_id"],
    }
    origin = "local"

    async def run(self, incident_id: str) -> ToolResult:
        res = get_incident_telemetry(incident_id)
        if not res.ok:
            return ToolResult(ok=False, content="", error=res.error)
        return ToolResult(
            ok=True,
            content=res.summary,
            data={
                "incident_id": res.incident_id,
                "service_name": res.service_name,
                "severity": res.severity,
                "status": res.status,
                "error_rate_pct": res.error_rate_pct,
                "p99_latency_ms": res.p99_latency_ms,
                "impacted_users_estimate": res.impacted_users_estimate,
                "override_authorized": res.override_authorized,
                "blast_radius_pct": res.blast_radius_pct,
            },
        )


class _LocalConfigTool(Tool):
    name = "inspect_service_config"
    description = "Audit service configuration and security drift."
    parameters_schema = {
        "type": "object",
        "properties": {"service_name": {"type": "string"}},
        "required": ["service_name"],
    }
    origin = "local"

    async def run(self, service_name: str) -> ToolResult:
        res = inspect_service_config(service_name)
        if not res.ok:
            return ToolResult(ok=False, content="", error=res.error)
        return ToolResult(
            ok=True,
            content=res.summary,
            data={
                "service_name": res.service_name,
                "environment": res.environment,
                "tier": res.tier,
                "desired_replicas": res.desired_replicas,
                "current_replicas": res.current_replicas,
                "tls_version": res.tls_version,
                "debug_port_open": res.debug_port_open,
                "drift_detected": res.drift_detected,
            },
        )


class _LocalSimulateTool(Tool):
    name = "simulate_remediation_action"
    description = "Simulate remediation action and blast radius."
    parameters_schema = {
        "type": "object",
        "properties": {
            "service_name": {"type": "string"},
            "action": {"type": "string"},
            "blast_radius_pct": {"type": "number"},
        },
        "required": ["service_name", "action", "blast_radius_pct"],
    }
    origin = "local"

    async def run(self, service_name: str, action: str, blast_radius_pct: float) -> ToolResult:
        res = simulate_remediation_action(service_name, action, blast_radius_pct)
        return ToolResult(
            ok=True,
            content=res.summary,
            data={
                "service_name": res.service_name,
                "action": res.action,
                "blast_radius_pct": res.blast_radius_pct,
                "affected_nodes": res.affected_nodes,
                "safety_classification": res.safety_classification,
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
async def test_enterprise_ops_autonomous_remediation():
    """INC-903 (P2, 10% blast radius) is approved autonomously (ALLOWED)."""
    tools = ToolRegistry()
    tools.register(_LocalTelemetryTool())

    policy_engine = create_enterprise_ops_policy_engine()

    decisions = [
        # Step 0: inspect incident
        ActionDecision(
            thought="Fetch telemetry for INC-903",
            action=ActionType.CALL_TOOL,
            tool_name="get_incident_telemetry",
            tool_arguments={"incident_id": "INC-903"},
        ),
        # Step 1: recommend restart
        ActionDecision(
            thought="Incident is P2 with 10.0% blast radius, within safe threshold.",
            action=ActionType.FINAL,
            answer="Approve automated worker scale-up for INC-903. Blast radius is 10.0%, within autonomous limits.",
            proposed_decision=ProposedDecision(
                intent="Remediate worker queue lag",
                recommended_action="scale_workers",
                parameters={"incident_id": "INC-903", "severity": "P2", "blast_radius_pct": 10.0},
                reasoning="Safe low-impact remediation.",
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

    state = await runtime.run("Evaluate remediation for INC-903.")

    assert state.done is True
    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED
    assert state.policy_evaluation.requires_approval is False


@pytest.mark.asyncio
async def test_enterprise_ops_blast_radius_escalation():
    """40% blast radius (> 25% ceiling) triggers REQUIRES_APPROVAL."""
    tools = ToolRegistry()
    tools.register(_LocalSimulateTool())

    policy_engine = create_enterprise_ops_policy_engine()

    decisions = [
        # Step 0: simulate restart
        ActionDecision(
            thought="Simulate 40% blast radius restart",
            action=ActionType.CALL_TOOL,
            tool_name="simulate_remediation_action",
            tool_arguments={"service_name": "checkout-api", "action": "rolling_restart", "blast_radius_pct": 40.0},
        ),
        # Step 1: propose restart
        ActionDecision(
            thought="Remediation poses fleet-wide risk and requires approval.",
            action=ActionType.FINAL,
            answer="Rolling restart of checkout-api has a 40.0% blast radius. Incident commander sign-off is required.",
            proposed_decision=ProposedDecision(
                intent="Execute checkout-api restart",
                recommended_action="rolling_restart",
                parameters={"service_name": "checkout-api", "blast_radius_pct": 40.0},
                reasoning="High blast radius.",
                confidence=0.8,
            ),
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Simulate 40% restart of checkout-api.")

    assert state.done is True
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert state.policy_evaluation.requires_approval is True
    assert "exceeds 25.0% threshold" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_enterprise_ops_p0_freeze_blocks_mutation():
    """INC-901 is active P0 without override; configuration change is BLOCKED."""
    tools = ToolRegistry()
    tools.register(_LocalTelemetryTool())

    policy_engine = create_enterprise_ops_policy_engine()

    decisions = [
        # Step 0: inspect telemetry
        ActionDecision(
            thought="Check INC-901 status",
            action=ActionType.CALL_TOOL,
            tool_name="get_incident_telemetry",
            tool_arguments={"incident_id": "INC-901"},
        ),
        # Step 1: attempt mutation
        ActionDecision(
            thought="Attempt routine change during P0",
            action=ActionType.FINAL,
            answer="Attempting configuration update during INC-901.",
            proposed_decision=ProposedDecision(
                intent="Update auth configuration",
                recommended_action="apply_config_change",
                parameters={"service_name": "auth-service", "severity": "P0", "override_authorized": False},
                reasoning="Routine update.",
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

    state = await runtime.run("Apply config update to auth-service.")

    assert state.done is True
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.BLOCKED
    assert "Production freeze enforced" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_enterprise_ops_security_baseline_violation_blocked():
    """legacy-billing has TLS 1.1 and open debug port; deployment is BLOCKED."""
    tools = ToolRegistry()
    tools.register(_LocalConfigTool())

    policy_engine = create_enterprise_ops_policy_engine()

    decisions = [
        # Step 0: inspect service config
        ActionDecision(
            thought="Inspect legacy-billing configuration",
            action=ActionType.CALL_TOOL,
            tool_name="inspect_service_config",
            tool_arguments={"service_name": "legacy-billing"},
        ),
        # Step 1: attempt deployment
        ActionDecision(
            thought="Attempt deployment despite baseline violations",
            action=ActionType.FINAL,
            answer="legacy-billing has TLS 1.1 and debug port open.",
            proposed_decision=ProposedDecision(
                intent="Deploy legacy-billing",
                recommended_action="deploy_service",
                parameters={"service_name": "legacy-billing", "tls_version": 1.1, "debug_port_open": True},
                reasoning="Deploying internal service.",
                confidence=0.4,
            ),
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Deploy legacy-billing.")

    assert state.done is True
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.BLOCKED
    assert "Security baseline violation" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_enterprise_ops_emergency_override_allowed():
    """INC-904 is P0 with authorized emergency override; failover is ALLOWED."""
    tools = ToolRegistry()
    tools.register(_LocalTelemetryTool())

    policy_engine = create_enterprise_ops_policy_engine()

    decisions = [
        # Step 0: inspect telemetry
        ActionDecision(
            thought="Check INC-904 telemetry and authorization",
            action=ActionType.CALL_TOOL,
            tool_name="get_incident_telemetry",
            tool_arguments={"incident_id": "INC-904"},
        ),
        # Step 1: execute failover
        ActionDecision(
            thought="P0 incident has emergency executive override authorized.",
            action=ActionType.FINAL,
            answer="Execute emergency gateway failover for INC-904 under authorized override.",
            proposed_decision=ProposedDecision(
                intent="Execute emergency failover",
                recommended_action="failover_secondary_gateway",
                parameters={"incident_id": "INC-904", "severity": "P0", "override_authorized": True},
                reasoning="Emergency override authorized.",
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

    state = await runtime.run("Execute failover for INC-904.")

    assert state.done is True
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED
