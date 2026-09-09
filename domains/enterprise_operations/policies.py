"""Enterprise Operations & Production Governance Policy Rules.

Encodes deterministic SRE and platform security constraints:
1. production_blast_radius_cap: blast_radius_pct > 25.0% -> REQUIRES_APPROVAL
2. active_p0_incident_freeze: severity == 'P0' and not override_authorized -> BLOCKED
3. security_baseline_compliance: tls_version < 1.3 or debug_port_open -> BLOCKED
4. safe_automated_restart: blast_radius_pct <= 10.0% -> ALLOWED
"""

from typing import Any
from agent_runtime.agent.policy import (
    DecisionContext,
    PolicyEngine,
    PolicyRuleResult,
    PolicyStatus,
    ProposedDecision,
)


def create_enterprise_ops_policy_engine() -> PolicyEngine:
    """Builds the deterministic SRE and operational policy engine."""

    def _get_metric(metric_name: str, proposal: ProposedDecision, ctx: DecisionContext) -> float | None:
        if metric_name in proposal.parameters:
            v = proposal.parameters[metric_name]
            if v is not None:
                return float(v)

        if ctx.evidence_ledger:
            facts = ctx.evidence_ledger.extract_numerical_facts()
            for k, v in facts.items():
                if metric_name in k.lower():
                    return float(v)
        return None

    def _get_string(field_name: str, proposal: ProposedDecision, ctx: DecisionContext) -> str | None:
        if field_name in proposal.parameters:
            return str(proposal.parameters[field_name])

        if ctx.evidence_ledger:
            for it in ctx.evidence_ledger.all():
                if isinstance(it.data, dict) and field_name in it.data:
                    return str(it.data[field_name])
        return None

    def _get_bool(flag_name: str, proposal: ProposedDecision, ctx: DecisionContext) -> bool | None:
        if flag_name in proposal.parameters:
            return bool(proposal.parameters[flag_name])

        if ctx.evidence_ledger:
            for it in ctx.evidence_ledger.all():
                if isinstance(it.data, dict) and flag_name in it.data:
                    return bool(it.data[flag_name])
        return None

    def blast_radius_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        radius = _get_metric("blast_radius_pct", proposal, ctx)
        if radius is not None and radius > 25.0:
            return PolicyRuleResult(
                rule_name="production_blast_radius_cap",
                passed=False,
                status=PolicyStatus.REQUIRES_APPROVAL,
                message=f"Remediation blast radius ({radius:.1f}%) exceeds 25.0% threshold. SRE Incident Commander sign-off required.",
            )

        return PolicyRuleResult(
            rule_name="production_blast_radius_cap",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="Blast radius is within autonomous operational boundaries.",
        )

    def p0_freeze_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        sev = _get_string("severity", proposal, ctx)
        override = _get_bool("override_authorized", proposal, ctx)

        if sev and sev.upper() == "P0" and not override:
            return PolicyRuleResult(
                rule_name="active_p0_incident_freeze",
                passed=False,
                status=PolicyStatus.BLOCKED,
                message="Production freeze enforced: non-emergency mutations locked out during active P0 incident without authorized override.",
            )

        return PolicyRuleResult(
            rule_name="active_p0_incident_freeze",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="No P0 freeze violation detected.",
        )

    def security_baseline_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        tls = _get_metric("tls_version", proposal, ctx)
        debug_port = _get_bool("debug_port_open", proposal, ctx)

        if (tls is not None and tls < 1.3) or debug_port is True:
            reasons = []
            if tls is not None and tls < 1.3:
                reasons.append(f"TLS {tls} < 1.3 baseline")
            if debug_port is True:
                reasons.append("Debug port is open to public ingress")

            return PolicyRuleResult(
                rule_name="security_baseline_compliance",
                passed=False,
                status=PolicyStatus.BLOCKED,
                message=f"Security baseline violation: {'; '.join(reasons)}. Action blocked.",
            )

        return PolicyRuleResult(
            rule_name="security_baseline_compliance",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="Configuration complies with enterprise security baseline.",
        )

    return PolicyEngine(
        rules=[
            blast_radius_rule,
            p0_freeze_rule,
            security_baseline_rule,
        ]
    )
