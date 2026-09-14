"""Enterprise operations (SRE) policy rules.

Rules (limits come from `thresholds` in domain_manifest.yaml):
    active_p0_incident_freeze     change to a service with an active P0 and no
                                  authorized emergency override              -> BLOCKED
    production_blast_radius_cap   blast radius > max_autonomous_blast_radius_pct
                                  (unless an emergency override covers it)   -> REQUIRES_APPROVAL
    security_baseline_compliance  deploy/config change with TLS < min_tls_version
                                  or an open debug port                      -> BLOCKED
    Any fact the evidence does not establish                                  -> REQUIRES_APPROVAL

WHAT CHANGED AND WHY
    - Facts come from evidence, never from the proposal. The old rules read
      `severity` and `override_authorized` from the proposal's parameters, so
      a model could grant itself an emergency override by writing
      {"override_authorized": true}. `bool("false")` was also True, so even a
      string "false" override counted as authorized.
    - A missing fact is not a pass. Previously an agent that looked nothing
      up had no severity, no TLS version and no blast radius - and every rule
      allowed the change.
    - "No active P0 incident" is accepted only from a complete per-service
      listing (`list_service_incidents`); a single incident lookup cannot rule
      out another open incident on the same service. Any record showing an
      active, un-overridden P0 still blocks on its own.
    - Blast radius is the LARGER of the proposal's own figure and any figure
      in evidence for the service (incident assessment, simulation).
    - Each rule applies to the actions it governs: the freeze and blast-radius
      cap to anything that changes production, the security baseline to
      deployments and configuration changes. Read-only actions (inspect,
      investigate, ...) are exempt; unrecognised actions are governed.
    - The manifest used to list a `safe_automated_restart` rule ("<= 10% ->
      ALLOWED"). Under deny-overrides precedence an ALLOWED rule can never
      change an outcome, and no code implemented it; it has been removed.
"""

from collections.abc import Mapping
from typing import Any

from agent_runtime.agent.facts import (
    as_bool,
    find_entity_records,
    is_governed_action,
    iter_scoped_mappings,
    normalize_identifier,
    numeric_candidates,
    proposal_number,
    proposal_target,
    resolve_fact,
)
from agent_runtime.agent.policy import DecisionContext, PolicyEngine, PolicyRuleResult, ProposedDecision

REQUIRED_THRESHOLDS = ("max_autonomous_blast_radius_pct", "min_tls_version")

_READ_ONLY = frozenset(
    {"investigate", "inspect", "monitor", "audit", "diagnose", "escalate", "page", "notify", "review", "simulate",
     "observe", "report", "analyze", "analyse"}
)
_CHANGING = frozenset(
    {"restart", "scale", "failover", "rollback", "revert", "deploy", "apply", "change", "update", "patch", "config",
     "configuration", "migrate", "drain", "restore", "release", "upgrade"}
)
_DEPLOYING = frozenset(
    {"deploy", "deployment", "release", "promote", "config", "configuration", "change", "update", "upgrade", "migrate",
     "patch", "enable", "expose", "launch"}
)
# Remediations that restart or shift existing workloads without shipping a
# new build or configuration.
_NON_DEPLOYING = _READ_ONLY | frozenset({"restart", "scale", "failover", "drain"})

_CLOSED_STATUSES = frozenset({"resolved", "closed"})


def _target_service(proposal: ProposedDecision, ctx: DecisionContext) -> str | None:
    """The service a proposal changes: named directly, or found through the
    incident it names (only via evidence attributed to that incident)."""
    service = proposal_target(proposal.parameters, "service_name", "service")
    if service is not None:
        return service
    incident = proposal_target(proposal.parameters, "incident_id", "incident")
    if incident is None:
        return None
    owner = resolve_fact(ctx.evidence_ledger, "service_name", entity_key="incident_id", entity_value=incident)
    return str(owner.value) if owner.conclusive else None


def _incidents(ctx: DecisionContext, service: str) -> list[Mapping[str, Any]]:
    records = find_entity_records(
        ctx.evidence_ledger, entity_key="service_name", entity_value=service, required_fields=("incident_id", "severity")
    )
    return [record for _, record in records]


def _is_open(record: Mapping[str, Any]) -> bool:
    # An incident with no recorded status is treated as open (fail closed).
    status = record.get("status")
    return status is None or normalize_identifier(status) not in _CLOSED_STATUSES


def _is_open_p0(record: Mapping[str, Any]) -> bool:
    severity = record.get("severity")
    # Unknown severity counts as P0 for the freeze: better an unnecessary
    # block on malformed data than a change during a real outage.
    return _is_open(record) and (severity is None or normalize_identifier(severity) == "p0")


def _has_complete_listing(ctx: DecisionContext, service: str) -> bool:
    """True when evidence holds a full incident listing for the service."""
    target = normalize_identifier(service)
    return any(
        scope == target and isinstance(mapping.get("incidents"), tuple)
        for _, scope, mapping in iter_scoped_mappings(ctx.evidence_ledger, "service_name")
    )


def _override_ids(ctx: DecisionContext, service: str) -> list[str]:
    """Open incidents on the service that carry an authorized emergency override."""
    return sorted({
        str(record["incident_id"])
        for record in _incidents(ctx, service)
        if _is_open(record) and as_bool(record.get("override_authorized")) is True
    })


def create_policy_engine(thresholds: Mapping[str, float]) -> PolicyEngine:
    max_radius = float(thresholds["max_autonomous_blast_radius_pct"])
    min_tls = float(thresholds["min_tls_version"])

    def active_p0_incident_freeze(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "active_p0_incident_freeze"
        if not is_governed_action(proposal.recommended_action, governed=_CHANGING, exempt=_READ_ONLY):
            return PolicyRuleResult.allowed(rule, f"'{proposal.recommended_action}' is read-only; freeze not applicable.")
        service = _target_service(proposal, ctx)
        if service is None:
            return PolicyRuleResult.requires_approval(rule, "The proposal does not identify the service it changes.")

        # A record whose override is anything other than an explicit true
        # (false, null, "false") is un-overridden.
        frozen_by = sorted({
            str(record["incident_id"])
            for record in _incidents(ctx, service)
            if _is_open_p0(record) and as_bool(record.get("override_authorized")) is not True
        })
        if frozen_by:
            return PolicyRuleResult.blocked(
                rule,
                f"Production freeze enforced: {service} has active P0 incident(s) {frozen_by} without an authorized "
                "emergency override; non-emergency mutations are locked out.",
            )
        if not _has_complete_listing(ctx, service):
            return PolicyRuleResult.requires_approval(
                rule,
                f"Incident status of {service} is not established: only a complete listing "
                "(list_service_incidents) can show that no active P0 incident exists.",
            )
        overrides = _override_ids(ctx, service)
        if overrides:
            return PolicyRuleResult.allowed(rule, f"Active incident(s) {overrides} on {service} carry an authorized emergency override.")
        return PolicyRuleResult.allowed(rule, f"No active P0 incident on {service}.")

    def production_blast_radius_cap(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "production_blast_radius_cap"
        if not is_governed_action(proposal.recommended_action, governed=_CHANGING, exempt=_READ_ONLY):
            return PolicyRuleResult.allowed(rule, f"'{proposal.recommended_action}' is read-only; not applicable.")
        service = _target_service(proposal, ctx)
        if service is None:
            return PolicyRuleResult.requires_approval(rule, "The proposal does not identify the service it changes.")

        proposed = proposal_number(proposal.parameters, "blast_radius_pct")
        assessed = resolve_fact(ctx.evidence_ledger, "blast_radius_pct", entity_key="service_name", entity_value=service)
        radii = [radius for radius in (proposed, *numeric_candidates(assessed)) if radius is not None]
        overrides = _override_ids(ctx, service)

        # An authorized emergency override IS the human sign-off this rule
        # would otherwise ask for, so it covers an unassessed or large radius.
        if not radii:
            if overrides:
                return PolicyRuleResult.allowed(rule, f"Blast radius not assessed, but covered by emergency override {overrides}.")
            return PolicyRuleResult.requires_approval(
                rule, f"The blast radius of this change to {service} is not established by the proposal or evidence."
            )
        radius = max(radii)
        if radius > max_radius:
            if overrides:
                return PolicyRuleResult.allowed(
                    rule, f"Blast radius {radius:.1f}% exceeds {max_radius:.1f}%, but is covered by emergency override {overrides}."
                )
            return PolicyRuleResult.requires_approval(
                rule,
                f"Remediation blast radius ({radius:.1f}%) exceeds {max_radius:.1f}% threshold. "
                "SRE Incident Commander sign-off required.",
            )
        if proposed is None and not assessed.attributed:
            return PolicyRuleResult.requires_approval(
                rule, f"The blast radius in evidence could not be attributed to {service}, so it cannot support an approval."
            )
        return PolicyRuleResult.allowed(rule, f"Blast radius {radius:.1f}% is within the {max_radius:.1f}% autonomous limit.")

    def security_baseline_compliance(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "security_baseline_compliance"
        if not is_governed_action(proposal.recommended_action, governed=_DEPLOYING, exempt=_NON_DEPLOYING):
            return PolicyRuleResult.allowed(
                rule, f"'{proposal.recommended_action}' ships no new build or configuration; not applicable."
            )
        service = _target_service(proposal, ctx)
        if service is None:
            return PolicyRuleResult.requires_approval(rule, "The proposal does not identify the service it deploys.")

        tls = resolve_fact(ctx.evidence_ledger, "tls_version", entity_key="service_name", entity_value=service)
        debug = resolve_fact(ctx.evidence_ledger, "debug_port_open", entity_key="service_name", entity_value=service)
        tls_versions = numeric_candidates(tls)
        violations = []
        if tls_versions and min(tls_versions) < min_tls:
            violations.append(f"TLS {min(tls_versions)} < {min_tls} baseline")
        if any(as_bool(value) is True for value in debug.candidates):
            violations.append("debug port is open to ingress")
        if violations:
            return PolicyRuleResult.blocked(rule, f"Security baseline violation: {'; '.join(violations)}. Action blocked.")

        if not (tls.conclusive and debug.conclusive and tls_versions and as_bool(debug.value) is False):
            return PolicyRuleResult.requires_approval(
                rule,
                f"Security baseline of {service} is not established ({tls.describe()}; {debug.describe()}); "
                "inspect its configuration first.",
            )
        return PolicyRuleResult.allowed(rule, f"{service} meets the security baseline (TLS {tls_versions[0]}, debug port closed).")

    return PolicyEngine(
        rules=[active_p0_incident_freeze, production_blast_radius_cap, security_baseline_compliance],
        thresholds=thresholds,
        name="enterprise_operations",
    )
