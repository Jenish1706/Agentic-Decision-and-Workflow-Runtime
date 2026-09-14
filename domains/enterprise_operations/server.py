"""MCP server for the Enterprise Systems Operations domain.

Tools
    get_incident_telemetry       one incident's diagnostics                  (record)
    list_service_incidents       every incident recorded for one service     (record)
    inspect_service_config       deployment configuration and drift          (record)
    simulate_remediation_action  capacity impact of a proposed remediation   (statistics)

WHY `list_service_incidents` EXISTS
    The production-freeze rule (policies.py) must know whether a service has
    an active P0 incident. Looking up one incident cannot prove that no OTHER
    incident is open on the same service, so the rule accepts "no active P0"
    only from a complete per-service listing. Before this tool existed there
    was no way to establish that from evidence at all - the old rule simply
    read `severity` and `override_authorized` from the model's own proposal.

FIXES IN THIS VERSION
    - Data files that are missing or corrupt now raise. The old loader
      returned [] for both, so a broken incident registry looked exactly like
      "no incidents", i.e. like a safe time to change production.
    - Absent fields are null, not invented: `override_authorized` and
      `debug_port_open` used to default to False, turning "unknown" into a
      confident answer.
    - The simulation assumed every service runs 12 nodes. It now uses the
      service's real `current_replicas` and refuses services it cannot find.
    - The simulation no longer emits a downtime estimate (`radius * 1.5`
      seconds - a formula with no basis) or its own risk classification
      (hard-coded 10%/25% cut-offs that duplicated, and could silently drift
      from, the policy threshold). Deciding what is too risky is the policy
      engine's job; this tool reports capacity facts.
"""

import json
import math
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parent / "data"

server = MCPServer("enterprise-operations")


class IncidentRecord(BaseModel):
    incident_id: str
    service_name: str | None = None
    severity: str | None = None
    status: str | None = None
    title: str | None = None
    error_rate_pct: float | None = None
    p99_latency_ms: float | None = None
    affected_regions: list[str] = Field(default_factory=list)
    impacted_users_estimate: int | None = None
    queue_backlog_items: int | None = None
    root_cause_summary: str | None = None
    recommended_action: str | None = None
    # Blast radius of the recommended remediation, when one has been assessed.
    blast_radius_pct: float | None = None
    override_authorized: bool | None = None


class IncidentTelemetryResult(IncidentRecord):
    ok: bool
    summary: str
    error: str | None = None


class ServiceIncidentsResult(BaseModel):
    ok: bool
    service_name: str
    incident_count: int = 0
    # null (not []) when the service is unknown, so a typo can never read as
    # "complete listing, no incidents".
    incidents: list[IncidentRecord] | None = None
    summary: str
    error: str | None = None


class ServiceConfigResult(BaseModel):
    ok: bool
    service_name: str
    environment: str | None = None
    tier: str | None = None
    desired_replicas: int | None = None
    current_replicas: int | None = None
    tls_version: float | None = None
    debug_port_open: bool | None = None
    drift_detected: bool | None = None
    drift_details: str | None = None
    summary: str
    error: str | None = None


class RemediationSimulationResult(BaseModel):
    ok: bool
    service_name: str
    action: str
    blast_radius_pct: float | None = None
    current_replicas: int | None = None
    affected_nodes: int | None = None
    remaining_nodes: int | None = None
    summary: str
    error: str | None = None


def _load_records(filename: str) -> list[dict[str, Any]]:
    """Load one registry file. Raises instead of returning an empty list: an
    unreadable registry must fail the tool call, not look like an empty one."""
    path = DATA_DIR / filename
    records = json.loads(path.read_text(encoding="utf-8"))  # FileNotFoundError / JSONDecodeError propagate
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} must contain a JSON list of objects.")
    return records


def _incident(record: dict[str, Any]) -> IncidentRecord:
    fields = {name: record.get(name) for name in IncidentRecord.model_fields}
    fields["affected_regions"] = record.get("affected_regions") or []
    return IncidentRecord.model_validate(fields)


@server.tool(meta={"category": "record"})
def get_incident_telemetry(incident_id: str) -> IncidentTelemetryResult:
    """Retrieve one production incident by ID (e.g. "INC-901"): service, severity (P0-P3), status,
    error rate, p99 latency, impact, root cause, recommended action, the assessed blast radius of
    that action, and whether an emergency override is authorized."""
    clean_id = incident_id.strip().upper()
    for record in _load_records("incidents.json"):
        if str(record.get("incident_id", "")).upper() != clean_id:
            continue
        incident = _incident(record)
        summary = (
            f"Incident {clean_id} [{incident.severity}, {incident.status}] on {incident.service_name}: "
            f"{incident.title}. Error rate {incident.error_rate_pct}%, p99 latency {incident.p99_latency_ms}ms. "
            f"Root cause: {incident.root_cause_summary} Recommended action: {incident.recommended_action}."
        )
        return IncidentTelemetryResult(ok=True, summary=summary, **incident.model_dump())

    return IncidentTelemetryResult(
        ok=False,
        incident_id=clean_id,
        summary=f"Incident {clean_id} not found in the operations registry.",
        error=f"No incident matching '{clean_id}'.",
    )


@server.tool(meta={"category": "record"})
def list_service_incidents(service_name: str) -> ServiceIncidentsResult:
    """List EVERY incident recorded for one service (any severity or status), e.g.
    "auth-service". Use this to establish whether a service has an active P0 incident before
    proposing a change to it."""
    clean_name = service_name.strip().lower()
    incidents = [
        _incident(record)
        for record in _load_records("incidents.json")
        if str(record.get("service_name", "")).lower() == clean_name
    ]
    known_services = {str(record.get("service_name", "")).lower() for record in _load_records("service_configs.json")}
    if not incidents and clean_name not in known_services:
        return ServiceIncidentsResult(
            ok=False,
            service_name=clean_name,
            summary=f"Service '{clean_name}' is not in the service inventory.",
            error=f"Unknown service '{clean_name}'.",
        )

    described = ", ".join(f"{i.incident_id} ({i.severity}, {i.status})" for i in incidents) or "none"
    return ServiceIncidentsResult(
        ok=True,
        service_name=clean_name,
        incident_count=len(incidents),
        incidents=incidents,
        summary=f"{len(incidents)} incident(s) recorded for {clean_name}: {described}.",
    )


@server.tool(meta={"category": "record"})
def inspect_service_config(service_name: str) -> ServiceConfigResult:
    """Audit one service's deployment configuration: replica counts, TLS version, debug port
    exposure, and configuration drift."""
    clean_name = service_name.strip().lower()
    for record in _load_records("service_configs.json"):
        if str(record.get("service_name", "")).lower() != clean_name:
            continue
        summary = (
            f"Service {clean_name} ({record.get('tier')}): replicas {record.get('current_replicas')}/"
            f"{record.get('desired_replicas')}, TLS {record.get('tls_version')}, debug port open "
            f"{record.get('debug_port_open')}, drift detected {record.get('drift_detected')}."
        )
        if record.get("drift_details"):
            summary += f" Details: {record['drift_details']}"
        return ServiceConfigResult(
            ok=True,
            service_name=clean_name,
            environment=record.get("environment"),
            tier=record.get("tier"),
            desired_replicas=record.get("desired_replicas"),
            current_replicas=record.get("current_replicas"),
            tls_version=record.get("tls_version"),
            debug_port_open=record.get("debug_port_open"),
            drift_detected=record.get("drift_detected"),
            drift_details=record.get("drift_details"),
            summary=summary,
        )

    return ServiceConfigResult(
        ok=False,
        service_name=clean_name,
        summary=f"Service '{clean_name}' not found in the configuration inventory.",
        error=f"Unknown service '{clean_name}'.",
    )


@server.tool(meta={"category": "statistics"})
def simulate_remediation_action(service_name: str, action: str, blast_radius_pct: float) -> RemediationSimulationResult:
    """Estimate the capacity impact of a remediation (e.g. "rolling_restart", "scale_workers")
    that takes `blast_radius_pct` percent of a service's current replicas out of rotation.

    Args:
        service_name: Service to remediate, e.g. "checkout-api".
        action: Remediation name.
        blast_radius_pct: Share of the service's replicas affected, greater than 0 and at most 100.
    """
    # Out-of-range input used to be clamped silently (150 -> 100, -5 -> 0),
    # which hid a malformed request behind a plausible-looking result.
    if not 0.0 < blast_radius_pct <= 100.0:
        raise ValueError(f"blast_radius_pct must be in (0, 100], got {blast_radius_pct}.")

    clean_name = service_name.strip().lower()
    config = next(
        (record for record in _load_records("service_configs.json") if str(record.get("service_name", "")).lower() == clean_name),
        None,
    )
    replicas = config.get("current_replicas") if config else None
    if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas <= 0:
        return RemediationSimulationResult(
            ok=False,
            service_name=clean_name,
            action=action,
            summary=f"Cannot simulate: no replica count on record for '{clean_name}'.",
            error=f"Unknown service or missing current_replicas for '{clean_name}'.",
        )

    # Round UP: an impact estimate should never understate how many nodes are hit.
    affected = min(replicas, max(1, math.ceil(replicas * blast_radius_pct / 100.0)))
    return RemediationSimulationResult(
        ok=True,
        service_name=clean_name,
        action=action,
        blast_radius_pct=float(blast_radius_pct),
        current_replicas=replicas,
        affected_nodes=affected,
        remaining_nodes=replicas - affected,
        summary=(
            f"{action} on {clean_name}: blast radius {blast_radius_pct:.1f}% affects {affected} of "
            f"{replicas} replicas, leaving {replicas - affected} serving."
        ),
    )


if __name__ == "__main__":
    server.run()
