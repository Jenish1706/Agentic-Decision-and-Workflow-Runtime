"""MCP Enterprise Operations Server: Production incident telemetry, configuration drift audit,
and blast-radius remediation simulation.

Enables agents to:
1. Inspect live production incident diagnostics (severity, error rate, p99 latency, affected regions).
2. Audit service configurations against security baselines (TLS version, exposed debug ports, replica counts).
3. Simulate remediation actions and assess blast radius prior to execution.
"""

import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel

from mcp.server.mcpserver import MCPServer
from ...config import REPO_ROOT

server = MCPServer("enterprise-operations")

DATA_DIR = REPO_ROOT / "domains" / "enterprise_operations" / "data"


class IncidentTelemetryResult(BaseModel):
    ok: bool
    incident_id: str
    service_name: str | None = None
    severity: str | None = None
    status: str | None = None
    title: str | None = None
    error_rate_pct: float | None = None
    p99_latency_ms: float | None = None
    affected_regions: list[str] = []
    impacted_users_estimate: int | None = None
    root_cause_summary: str | None = None
    recommended_action: str | None = None
    override_authorized: bool = False
    blast_radius_pct: float | None = None
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
    debug_port_open: bool = False
    drift_detected: bool = False
    drift_details: str | None = None
    summary: str
    error: str | None = None


class RemediationSimulationResult(BaseModel):
    ok: bool
    service_name: str
    action: str
    blast_radius_pct: float
    affected_nodes: int
    estimated_downtime_seconds: int
    safety_classification: str
    summary: str


def _load_json(filename: str) -> list[dict[str, Any]]:
    path = DATA_DIR / filename
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


@server.tool()
def get_incident_telemetry(incident_id: str) -> IncidentTelemetryResult:
    """Retrieve production incident diagnostic telemetry by ID (e.g. 'INC-901', 'INC-902').

    Returns severity (P0/P1/P2), error rates, p99 latency, root cause summary, and recommended action.
    """
    clean_id = incident_id.strip().upper()
    incidents = _load_json("incidents.json")

    for inc in incidents:
        if inc.get("incident_id", "").upper() == clean_id:
            summary = (
                f"Incident {clean_id} [{inc.get('severity', 'P3')}] ({inc.get('service_name')}): "
                f"{inc.get('title')}. Error rate: {inc.get('error_rate_pct')}%, "
                f"p99 latency: {inc.get('p99_latency_ms')}ms. "
                f"Root cause: {inc.get('root_cause_summary')} "
                f"Recommended action: {inc.get('recommended_action')}"
            )
            return IncidentTelemetryResult(
                ok=True,
                incident_id=clean_id,
                service_name=inc.get("service_name"),
                severity=inc.get("severity"),
                status=inc.get("status"),
                title=inc.get("title"),
                error_rate_pct=inc.get("error_rate_pct"),
                p99_latency_ms=inc.get("p99_latency_ms"),
                affected_regions=inc.get("affected_regions", []),
                impacted_users_estimate=inc.get("impacted_users_estimate"),
                root_cause_summary=inc.get("root_cause_summary"),
                recommended_action=inc.get("recommended_action"),
                override_authorized=inc.get("override_authorized", False),
                blast_radius_pct=inc.get("blast_radius_pct"),
                summary=summary,
            )

    return IncidentTelemetryResult(
        ok=False,
        incident_id=clean_id,
        summary=f"Incident {clean_id} not found in enterprise operations registry.",
        error=f"No record matching incident '{clean_id}'.",
    )


@server.tool()
def inspect_service_config(service_name: str) -> ServiceConfigResult:
    """Audit service deployment configuration and detect baseline drift or security policy violations.

    Returns replica status, TLS version, debug port exposure, and drift details.
    """
    clean_name = service_name.strip().lower()
    configs = _load_json("service_configs.json")

    for cfg in configs:
        if cfg.get("service_name", "").lower() == clean_name:
            summary = (
                f"Service {clean_name} ({cfg.get('tier')}): Replicas={cfg.get('current_replicas')}/{cfg.get('desired_replicas')}, "
                f"TLS={cfg.get('tls_version')}, DebugPortOpen={cfg.get('debug_port_open')}, "
                f"DriftDetected={cfg.get('drift_detected')}."
            )
            if cfg.get("drift_details"):
                summary += f" Details: {cfg.get('drift_details')}"

            return ServiceConfigResult(
                ok=True,
                service_name=clean_name,
                environment=cfg.get("environment"),
                tier=cfg.get("tier"),
                desired_replicas=cfg.get("desired_replicas"),
                current_replicas=cfg.get("current_replicas"),
                tls_version=cfg.get("tls_version"),
                debug_port_open=cfg.get("debug_port_open", False),
                drift_detected=cfg.get("drift_detected", False),
                drift_details=cfg.get("drift_details"),
                summary=summary,
            )

    return ServiceConfigResult(
        ok=False,
        service_name=clean_name,
        summary=f"Service '{clean_name}' not found in configuration inventory.",
        error=f"Service '{clean_name}' could not be resolved.",
    )


@server.tool()
def simulate_remediation_action(
    service_name: str, action: str, blast_radius_pct: float
) -> RemediationSimulationResult:
    """Simulate an operational remediation action (e.g. 'rolling_restart', 'rollback_deployment', 'scale_workers')
    and calculate estimated impact and downtime.
    """
    radius = max(0.0, min(100.0, float(blast_radius_pct)))
    nodes = max(1, int(round(12 * (radius / 100.0))))
    downtime = 0 if radius <= 10.0 else int(radius * 1.5)

    classification = "SAFE_LOW_IMPACT" if radius <= 10.0 else ("MODERATE_IMPACT" if radius <= 25.0 else "HIGH_RISK_FLEET_WIDE")

    summary = (
        f"Remediation simulation for {service_name} [{action}]: Blast radius is {radius:.1f}%, "
        f"affecting ~{nodes} node(s) with {downtime}s estimated disruption. Risk level: {classification}."
    )

    return RemediationSimulationResult(
        ok=True,
        service_name=service_name,
        action=action,
        blast_radius_pct=radius,
        affected_nodes=nodes,
        estimated_downtime_seconds=downtime,
        safety_classification=classification,
        summary=summary,
    )


if __name__ == "__main__":
    server.run()
