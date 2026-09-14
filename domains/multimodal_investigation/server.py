"""MCP server for the Multimodal Business Investigation domain.

HONEST SCOPE
    The tools are named after what a vision pipeline does - read a scanned
    invoice, read values off a chart - but no image is processed here. Each
    "scan" and "chart" is a JSON fixture holding the fields a vision-language
    model would extract. That keeps the domain deterministic and testable,
    and it exercises the rest of the runtime (evidence ledger, numerical
    verification, policy) exactly as a real extractor behind the same tool
    interface would. Swapping in a real VLM means changing these function
    bodies, not the agent.

Tools
    extract_document_metrics  fields, line items and arithmetic of one invoice   (vision)
    analyze_chart_data        series, summary metrics and SLA status of a chart  (vision)
    list_visual_artifacts     which invoices and charts exist                    (record)

    Listing is a `record` lookup, not `vision`: it reads a directory, not an
    image, so it must not consume the extraction budget.

WHAT THE TOOLS COMPUTE INSTEAD OF COPYING
    - Invoice arithmetic (line-item total, subtotal + tax, discrepancy) is
      computed from the extracted fields, never read from the document's own
      claims.
    - `sla_breach_detected` is computed from the monthly average uptime and
      the contractual threshold. The fixture carries a pre-baked flag; using
      it would mean trusting the thing being audited.
    - Fields absent from a document are reported as null. The previous
      version invented defaults - `settlement_status="unpaid"` and
      `ocr_confidence=0.98` - so a document with no status looked safe to pay.
    - Fixture `notes` are authoring annotations that state the expected
      finding ("does not match ... $500 phantom surcharge"). They are not
      returned, so the agent has to reach the finding from the fields.

PATH CONFINEMENT (a security fix)
    The previous resolver accepted any absolute path that existed, so a
    model-chosen argument could read arbitrary JSON files on the host, and it
    fell back to `rglob(f"*{name}*")`, where a name such as "*" matched
    whichever file came first. Artifacts are now looked up only in an index
    of this domain's own data folders, keyed by file stem and by the id stored
    inside each file; directory components of the argument are discarded.
"""

import json
from pathlib import Path, PureWindowsPath
from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parent / "data"
INVOICE_DIR = DATA_DIR / "invoices"
CHART_DIR = DATA_DIR / "charts"

server = MCPServer("multimodal-investigation")

_CHART_METRIC_FIELDS = (
    "monthly_average_uptime_pct",
    "min_uptime_pct",
    "total_outage_minutes",
    "contractual_sla_threshold_pct",
    "overall_conversion_rate_pct",
    "biggest_dropoff_step",
    "biggest_dropoff_pct",
)


class DocumentExtractionResult(BaseModel):
    ok: bool
    document_id: str
    document_type: str | None = None
    vendor_name: str | None = None
    invoice_date: str | None = None
    currency: str | None = None
    line_items: list[dict[str, Any]] = Field(default_factory=list)
    subtotal: float | None = None
    tax_rate_pct: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None
    settlement_status: str | None = None
    is_duplicate: bool | None = None
    original_reference_id: str | None = None
    ocr_confidence: float | None = None
    # Computed here from the fields above (rounded to cents).
    line_items_total: float | None = None
    computed_total: float | None = None
    total_discrepancy: float | None = None
    summary: str
    error: str | None = None


class ChartAnalysisResult(BaseModel):
    ok: bool
    artifact_id: str
    artifact_name: str | None = None  # the file stem, as listed by list_visual_artifacts
    chart_type: str | None = None
    title: str | None = None
    service_name: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    data_points: list[dict[str, Any]] = Field(default_factory=list)
    penalty_clause: str | None = None
    sla_breach_detected: bool | None = None
    summary: str
    error: str | None = None


class ArtifactsListResult(BaseModel):
    invoices: list[str]
    charts: list[str]
    total_artifacts: int


def _artifact_index(folder: Path, id_field: str) -> dict[str, Path]:
    """Every accepted reference (file stem and internal id, casefolded) -> file."""
    index: dict[str, Path] = {}
    if not folder.is_dir():
        return index
    for path in sorted(folder.glob("*.json")):
        index[path.stem.casefold()] = path
        try:
            internal_id = json.loads(path.read_text(encoding="utf-8")).get(id_field)
        except (OSError, json.JSONDecodeError, AttributeError):
            continue  # still reachable by stem; extraction then reports the parse error
        if internal_id:
            index.setdefault(str(internal_id).strip().casefold(), path)
    return index


def _resolve_artifact(reference: str, folder: Path, id_field: str) -> Path | None:
    # PureWindowsPath splits on both "/" and "\", so "../../x" and "..\\x" both reduce to "x".
    key = PureWindowsPath(reference.strip()).name.casefold().removesuffix(".json")
    return _artifact_index(folder, id_field).get(key)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@server.tool(meta={"category": "vision"})
def extract_document_metrics(document_path: str) -> DocumentExtractionResult:
    """Extract the fields of one scanned invoice: vendor, line items, subtotal, tax, total,
    settlement status and duplicate flag, plus arithmetic computed from those fields
    (line_items_total, computed_total = subtotal + tax, total_discrepancy = total - computed_total).

    Args:
        document_path: The invoice's file name or document id, e.g. "inv-2026-001.json" or
            "INV-2026-001". Call list_visual_artifacts to see what exists.
    """
    path = _resolve_artifact(document_path, INVOICE_DIR, "document_id")
    if path is None:
        return DocumentExtractionResult(
            ok=False,
            document_id=document_path,
            summary=f"Document not found: {document_path}",
            error=f"No invoice matching '{document_path}' exists in domain storage.",
        )
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return DocumentExtractionResult(
            ok=False, document_id=document_path, summary=f"Extraction error: {exc}", error=str(exc)
        )

    line_items = [item for item in content.get("line_items", []) if isinstance(item, dict)]
    amounts = [_number(item.get("amount")) for item in line_items]
    subtotal = _number(content.get("subtotal"))
    tax = _number(content.get("tax_amount"))
    total = _number(content.get("total_amount"))

    line_items_total = round(sum(amounts), 2) if amounts and None not in amounts else None
    computed_total = round(subtotal + tax, 2) if subtotal is not None and tax is not None else None
    discrepancy = round(total - computed_total, 2) if total is not None and computed_total is not None else None

    document_id = str(content.get("document_id") or path.stem)
    status = content.get("settlement_status")
    # Strict identity check: a flag serialized as the string "false" must not count as true.
    is_duplicate = content.get("is_duplicate") is True

    def money(value: float | None) -> str:
        return "not present" if value is None else f"${value:,.2f}"

    summary = (
        f"{document_id} ({content.get('vendor_name', 'unknown vendor')}): subtotal {money(subtotal)}, "
        f"tax {money(tax)}, billed total {money(total)}; computed subtotal + tax = {money(computed_total)}, "
        f"discrepancy {money(discrepancy)}; status {status or 'not present'}; duplicate flag {is_duplicate}."
    )
    return DocumentExtractionResult(
        ok=True,
        document_id=document_id,
        document_type=content.get("document_type"),
        vendor_name=content.get("vendor_name"),
        invoice_date=content.get("invoice_date"),
        currency=content.get("currency"),
        line_items=line_items,
        subtotal=subtotal,
        tax_rate_pct=_number(content.get("tax_rate_pct")),
        tax_amount=tax,
        total_amount=total,
        settlement_status=status,
        is_duplicate=is_duplicate,
        original_reference_id=content.get("original_reference_id"),
        ocr_confidence=_number(content.get("ocr_confidence")),
        line_items_total=line_items_total,
        computed_total=computed_total,
        total_discrepancy=discrepancy,
        summary=summary,
    )


@server.tool(meta={"category": "vision"})
def analyze_chart_data(image_path: str) -> ChartAnalysisResult:
    """Read one telemetry or funnel chart: its data points (weekly uptime, or funnel stages),
    summary metrics, and - for SLA charts - whether the monthly average uptime falls below the
    contractual SLA threshold (sla_breach_detected, computed from those two values).

    Args:
        image_path: The chart's name or id, e.g. "latency_sla_september" or "CHART-SLA-0926".
            Call list_visual_artifacts to see what exists.
    """
    path = _resolve_artifact(image_path, CHART_DIR, "artifact_id")
    if path is None:
        return ChartAnalysisResult(
            ok=False,
            artifact_id=image_path,
            summary=f"Chart not found: {image_path}",
            error=f"No chart matching '{image_path}' exists in domain storage.",
        )
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ChartAnalysisResult(ok=False, artifact_id=image_path, summary=f"Chart error: {exc}", error=str(exc))

    metrics = {name: content[name] for name in _CHART_METRIC_FIELDS if name in content}
    data_points = content.get("data_points") or content.get("stages") or []

    average = _number(metrics.get("monthly_average_uptime_pct"))
    threshold = _number(metrics.get("contractual_sla_threshold_pct"))
    breach = (average < threshold) if average is not None and threshold is not None else None

    artifact_id = str(content.get("artifact_id") or path.stem)
    summary = f"{artifact_id} '{content.get('title', '')}': {len(data_points)} data points; metrics {json.dumps(metrics)}"
    if breach is not None:
        summary += (
            f"; monthly average uptime {average}% vs contractual threshold {threshold}% -> "
            f"{'SLA BREACH' if breach else 'within SLA'}"
        )
    return ChartAnalysisResult(
        ok=True,
        artifact_id=artifact_id,
        artifact_name=path.stem,
        chart_type=content.get("chart_type"),
        title=content.get("title"),
        service_name=content.get("service_name"),
        metrics=metrics,
        data_points=data_points,
        penalty_clause=content.get("contractual_penalty_clause"),
        sla_breach_detected=breach,
        summary=summary,
    )


@server.tool(meta={"category": "record"})
def list_visual_artifacts() -> ArtifactsListResult:
    """List the invoice documents and charts available for inspection."""
    invoices = sorted(path.stem for path in INVOICE_DIR.glob("*.json")) if INVOICE_DIR.is_dir() else []
    charts = sorted(path.stem for path in CHART_DIR.glob("*.json")) if CHART_DIR.is_dir() else []
    return ArtifactsListResult(invoices=invoices, charts=charts, total_artifacts=len(invoices) + len(charts))


if __name__ == "__main__":
    server.run()
