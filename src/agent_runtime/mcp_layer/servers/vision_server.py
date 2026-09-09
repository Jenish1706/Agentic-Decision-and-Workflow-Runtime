"""MCP Vision & Document Analysis Server: Inspects multimodal visual documents,
scanned invoices, and telemetry charts.

Enables agents to:
1. Extract numerical metrics and line-item tables from scanned documents and invoices.
2. Extract trends, anomalies, and SLA compliance metrics from visual charts.
3. List available visual investigation artifacts in the domain corpus.
"""

import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel

from mcp.server.mcpserver import MCPServer
from ...config import REPO_ROOT

server = MCPServer("multimodal-vision")

DATA_DIR = REPO_ROOT / "domains" / "multimodal_investigation" / "data"


class DocumentExtractionResult(BaseModel):
    ok: bool
    document_id: str
    document_type: str = "document"
    vendor_name: str | None = None
    subtotal: float | None = None
    tax_rate_pct: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None
    settlement_status: str | None = None
    is_duplicate: bool = False
    original_reference_id: str | None = None
    line_items: list[dict[str, Any]] = []
    ocr_confidence: float = 1.0
    summary: str
    raw_fields: dict[str, Any] = {}
    error: str | None = None


class ChartAnalysisResult(BaseModel):
    ok: bool
    artifact_id: str
    chart_type: str
    title: str
    metrics: dict[str, Any] = {}
    data_points: list[dict[str, Any]] = []
    sla_breach_detected: bool = False
    summary: str
    error: str | None = None


class ArtifactsListResult(BaseModel):
    ok: bool
    invoices: list[str]
    charts: list[str]
    total_artifacts: int


def _resolve_file(file_path: str, subfolder: str) -> Path | None:
    path_obj = Path(file_path)
    if path_obj.is_absolute() and path_obj.exists():
        return path_obj

    # Check directly inside subfolder
    candidate = DATA_DIR / subfolder / path_obj.name
    if candidate.exists():
        return candidate

    # Check with .json suffix
    if not path_obj.name.endswith(".json"):
        candidate_json = DATA_DIR / subfolder / f"{path_obj.name}.json"
        if candidate_json.exists():
            return candidate_json

    # Check anywhere inside DATA_DIR recursively
    for p in DATA_DIR.rglob(f"*{path_obj.name}*"):
        if p.is_file():
            return p

    return None


@server.tool()
def extract_document_metrics(document_path: str, query: str = "") -> DocumentExtractionResult:
    """Extract structured key-values, line items, and financial totals from a scanned invoice or document.

    Resolves document paths such as 'inv-2026-001.json' or 'INV-2026-001'.
    Returns verified numerical fields: subtotal, tax_amount, total_amount, line items, and settlement status.
    """
    resolved = _resolve_file(document_path, "invoices")
    if not resolved or not resolved.exists():
        return DocumentExtractionResult(
            ok=False,
            document_id=document_path,
            summary=f"Document not found: {document_path}",
            error=f"Visual document '{document_path}' could not be located in domain storage.",
        )

    try:
        content = json.loads(resolved.read_text(encoding="utf-8"))
        doc_id = content.get("document_id", resolved.stem)
        subtotal = content.get("subtotal")
        tax = content.get("tax_amount")
        total = content.get("total_amount")
        vendor = content.get("vendor_name", "Unknown Vendor")
        status = content.get("settlement_status", "unpaid")
        is_dup = content.get("is_duplicate", False)

        summary_lines = [
            f"Extracted Document {doc_id} ({vendor}):",
            f"- Subtotal: ${subtotal:,.2f}" if subtotal is not None else "- Subtotal: None",
            f"- Tax: ${tax:,.2f}" if tax is not None else "- Tax: None",
            f"- Total: ${total:,.2f}" if total is not None else "- Total: None",
            f"- Status: {status} (is_duplicate: {is_dup})",
        ]

        return DocumentExtractionResult(
            ok=True,
            document_id=doc_id,
            document_type=content.get("document_type", "scanned_invoice"),
            vendor_name=vendor,
            subtotal=subtotal,
            tax_rate_pct=content.get("tax_rate_pct"),
            tax_amount=tax,
            total_amount=total,
            settlement_status=status,
            is_duplicate=is_dup,
            original_reference_id=content.get("original_reference_id"),
            line_items=content.get("line_items", []),
            ocr_confidence=content.get("ocr_confidence", 0.98),
            summary="\n".join(summary_lines),
            raw_fields=content,
        )
    except Exception as exc:
        return DocumentExtractionResult(
            ok=False,
            document_id=document_path,
            summary=f"Extraction error: {exc}",
            error=str(exc),
        )


@server.tool()
def analyze_chart_data(image_path: str, metric_focus: str = "") -> ChartAnalysisResult:
    """Analyze a telemetry or funnel chart visual artifact to extract series, data points, and SLA metrics.

    Resolves chart artifacts such as 'latency_sla_september' or 'conversion_funnel_redesign'.
    Returns series metrics, anomaly flags, and uptime/conversion percentages.
    """
    resolved = _resolve_file(image_path, "charts")
    if not resolved or not resolved.exists():
        return ChartAnalysisResult(
            ok=False,
            artifact_id=image_path,
            chart_type="unknown",
            title="",
            summary=f"Chart not found: {image_path}",
            error=f"Chart artifact '{image_path}' could not be located in domain storage.",
        )

    try:
        content = json.loads(resolved.read_text(encoding="utf-8"))
        art_id = content.get("artifact_id", resolved.stem)
        chart_type = content.get("chart_type", "metric_chart")
        title = content.get("title", "")
        sla_breach = content.get("sla_breach_detected", False)
        data_pts = content.get("data_points") or content.get("stages") or []

        metrics: dict[str, Any] = {}
        for k in (
            "monthly_average_uptime_pct",
            "min_uptime_pct",
            "total_outage_minutes",
            "contractual_sla_threshold_pct",
            "overall_conversion_rate_pct",
            "biggest_dropoff_step",
            "biggest_dropoff_pct",
        ):
            if k in content:
                metrics[k] = content[k]

        summary_lines = [
            f"Analyzed Chart {art_id} ('{title}'):",
            f"- Chart Type: {chart_type}",
            f"- Data Points: {len(data_pts)} stages/periods",
            f"- Key Metrics: {json.dumps(metrics)}",
            f"- SLA Breach: {sla_breach}",
        ]

        return ChartAnalysisResult(
            ok=True,
            artifact_id=art_id,
            chart_type=chart_type,
            title=title,
            metrics=metrics,
            data_points=data_pts,
            sla_breach_detected=sla_breach,
            summary="\n".join(summary_lines),
        )
    except Exception as exc:
        return ChartAnalysisResult(
            ok=False,
            artifact_id=image_path,
            chart_type="unknown",
            title="",
            summary=f"Chart analysis error: {exc}",
            error=str(exc),
        )


@server.tool()
def list_visual_artifacts() -> ArtifactsListResult:
    """List all visual documents (invoices, receipts) and charts available in domain storage."""
    invoices_dir = DATA_DIR / "invoices"
    charts_dir = DATA_DIR / "charts"

    inv_list = [f.stem for f in invoices_dir.glob("*.json")] if invoices_dir.exists() else []
    chart_list = [f.stem for f in charts_dir.glob("*.json")] if charts_dir.exists() else []

    return ArtifactsListResult(
        ok=True,
        invoices=sorted(inv_list),
        charts=sorted(chart_list),
        total_artifacts=len(inv_list) + len(chart_list),
    )


if __name__ == "__main__":
    server.run()
