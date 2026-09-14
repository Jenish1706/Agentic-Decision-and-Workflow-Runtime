"""MCP protocol tests using the SDK's in-process connection mode (passing
the MCPServer instance directly to Client, instead of spawning a
subprocess) - this exercises the real discovery/invocation/schema code
path without subprocess overhead. The subprocess path (mcp_layer/client.py
launching each server with StdioServerParameters) is exercised by
scripts/demo_mcp.py, which is the "genuinely a separate process" version of
the same protocol.
"""

import pytest
from mcp.client import Client

import domains.credit_risk.server as credit_risk_server
import domains.enterprise_operations.server as operations_server
import domains.multimodal_investigation.server as multimodal_server
import domains.product_analytics.server as analytics_server
from agent_runtime.agent.tools.mcp_tool import _category_from_meta
from agent_runtime.mcp_layer.servers import records_server


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        (analytics_server, {"execute_sql": "sql", "describe_tables": "record", "test_significance": "statistics"}),
        (credit_risk_server, {"execute_sql": "sql", "describe_tables": "record", "get_account_profile": "record"}),
        (
            multimodal_server,
            {"extract_document_metrics": "vision", "analyze_chart_data": "vision", "list_visual_artifacts": "record"},
        ),
        (
            operations_server,
            {
                "get_incident_telemetry": "record",
                "list_service_incidents": "record",
                "inspect_service_config": "record",
                "simulate_remediation_action": "statistics",
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_domain_tool_categories_survive_discovery(module, expected):
    # Budgets and evidence provenance depend on these declarations reaching
    # the client through the protocol; tests/fakes.py FunctionTool relies on
    # them matching too.
    async with Client(module.server) as client:
        listing = await client.list_tools()
        discovered = {tool.name: _category_from_meta(tool.meta).value for tool in listing.tools}
    assert discovered == expected


@pytest.mark.asyncio
async def test_discovers_records_server_tools_with_schemas():
    async with Client(records_server.server) as client:
        listing = await client.list_tools()
        names = {tool.name for tool in listing.tools}
        assert {"lookup_record", "search_records"} <= names
        lookup = next(t for t in listing.tools if t.name == "lookup_record")
        assert "record_id" in lookup.input_schema["properties"]


@pytest.mark.asyncio
async def test_lookup_record_hit_and_miss():
    async with Client(records_server.server) as client:
        hit = await client.call_tool("lookup_record", {"record_id": "ferry"})
        assert hit.structured_content["found"] is True
        assert hit.structured_content["record"]["owner_team"] == "nimbus-data"

        miss = await client.call_tool("lookup_record", {"record_id": "does-not-exist"})
        assert miss.structured_content["found"] is False


@pytest.mark.asyncio
async def test_search_records_by_field():
    async with Client(records_server.server) as client:
        result = await client.call_tool("search_records", {"field": "owner_team", "value": "nimbus-platform"})
        matches = result.structured_content["matches"]
        ids = {m["id"] for m in matches}
        assert ids == {"anchor", "beacon", "lighthouse"}


@pytest.mark.asyncio
async def test_malformed_call_missing_required_argument_is_reported_not_fatal():
    async with Client(records_server.server) as client:
        result = await client.call_tool("lookup_record", {})
        assert result.is_error
