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

from agent_runtime.mcp_layer.servers import records_server


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
