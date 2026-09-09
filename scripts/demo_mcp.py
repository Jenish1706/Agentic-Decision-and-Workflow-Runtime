"""Demonstrates the raw MCP protocol path with no agent involved: connect to
both servers as subprocesses, discover their tools (schemas included), then
invoke one tool on each. This is what "protocol-accessed tool" means
concretely, as distinct from the local calculator tool's direct function
call - see agent/tools/calculator.py vs agent/tools/mcp_tool.py.
"""

import asyncio
import json

from agent_runtime.mcp_layer.client import MCPToolClient


async def main() -> None:
    async with MCPToolClient() as client:
        print("=== Tool discovery ===")
        discovered = await client.discover_tools()
        for tool in discovered:
            print(f"[{tool.server_name}] {tool.name}: {tool.description}")
            print(f"    input_schema: {tool.input_schema}")

        print("\n=== Invoking knowledge.search_documents ===")
        result = await client.call_tool("knowledge", "search_documents", {"query": "Aurora rate limit", "top_k": 2})
        print(json.dumps(result.structured_content, indent=2))

        print("\n=== Invoking records.lookup_record ===")
        result = await client.call_tool("records", "lookup_record", {"record_id": "ferry"})
        print(json.dumps(result.structured_content, indent=2))

        print("\n=== Invoking records.search_records ===")
        result = await client.call_tool("records", "search_records", {"field": "owner_team", "value": "nimbus-platform"})
        print(json.dumps(result.structured_content, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
