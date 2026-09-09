"""Covers agent/mcp_retriever.py directly - the fix for the
verification-bypass bug described in ARCHITECTURE.md. Uses the SDK's
in-process Client mode against the real knowledge server (so this does load
a real embedding model - a few seconds - unlike the rest of the fast suite,
but it's the one test that would have caught the original bug: 'retrieve'
must actually go through MCP and come back as usable RetrievedChunks."""

import pytest
from mcp.client import Client

from agent_runtime.agent.mcp_retriever import MCPRetriever
from agent_runtime.mcp_layer.servers import knowledge_server


class _SingleServerClient:
    """Adapts a single in-process mcp.client.Client to the
    {server_name: call_tool(...)} shape MCPRetriever expects from
    MCPToolClient, without spawning a subprocess."""

    def __init__(self, client: Client):
        self._client = client

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        return await self._client.call_tool(tool_name, arguments)


@pytest.mark.asyncio
async def test_mcp_retriever_returns_chunks_from_the_real_knowledge_server():
    async with Client(knowledge_server.server) as client:
        retriever = MCPRetriever(_SingleServerClient(client))
        results = await retriever.retrieve("Aurora rate limit")

    assert len(results) > 0
    top = results[0]
    assert top.chunk.doc_id  # populated, not empty
    assert top.chunk.text
    assert isinstance(top.score, float)


@pytest.mark.asyncio
async def test_mcp_retriever_chunk_ids_are_unique_per_hit():
    async with Client(knowledge_server.server) as client:
        retriever = MCPRetriever(_SingleServerClient(client))
        results = await retriever.retrieve("on-call escalation timing")

    chunk_ids = [r.chunk.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))
