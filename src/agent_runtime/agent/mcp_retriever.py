"""Routes the agent loop's 'retrieve' action through the knowledge MCP
server, instead of calling RagPipeline in-process.

This exists because of a real bug found by actually running the agent: the
knowledge server's `search_documents` tool and the dedicated 'retrieve'
action both did document search, but only 'retrieve' fed results into
`citation_map` - so an agent that reached for `search_documents` as a
regular tool call could make claims that verify_citations() never saw and
therefore never checked. Making 'retrieve' itself MCP-mediated - rather
than also exposing search_documents as a separately-callable tool - removes
the duplicate path entirely: there is exactly one way to search documents,
and it is both protocol-accessed and verification-covered. It also means
MCP genuinely carries the agent's core retrieval, not just side tools -
see MCP.md.

(Multi-agent's RetrieverWorker still calls RagPipeline directly in-process
- see multi_agent/workers.py - since by that point in the architecture the
MCP-vs-local contrast has already been made concretely, and that worker has
no independent tool-call path that could bypass anything.)
"""

from ..mcp_layer.client import MCPToolClient
from ..rag.documents import Chunk
from ..rag.pipeline import RetrievedChunk


class MCPRetriever:
    def __init__(self, client: MCPToolClient, server_name: str = "knowledge", tool_name: str = "search_documents"):
        self._client = client
        self._server_name = server_name
        self._tool_name = tool_name

    async def retrieve(self, query: str) -> list[RetrievedChunk]:
        result = await self._client.call_tool(self._server_name, self._tool_name, {"query": query})
        hits = (result.structured_content or {}).get("hits", [])
        retrieved = []
        for position, hit in enumerate(hits):
            chunk = Chunk(
                chunk_id=f"{hit['doc_id']}::mcp{position}",
                doc_id=hit["doc_id"],
                doc_title=hit["doc_title"],
                position=position,
                text=hit["text"],
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=hit["score"]))
        return retrieved
