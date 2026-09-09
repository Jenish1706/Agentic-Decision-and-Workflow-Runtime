"""MCP Knowledge Server: exposes the RAG pipeline (retrieve + rerank) as an
MCP tool, plus a resource listing every document in the corpus.

Run standalone for manual inspection:
    uv run mcp dev src/agent_runtime/mcp_layer/servers/knowledge_server.py
or let mcp_layer/client.py launch it as a subprocess (the normal path).

This process shares no Python objects with whatever calls it - it builds its
own RagPipeline from disk at startup, exactly as a genuinely separate
service would. That's what makes this "protocol-accessed", not "a function
call with extra steps": the only thing caller and server share is the MCP
wire protocol over stdio.
"""

from pydantic import BaseModel

from mcp.server.mcpserver import MCPServer

from ...config import settings
from ...rag.documents import load_documents
from ...rag.ingest import build_rag_pipeline

server = MCPServer("nimbus-knowledge")
_pipeline = build_rag_pipeline()
_documents = load_documents(settings.documents_dir)


class SearchHit(BaseModel):
    doc_id: str
    doc_title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]


@server.tool()
def search_documents(query: str, top_k: int = 3) -> SearchResponse:
    """Semantically search the Nimbus engineering knowledge base (architecture docs,
    runbooks, API references, postmortems, policies, FAQs) for chunks relevant to `query`."""
    results = _pipeline.retrieve(query, top_k_rerank=top_k)
    return SearchResponse(
        hits=[
            SearchHit(doc_id=r.chunk.doc_id, doc_title=r.chunk.doc_title, text=r.chunk.text, score=r.score)
            for r in results
        ]
    )


@server.resource("documents://list")
def list_documents() -> str:
    """Lists every document id, title, and category in the knowledge base."""
    return "\n".join(f"{d.doc_id} | {d.title} | category={d.category}" for d in _documents)


if __name__ == "__main__":
    server.run()
