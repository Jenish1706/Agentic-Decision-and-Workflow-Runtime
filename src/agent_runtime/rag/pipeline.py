"""Ties embeddings + vector index + reranker into one retrieve() call:
embed the query -> brute-force top-k by cosine similarity -> cross-encoder
rerank down to a small top-n. This is the one class the agent loop and the
MCP knowledge server both call - neither knows or cares how retrieval is
implemented underneath."""

from dataclasses import dataclass

from .documents import Chunk
from .embeddings import Embedder
from .reranker import Reranker
from .vector_index import VectorIndex


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        reranker: Reranker,
        chunks_by_id: dict[str, Chunk],
        top_k_retrieve: int,
        top_k_rerank: int,
    ):
        self._embedder = embedder
        self._index = index
        self._reranker = reranker
        self._chunks_by_id = chunks_by_id
        self._top_k_retrieve = top_k_retrieve
        self._top_k_rerank = top_k_rerank

    def retrieve(self, query: str, top_k_rerank: int | None = None) -> list[RetrievedChunk]:
        query_vector = self._embedder.embed([query])[0]
        hits = self._index.search(query_vector, self._top_k_retrieve)
        if not hits:
            return []
        candidates = [(hit.chunk_id, self._chunks_by_id[hit.chunk_id].text) for hit in hits]
        reranked = self._reranker.rerank(query, candidates, top_k_rerank or self._top_k_rerank)
        return [RetrievedChunk(chunk=self._chunks_by_id[chunk_id], score=score) for chunk_id, score in reranked]

    def retrieve_naive(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Bi-encoder similarity only, no reranking. Exists so the eval
        harness can report naive-vs-reranked retrieval quality side by side
        (Constraint 5's requested comparison), not for runtime use."""
        query_vector = self._embedder.embed([query])[0]
        hits = self._index.search(query_vector, top_k)
        return [RetrievedChunk(chunk=self._chunks_by_id[hit.chunk_id], score=hit.score) for hit in hits]
