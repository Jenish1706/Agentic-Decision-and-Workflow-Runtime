"""Builds a ready-to-query RagPipeline from the documents directory: load ->
chunk -> embed -> index. This is the one function that wires the RAG
sub-modules together into something runnable; everything it calls is unit
tested independently."""

from pathlib import Path

from ..config import settings as default_settings
from .chunking import chunk_text
from .documents import Chunk, load_documents
from .embeddings import Embedder
from .pipeline import RagPipeline
from .reranker import Reranker
from .vector_index import VectorIndex


def build_rag_pipeline(
    documents_dir: Path | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> RagPipeline:
    documents_dir = documents_dir or default_settings.documents_dir
    embedder = embedder or Embedder(default_settings.embedding_model)
    reranker = reranker or Reranker(default_settings.reranker_model)

    documents = load_documents(documents_dir)
    chunks_by_id: dict[str, Chunk] = {}
    for doc in documents:
        pieces = chunk_text(doc.text, default_settings.chunk_size, default_settings.chunk_overlap)
        for position, piece in enumerate(pieces):
            chunk_id = f"{doc.doc_id}::{position}"
            chunks_by_id[chunk_id] = Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                doc_title=doc.title,
                position=position,
                text=piece,
            )

    index = VectorIndex(dim=embedder.dim)
    if chunks_by_id:
        ids = list(chunks_by_id.keys())
        vectors = embedder.embed([chunks_by_id[cid].text for cid in ids])
        index.add(ids, vectors)

    return RagPipeline(
        embedder=embedder,
        index=index,
        reranker=reranker,
        chunks_by_id=chunks_by_id,
        top_k_retrieve=default_settings.top_k_retrieve,
        top_k_rerank=default_settings.top_k_rerank,
    )
