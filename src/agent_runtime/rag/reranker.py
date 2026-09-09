"""Cross-encoder reranking.

Why rerank at all: the embedder scores a query against each chunk
independently (a "bi-encoder") - fast enough to search hundreds of vectors,
but it never lets the query and the chunk attend to each other, so it
regularly ranks a topically-similar-but-wrong chunk above the actually
relevant one. A cross-encoder scores the (query, chunk) pair jointly, which
is far more accurate but too slow to run over an entire corpus. The
standard pattern - used here - is a fast bi-encoder search for a shortlist
(top_k_retrieve), then a slower, more accurate cross-encoder rerank of just
that shortlist down to top_k_rerank.
"""

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str):
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_n: int) -> list[tuple[str, float]]:
        """candidates: (chunk_id, chunk_text) pairs. Returns the top_n
        (chunk_id, score) pairs, sorted by descending score."""
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip((c[0] for c in candidates), scores), key=lambda pair: -pair[1])
        return [(chunk_id, float(score)) for chunk_id, score in ranked[:top_n]]
