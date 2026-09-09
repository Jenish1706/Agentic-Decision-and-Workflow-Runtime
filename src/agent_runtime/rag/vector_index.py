"""Brute-force cosine-similarity vector search, built from scratch.

Why brute-force instead of FAISS/HNSW: this corpus is dozens of documents
and a few hundred chunks. Similarity search over that is one (n, dim) x
(dim,) matrix-vector product - microseconds on a CPU, no index structure
needed. An approximate nearest-neighbor index earns its complexity past
roughly 10^5-10^6 vectors, where brute force actually becomes too slow; below
that it's added complexity with no measurable benefit. Writing the real
similarity math here (rather than importing it) is also just more useful
for being able to explain "how does vector search actually work" in an
interview.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    score: float


class VectorIndex:
    def __init__(self, dim: int):
        self._dim = dim
        self._ids: list[str] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors must have the same length")
        if vectors.shape[0] == 0:
            return
        self._ids.extend(ids)
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievalResult]:
        if not self._ids or top_k <= 0:
            return []
        scores = self._vectors @ query_vector  # unit-normalized vectors -> dot product == cosine similarity
        k = min(top_k, len(self._ids))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [RetrievalResult(chunk_id=self._ids[i], score=float(scores[i])) for i in top_idx]
