"""Embedding interface. sentence-transformers is used as infrastructure (a
pretrained encoder) - training or implementing an embedding model is out of
scope; what this project builds itself is everything downstream of the
vector (the index, search, and reranking)."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns an (n, dim) float32 array of L2-normalized embeddings.

        Normalizing at embed time means cosine similarity reduces to a plain
        dot product later in VectorIndex - one less thing to get wrong.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)
