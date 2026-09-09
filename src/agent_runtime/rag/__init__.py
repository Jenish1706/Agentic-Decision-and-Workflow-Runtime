"""RAG sub-package. Deliberately no re-exports here beyond what's free to
import: Embedder/Reranker/RagPipeline pull in sentence-transformers (and
therefore torch) as soon as they're imported, which is a real handful of
seconds - fine to pay once for an actual run, but worth not forcing on
every test that only needs chunking or vector_index. Import those directly
from their submodules."""
