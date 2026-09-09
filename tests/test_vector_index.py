import numpy as np

from agent_runtime.rag.vector_index import VectorIndex


def test_empty_index_returns_no_results():
    index = VectorIndex(dim=3)
    results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=5)
    assert results == []


def test_ranks_by_cosine_similarity_descending():
    index = VectorIndex(dim=2)
    # unit vectors at 0deg, 90deg, 45deg from the query direction
    index.add(
        ["orthogonal", "diagonal", "exact_match"],
        np.array([[0.0, 1.0], [0.7071, 0.7071], [1.0, 0.0]], dtype=np.float32),
    )
    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)
    assert [r.chunk_id for r in results] == ["exact_match", "diagonal", "orthogonal"]
    assert results[0].score > results[1].score > results[2].score
    assert results[0].score == 1.0


def test_top_k_smaller_than_index_size_returns_only_top_k():
    index = VectorIndex(dim=2)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    index.add(["a", "b", "c"], vectors)
    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "a"


def test_top_k_larger_than_index_size_is_clamped():
    index = VectorIndex(dim=2)
    index.add(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=50)
    assert len(results) == 2


def test_add_rejects_mismatched_lengths():
    index = VectorIndex(dim=2)
    try:
        index.add(["a", "b"], np.array([[1.0, 0.0]], dtype=np.float32))
        raised = False
    except ValueError:
        raised = True
    assert raised
