"""RAG-side metrics.

Recall@k and MRR are computed directly against the RAG pipeline (not
per-agent-mode) - retrieval quality is a property of the retriever, and
scoring it the same way regardless of which agent later consumes it is
what lets these numbers mean the same thing across the single-vs-multi-agent
comparison.

Groundedness and citation correctness ARE computed per final answer (they
depend on what the agent/synthesizer actually wrote), so those take an
already-produced answer + evidence as input.
"""

from dataclasses import dataclass

from ..agent.llm_judge import llm_judge_faithfulness
from ..llm.base import LLMClient
from ..rag.pipeline import RagPipeline
from .dataset import EvalQuery


@dataclass(frozen=True)
class RetrievalScore:
    recall_at_k: float | None
    reciprocal_rank: float | None


def score_retrieval(rag: RagPipeline, query: EvalQuery, top_k: int = 3) -> RetrievalScore:
    """None (not 0.0) means 'not applicable' - this query has no expected
    documents (e.g. it's answerable purely from structured records, or is
    an out-of-corpus trap), so retrieval quality isn't a meaningful score
    for it and it should be excluded from the aggregate, not penalized."""
    if not query.expected_doc_ids:
        return RetrievalScore(recall_at_k=None, reciprocal_rank=None)

    results = rag.retrieve(query.question, top_k_rerank=top_k)
    retrieved_doc_ids = [r.chunk.doc_id for r in results]
    expected = set(query.expected_doc_ids)

    hit_count = len({d for d in retrieved_doc_ids if d in expected})
    recall = hit_count / len(expected)

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            reciprocal_rank = 1.0 / rank
            break

    return RetrievalScore(recall_at_k=recall, reciprocal_rank=reciprocal_rank)


def score_retrieval_naive(rag: RagPipeline, query: EvalQuery, top_k: int = 3) -> RetrievalScore:
    """Same as score_retrieval but skips the reranking step - this is the
    'naive retrieval vs reranked retrieval' comparison Constraint 5 asks
    for, computed by calling the pipeline's bi-encoder-only path."""
    if not query.expected_doc_ids:
        return RetrievalScore(recall_at_k=None, reciprocal_rank=None)

    results = rag.retrieve_naive(query.question, top_k=top_k)
    retrieved_doc_ids = [r.chunk.doc_id for r in results]
    expected = set(query.expected_doc_ids)

    hit_count = len({d for d in retrieved_doc_ids if d in expected})
    recall = hit_count / len(expected)

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected:
            reciprocal_rank = 1.0 / rank
            break

    return RetrievalScore(recall_at_k=recall, reciprocal_rank=reciprocal_rank)


async def score_groundedness(llm: LLMClient, query: EvalQuery, answer: str, evidence_text: str) -> tuple[bool, list[str]]:
    verdict = await llm_judge_faithfulness(llm, query.question, answer, evidence_text)
    return verdict.grounded, verdict.issues


def score_citation_correctness(
    citations: list[str], citation_sources: dict[str, str], expected_doc_ids: list[str]
) -> float | None:
    """Fraction of cited tags that point to a chunk from an expected
    document - i.e. not just "a valid-looking citation" but "cited the
    right source". None when there's nothing to check against."""
    if not expected_doc_ids or not citations:
        return None
    expected = set(expected_doc_ids)
    correct = sum(1 for tag in citations if citation_sources.get(tag) in expected)
    return correct / len(citations)
