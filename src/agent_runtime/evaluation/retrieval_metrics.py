"""Search-efficiency metrics: what did retrieval actually COST, separately
from whether the answer was right.

This is deliberately distinct from rag_metrics.py, which measures retrieval
*quality* (did the right documents come back). A run can score perfect
Recall@k and still be badly inefficient - searching four times for the same
thing, or accumulating sixteen chunks of context to answer a one-fact
question. This project's own evaluation showed exactly that: the single
agent reached correct answers while making 1.60 unnecessary tool calls per
query on average, and on three questions exhausted its step budget
mid-search and returned nothing at all.

Naming note: these measure efficiency against budgets that are mostly
*implicit* (top-k per call, max_steps per run). The only explicit retrieval
budget in the codebase is `AgentRuntime(max_retrievals=...)`, added for the
budget experiment - see scripts/run_budget_experiment.py.
"""

from dataclasses import dataclass


def _normalize(query: str) -> str:
    """Two searches differing only in case, spacing, or trailing punctuation
    are the same search for redundancy purposes. Deliberately shallow - this
    is not semantic dedup, and near-paraphrases ('Cascade escalation' vs
    'Cascade escalation process') still count as distinct. That undercounts
    real redundancy, which is the safer direction to be wrong in."""
    return " ".join(query.lower().strip().strip("?.!").split())


@dataclass(frozen=True)
class SearchEfficiency:
    retrieval_calls: int
    unique_retrieval_queries: int
    retrieval_chunks_consumed: int
    unique_chunks_consumed: int
    retrieval_redundancy: float | None
    """Fraction of retrieved chunks that were already retrieved earlier in
    the same run: 1 - unique/total. None when nothing was retrieved (not 0.0
    - a run that never searched has no redundancy to speak of, and averaging
    it in as 'perfectly efficient' would flatter the result)."""

    repeated_query_calls: int
    """Retrieval calls issued for a query string already searched this run."""


def score_search_efficiency(retrieval_queries: list[str], retrieved_chunk_ids: list[str]) -> SearchEfficiency:
    normalized = [_normalize(q) for q in retrieval_queries]
    unique_queries = len(set(normalized))
    total_chunks = len(retrieved_chunk_ids)
    unique_chunks = len(set(retrieved_chunk_ids))

    redundancy = None if total_chunks == 0 else 1.0 - (unique_chunks / total_chunks)

    return SearchEfficiency(
        retrieval_calls=len(retrieval_queries),
        unique_retrieval_queries=unique_queries,
        retrieval_chunks_consumed=total_chunks,
        unique_chunks_consumed=unique_chunks,
        retrieval_redundancy=redundancy,
        repeated_query_calls=len(normalized) - unique_queries,
    )
