"""Worker roles for the multi-agent path. Only the Researcher runs a real
(small, bounded) agent loop - Retriever is a direct pipeline call and
Verifier is a single judge call, because neither needs autonomous
step-by-step reasoning to do its one job. Giving every worker its own full
ReAct loop would multiply agent loops (and LLM calls) for no added
capability.
"""

from dataclasses import dataclass

from ..agent.loop import AgentRuntime
from ..agent.tools.registry import ToolRegistry
from ..llm.base import LLMClient
from ..rag.context_builder import build_context
from ..rag.documents import Chunk
from ..rag.pipeline import RagPipeline


class RetrieverWorker:
    """Runs the RAG pipeline directly - no LLM call of its own, just
    retrieve -> rerank -> citation-tagged context.

    Note for the retrieval-budget experiment: this worker searches exactly
    ONCE per query, by construction. The multi-agent path therefore operates
    under an implicit retrieval budget of 1, which it never chose and cannot
    exceed - see EVALUATION.md's budget experiment for why that turns out to
    matter.
    """

    def __init__(self, rag: RagPipeline):
        self._rag = rag

    def run(self, query: str) -> tuple[str, dict[str, Chunk], list[str]]:
        """Returns (context_text, citation_map, retrieved_chunk_ids)."""
        results = self._rag.retrieve(query)
        chunk_ids = [r.chunk.chunk_id for r in results]
        if not results:
            return "No relevant documents found.", {}, chunk_ids
        context_text, citation_map = build_context([(r.chunk, r.score) for r in results])
        return context_text, citation_map, chunk_ids


@dataclass(frozen=True)
class ResearchOutcome:
    observations: list[str]
    tool_names_called: list[str]


class ResearcherWorker:
    """A small (max ~3 step), tool-only AgentRuntime instance (rag=None) -
    literally the same AgentRuntime class the single-agent path uses,
    reused here as a bounded specialist rather than reimplemented."""

    def __init__(self, llm: LLMClient, tools: ToolRegistry, max_steps: int = 3):
        self._runtime = AgentRuntime(llm=llm, tools=tools, rag=None, max_steps=max_steps, max_decision_retries=1)

    async def run(self, query: str) -> ResearchOutcome:
        # Each delegation is an independent sub-task. The worker keeps one
        # AgentRuntime for its whole life, and a runtime remembers recent
        # turns, so without this reset every question the Supervisor handled
        # leaked into the Researcher's prompt for the next one - including
        # across benchmark questions in the multi-agent evaluation.
        self._runtime.reset_session()
        state = await self._runtime.run(query)
        calls = state.tool_calls()
        return ResearchOutcome(
            observations=[f"{s.decision.tool_name}({s.decision.tool_arguments}) -> {s.observation}" for s in calls],
            tool_names_called=[s.decision.tool_name for s in calls if s.decision.tool_name],
        )
