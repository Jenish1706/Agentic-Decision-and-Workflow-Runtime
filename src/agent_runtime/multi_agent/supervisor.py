"""Supervisor: the orchestrator in Anthropic's orchestrator-workers pattern
(https://www.anthropic.com/engineering/building-effective-agents) - it looks
at the query, decides which specialist workers are actually needed (not "run
everything every time"), delegates, then runs one evaluator-optimizer cycle
(Synthesizer draft -> Verifier check -> at most one repair) before returning.

This is deliberately a different control structure from AgentRuntime's
step-by-step loop: the supervisor makes ONE delegation decision up front
rather than deciding action-by-action, because at this level the "actions"
are whole workers, and there are only ever two optional ones to choose
between.
"""

from pydantic import BaseModel, Field

from ..agent.llm_judge import JudgeVerdict, llm_judge_faithfulness
from ..agent.verification import extract_citation_tags
from ..llm.base import LLMClient, LLMError, Message
from .synthesizer import Synthesizer
from .workers import ResearcherWorker, RetrieverWorker


class DelegationPlan(BaseModel):
    thought: str
    needs_retrieval: bool = Field(description="True if answering requires searching the document knowledge base.")
    needs_tools: bool = Field(
        description="True if answering requires arithmetic or a structured service/team record lookup."
    )


class WorkerLogEntry(BaseModel):
    worker: str
    summary: str


class MultiAgentResult(BaseModel):
    query: str
    plan: DelegationPlan
    worker_log: list[WorkerLogEntry]
    final_answer: str
    citations: list[str]
    citation_sources: dict[str, str] = Field(default_factory=dict)
    evidence_text: str = ""
    tools_called: list[str] = Field(default_factory=list)
    # Same search-efficiency instrumentation as AgentState, so the two
    # architectures can be compared on identical measures.
    retrieval_queries: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    grounded: bool
    verification_issues: list[str]
    repair_attempts: int


# A thin version of this prompt ("You are a supervisor deciding which
# specialists to delegate to") failed badly in practice: the model read
# "specialists" as human job titles and answered things like "this falls under
# the expertise of a Database Administrator", selecting no workers at all for
# every question. Naming the two concrete workers, saying what they can
# actually reach, and stating the default corrected it. Tool/worker
# descriptions deserve the same care as any other prompt - see
# https://www.anthropic.com/engineering/building-effective-agents on tool design.
_PLAN_SYSTEM_PROMPT = """You route a question to automated workers in a \
retrieval system. You never answer the question yourself, and you must not rely \
on your own knowledge of the subject.

Two workers are available:

1. RETRIEVAL worker (needs_retrieval) - searches an internal engineering \
knowledge base for the fictional "Nimbus" platform: architecture overviews, \
runbooks, API references, incident postmortems, internal policies, and FAQs \
covering its services (Aurora, Compass, Anchor, Beacon, Lighthouse, Cascade, \
Ferry). Set this true for ANY question about how something works, what a limit \
or policy is, what happened in an incident, or what a procedure requires.

2. TOOLS worker (needs_tools) - can do arithmetic, and can look up structured \
records about services and teams (owner team, tier, uptime SLO, region, last \
deploy date). Set this true when the question needs a calculation, or a \
specific service/team attribute.

Both can be true; for a question spanning documentation and a record lookup, \
both SHOULD be true.

Default to needs_retrieval=true. Selecting neither worker means answering with \
no evidence at all, which is almost always wrong here - choose that only for \
pure small talk with no factual content."""


class Supervisor:
    def __init__(
        self,
        llm: LLMClient,
        retriever: RetrieverWorker,
        researcher: ResearcherWorker,
        synthesizer: Synthesizer,
        max_repair_attempts: int = 1,
    ):
        self._llm = llm
        self._retriever = retriever
        self._researcher = researcher
        self._synthesizer = synthesizer
        self._max_repair_attempts = max_repair_attempts

    async def run(self, query: str) -> MultiAgentResult:
        worker_log: list[WorkerLogEntry] = []
        try:
            plan = await self._llm.decide(
                [Message(role="system", content=_PLAN_SYSTEM_PROMPT), Message(role="user", content=query)],
                DelegationPlan,
            )
        except LLMError as exc:
            # A planner that can't produce a valid plan shouldn't sink the
            # query. Degrade toward gathering everything available: worse on
            # cost, better on recall, and honest about what happened.
            plan = DelegationPlan(
                thought=f"Planner failed ({exc}); defaulting to consulting every specialist.",
                needs_retrieval=True,
                needs_tools=True,
            )
            worker_log.append(WorkerLogEntry(worker="supervisor", summary="planning failed, using fallback plan"))

        evidence_text, citation_map = "", {}
        retrieval_queries: list[str] = []
        retrieved_chunk_ids: list[str] = []
        if plan.needs_retrieval:
            evidence_text, citation_map, retrieved_chunk_ids = self._retriever.run(query)
            retrieval_queries = [query]  # the retriever worker searches exactly once
            worker_log.append(WorkerLogEntry(worker="retriever", summary=f"retrieved {len(citation_map)} chunk(s)"))

        tool_observations: list[str] = []
        tools_called: list[str] = []
        if plan.needs_tools:
            outcome = await self._researcher.run(query)
            tool_observations = outcome.observations
            tools_called = outcome.tool_names_called
            worker_log.append(
                WorkerLogEntry(worker="researcher", summary=f"made {len(tool_observations)} tool call(s)")
            )

        combined_evidence = evidence_text
        if tool_observations:
            combined_evidence += "\n\nTool observations:\n" + "\n".join(tool_observations)

        draft = await self._synthesizer.synthesize(query, evidence_text, tool_observations)
        verdict, verifier_ok = await self._verify(query, draft, combined_evidence)
        worker_log.append(
            WorkerLogEntry(
                worker="verifier",
                summary=f"grounded={verdict.grounded}" if verifier_ok else "verifier unavailable",
            )
        )

        attempts = 0
        # `verifier_ok` gates the repair loop deliberately: repairing against a
        # verdict the verifier never actually produced would burn LLM calls
        # chasing feedback that doesn't mean anything.
        while verifier_ok and not verdict.grounded and attempts < self._max_repair_attempts:
            draft = await self._synthesizer.synthesize(
                query, evidence_text, tool_observations, feedback="; ".join(verdict.issues)
            )
            verdict, verifier_ok = await self._verify(query, draft, combined_evidence)
            attempts += 1
            worker_log.append(
                WorkerLogEntry(worker="synthesizer", summary=f"repair attempt {attempts}, grounded={verdict.grounded}")
            )

        # citations reflect what the answer actually cites, not everything
        # retrieved this run - same reasoning as AgentState.citations in
        # agent/loop.py (see verification.py's VerificationResult docstring).
        used_tags = extract_citation_tags(draft)
        return MultiAgentResult(
            query=query,
            plan=plan,
            worker_log=worker_log,
            final_answer=draft,
            citations=sorted(used_tags),
            citation_sources={tag: citation_map[tag].doc_id for tag in used_tags if tag in citation_map},
            evidence_text=combined_evidence,
            tools_called=tools_called,
            retrieval_queries=retrieval_queries,
            retrieved_chunk_ids=retrieved_chunk_ids,
            grounded=verdict.grounded,
            verification_issues=verdict.issues,
            repair_attempts=attempts,
        )

    async def _verify(self, query: str, draft: str, evidence: str) -> tuple[JudgeVerdict, bool]:
        """Returns (verdict, verifier_ok). "The verifier said this isn't
        grounded" and "the verifier itself broke" are different outcomes and
        the caller needs to tell them apart - conflating them would make an
        infrastructure failure look like a factual one."""
        try:
            return await llm_judge_faithfulness(self._llm, query, draft, evidence), True
        except LLMError as exc:
            return (
                JudgeVerdict(grounded=False, issues=[f"Verifier unavailable: {exc}"], confidence=0.0),
                False,
            )
