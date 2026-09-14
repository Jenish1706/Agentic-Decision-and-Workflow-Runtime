"""Agent state: the per-run scratchpad.

This is deliberately NOT memory. It exists only for the duration of one
`AgentRuntime.run()` call and is thrown away afterwards (see agent/memory/
for what persists). Four things are easy to conflate and are kept apart:

    STATE     this run's steps, evidence and outcome (this module)
    CONTEXT   what gets rendered into the next LLM prompt (loop._build_messages)
    MEMORY    what survives across runs (agent/memory/)
    RETRIEVAL the mechanism for pulling external text in (rag/, mcp_retriever)
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .actions import ActionDecision, ActionType
from .evidence import EvidenceLedger
from .policy import PolicyEvaluation, ProposedDecision


class StepStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class Step(BaseModel):
    index: int
    decision: ActionDecision
    observation: str
    status: StepStatus


class TerminationReason(str, Enum):
    FINAL_ANSWER = "final_answer"
    MAX_STEPS_REACHED = "max_steps_reached"
    UNRECOVERABLE_ERROR = "unrecoverable_error"


class AgentState(BaseModel):
    query: str
    steps: list[Step] = Field(default_factory=list)
    done: bool = False
    termination_reason: TerminationReason | None = None
    final_answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    citation_sources: dict[str, str] = Field(default_factory=dict, description="citation tag -> doc_id")

    # Whether `final_answer` passed deterministic verification. None until a
    # final answer exists. False happens only when the loop accepts a
    # best-effort answer on its last step: a FINAL_ANSWER termination alone
    # used to be indistinguishable from a verified answer, which let unverified
    # answers count as successes in evaluation and feed policy decisions.
    answer_verified: bool | None = None
    verification_issues: list[str] = Field(default_factory=list)

    # Search-efficiency instrumentation, recorded per retrieval so evaluation
    # can measure what searching cost separately from whether the answer was
    # right (evaluation/retrieval_metrics.py). Duplicates are kept on purpose:
    # the repeats are the signal.
    retrieval_queries: list[str] = Field(
        default_factory=list, description="Effective query text of every retrieval, in order, duplicates kept."
    )
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list, description="Every chunk id returned by every retrieval, duplicates kept."
    )

    evidence_ledger: EvidenceLedger = Field(default_factory=EvidenceLedger)
    proposed_decision: ProposedDecision | None = None
    # "final_step" when the model attached the proposal to its answer,
    # "extracted" when a follow-up call recovered it from the answer text
    # (see actions.DecisionExtraction). Kept so the audit trail never hides
    # which path produced the decision the policy engine judged.
    proposed_decision_source: str | None = None
    policy_evaluation: PolicyEvaluation | None = None
    budget_usage: dict[str, int] = Field(default_factory=dict)
    audit_trace: dict[str, Any] = Field(default_factory=dict)

    # EvidenceLedger and PolicyEvaluation are plain Python classes, not
    # pydantic models; pydantic stores them as-is.
    model_config = {"arbitrary_types_allowed": True}

    def tool_calls(self) -> list[Step]:
        return [step for step in self.steps if step.decision.action is ActionType.CALL_TOOL]
