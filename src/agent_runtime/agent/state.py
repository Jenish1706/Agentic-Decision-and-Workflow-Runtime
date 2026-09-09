"""Agent state: the per-run scratchpad. This is deliberately NOT memory -
it exists only for the duration of one `AgentRuntime.run()` call and is
thrown away afterwards (see agent/memory/ for what actually persists).
"""

from enum import Enum

from typing import Any
from pydantic import BaseModel, Field

from .actions import ActionDecision
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

    # Search-efficiency instrumentation. Recorded per retrieval so the
    # evaluation harness can measure what searching actually cost, separately
    # from whether the answer was right - see evaluation/retrieval_metrics.py.
    # Both keep duplicates on purpose: the repeats ARE the signal.
    retrieval_queries: list[str] = Field(
        default_factory=list, description="Effective query text of every retrieval, in order, duplicates kept."
    )
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list, description="Every chunk id returned by every retrieval, duplicates kept."
    )

    evidence_ledger: EvidenceLedger = Field(default_factory=EvidenceLedger)
    proposed_decision: ProposedDecision | None = None
    policy_evaluation: PolicyEvaluation | None = None
    audit_trace: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def tool_calls(self) -> list[Step]:
        from .actions import ActionType

        return [s for s in self.steps if s.decision.action is ActionType.CALL_TOOL]
