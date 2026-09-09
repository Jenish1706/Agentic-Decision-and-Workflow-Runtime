"""LLM-as-judge checks, shared by the multi-agent Verifier worker and the
evaluation harness (evaluation/rag_metrics.py, evaluation/agent_metrics.py).
Kept separate from verification.py's rule-based check because it's a
different tool for a different job: slower and costs a real LLM call, but
catches unsupported reasoning that a tag-matching regex never could."""

from pydantic import BaseModel, Field, field_validator

from ..llm.base import LLMClient, Message


class JudgeVerdict(BaseModel):
    grounded: bool = Field(description="True only if every factual claim in the answer is supported by the evidence.")
    issues: list[str] = Field(default_factory=list, description="Specific unsupported claims, if any.")
    confidence: float = Field(description="Confidence between 0.0 and 1.0 (NOT a percentage).")

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: object) -> object:
        """Small models routinely answer '70' when asked for a 0-1
        confidence, no matter how the field is described. Observed for real
        against Ollama: a bare `le=1.0` constraint turned that into a hard
        LLMError that aborted an entire multi-agent run. Rescaling an
        obvious percentage is strictly better than discarding a verdict
        that was otherwise perfectly usable.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        if 1.0 < value <= 100.0:
            value = value / 100.0
        return min(max(float(value), 0.0), 1.0)


class TaskSuccessVerdict(BaseModel):
    success: bool = Field(description="True if the actual answer conveys the same key facts as the expected answer.")
    reasoning: str


_FAITHFULNESS_SYSTEM_PROMPT = (
    "You are a strict fact-checker. Given a question, the evidence available to answer it, "
    "and a candidate answer, decide whether every factual claim in the answer is actually "
    "supported by the evidence. Do not use outside knowledge or assume anything not stated "
    "in the evidence. A refusal to answer (e.g. 'the knowledge base does not contain this') "
    "counts as grounded when the evidence genuinely does not cover the question."
)


async def llm_judge_faithfulness(llm: LLMClient, question: str, answer: str, evidence_text: str) -> JudgeVerdict:
    messages = [
        Message(role="system", content=_FAITHFULNESS_SYSTEM_PROMPT),
        Message(
            role="user",
            content=f"Question: {question}\n\nEvidence:\n{evidence_text or '(none retrieved)'}\n\nCandidate answer:\n{answer}",
        ),
    ]
    return await llm.decide(messages, JudgeVerdict)


_TASK_SUCCESS_SYSTEM_PROMPT = (
    "You grade whether an assistant's answer conveys the same key facts as a reference answer. "
    "Wording, order, and phrasing do not matter - only whether the key facts match. "
    "Minor omissions of secondary detail are acceptable; wrong or contradictory facts are not."
)


async def llm_judge_task_success(
    llm: LLMClient, question: str, expected_answer: str, actual_answer: str
) -> TaskSuccessVerdict:
    messages = [
        Message(role="system", content=_TASK_SUCCESS_SYSTEM_PROMPT),
        Message(
            role="user",
            content=(
                f"Question: {question}\n\nReference answer: {expected_answer}\n\n"
                f"Actual answer: {actual_answer}\n\nDoes the actual answer match the reference?"
            ),
        ),
    ]
    return await llm.decide(messages, TaskSuccessVerdict)
