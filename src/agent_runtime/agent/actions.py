"""The agent's decision schema.

This is the JSON shape the LLM is constrained to produce at every step
(Ollama's `format` parameter enforces the schema during decoding). The entire
"reasoning" surface of the agent loop is this one object, chosen once per
step and acted on.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .policy import ProposedDecision


class ActionType(str, Enum):
    RETRIEVE = "retrieve"
    CALL_TOOL = "call_tool"
    FINAL = "final"


class ActionDecision(BaseModel):
    thought: str = Field(description="One or two sentences: why this action, given what's been observed so far.")
    action: ActionType
    query: str | None = Field(default=None, description="Search query text. Used when action='retrieve'.")
    tool_name: str | None = Field(default=None, description="Exact tool name. Required when action='call_tool'.")
    tool_arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool call.")
    answer: str | None = Field(
        default=None, description="The final answer, with [n] citations. Required when action='final'."
    )
    proposed_decision: ProposedDecision | None = Field(
        default=None,
        description=(
            "Only with action='final', and only when the question asks whether an action should be "
            "taken. Identify the target entity and the action's own quantities in parameters."
        ),
    )

    @model_validator(mode="after")
    def _require_fields_for_action(self) -> "ActionDecision":
        # An empty 'query' on RETRIEVE is deliberately NOT rejected here,
        # unlike the other two actions: the loop falls back to searching with
        # the original question. Rejecting it was observed, against a real
        # local model, to fail every retry on this one field and abort the run
        # over something with a perfectly sensible default.
        if self.action is ActionType.CALL_TOOL and not self.tool_name:
            raise ValueError("action='call_tool' requires a non-empty 'tool_name'")
        if self.action is ActionType.FINAL and not self.answer:
            raise ValueError("action='final' requires a non-empty 'answer'")
        return self


class DecisionExtraction(BaseModel):
    """Schema for the follow-up call that recovers a decision the model
    described in its answer but did not attach as `proposed_decision`.

    WHY A SEPARATE CALL
        In two smoke runs of five domain decision questions each against
        qwen2.5:7b, the model attached the optional nested
        `proposed_decision` in only 1 of the 10 runs - once leaving it null
        while spelling the decision out in prose ("the proposed decision is
        to rollout_variant_b with an experiment_id of 'EXP-204'"). Policy
        governance then never ran. Every field here is REQUIRED, so
        schema-constrained decoding has to produce each one. (Details:
        EVALUATION.md, "real-model smoke runs".)

    The extraction proposes; it does not decide. The policy engine still
    judges the result against evidence, exactly as for an attached proposal,
    and the audit trace records which of the two paths produced it.
    """

    asks_for_decision: bool = Field(
        description="True only if the question asks whether an action should be taken (approve, pay, roll out, ...)."
    )
    recommended_action: str = Field(
        description="The action being evaluated, even if the answer recommends against it. Empty if none."
    )
    parameters: dict[str, Any] = Field(
        description="The action's target identifiers and its own quantities. Empty if none."
    )
    reasoning: str = Field(description="One sentence taken from the answer.")
    confidence: float = Field(description="Confidence between 0.0 and 1.0.")
