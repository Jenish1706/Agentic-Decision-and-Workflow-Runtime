"""The agent's decision schema. This is the JSON shape the LLM is
constrained to produce at every step - the entire "reasoning" surface of
the agent loop is this one object, chosen once per step and acted on."""

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
    query: str | None = Field(default=None, description="Search query text. Required when action='retrieve'.")
    tool_name: str | None = Field(default=None, description="Exact tool name. Required when action='call_tool'.")
    tool_arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool call.")
    answer: str | None = Field(
        default=None, description="The final answer, with [n] citations. Required when action='final'."
    )
    proposed_decision: ProposedDecision | None = Field(
        default=None, description="Structured business decision proposal, if proposing an actionable decision."
    )

    @model_validator(mode="after")
    def _require_fields_for_action(self) -> "ActionDecision":
        # Note: an empty 'query' on RETRIEVE is deliberately NOT rejected here
        # (unlike the other two actions) - see loop.py's fallback. Rejecting
        # it was observed, running this project's own demo against Ollama, to
        # sometimes fail all retries in a row on this exact field and abort
        # the whole run over what has a perfectly sensible default: search
        # using the original question.
        if self.action is ActionType.CALL_TOOL and not self.tool_name:
            raise ValueError("action='call_tool' requires a non-empty 'tool_name'")
        if self.action is ActionType.FINAL and not self.answer:
            raise ValueError("action='final' requires a non-empty 'answer'")
        return self
