"""Deterministic Decision and Policy Engine.

Invariant: LLM proposes; deterministic components verify and constrain.
The LLM produces a `ProposedDecision`. The `PolicyEngine` independently evaluates it
against registered deterministic rules without invoking any LLM.
Precedence: BLOCKED > REQUIRES_APPROVAL > ALLOWED.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from pydantic import BaseModel, Field

from .evidence import EvidenceLedger


class PolicyStatus(str, Enum):
    ALLOWED = "allowed"
    REQUIRES_APPROVAL = "requires_approval"
    BLOCKED = "blocked"


class ProposedDecision(BaseModel):
    intent: str
    recommended_action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True)
class PolicyRuleResult:
    rule_name: str
    passed: bool
    status: PolicyStatus
    message: str


@dataclass(frozen=True)
class PolicyEvaluation:
    final_status: PolicyStatus
    rule_results: list[PolicyRuleResult]
    proposed_action: str
    enforced_action: str | None = None
    audit_notes: str = ""

    @property
    def is_allowed(self) -> bool:
        return self.final_status == PolicyStatus.ALLOWED

    @property
    def requires_approval(self) -> bool:
        return self.final_status == PolicyStatus.REQUIRES_APPROVAL

    @property
    def is_blocked(self) -> bool:
        return self.final_status == PolicyStatus.BLOCKED

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_status": self.final_status.value,
            "proposed_action": self.proposed_action,
            "enforced_action": self.enforced_action,
            "audit_notes": self.audit_notes,
            "rule_results": [
                {
                    "rule_name": r.rule_name,
                    "passed": r.passed,
                    "status": r.status.value,
                    "message": r.message,
                }
                for r in self.rule_results
            ],
        }


@dataclass(frozen=True)
class DecisionContext:
    subject_or_entity: str
    quantitative_facts: dict[str, Any] = field(default_factory=dict)
    evidence_ledger: EvidenceLedger | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Type alias for deterministic policy rule function
# Signature: (proposal: ProposedDecision, context: DecisionContext) -> PolicyRuleResult
PolicyRuleFn = Callable[[ProposedDecision, DecisionContext], PolicyRuleResult]


class PolicyEngine:
    """Deterministic policy rule evaluator. Does NOT call an LLM.

    Precedence order:
    BLOCKED > REQUIRES_APPROVAL > ALLOWED
    """

    def __init__(self, rules: list[PolicyRuleFn] | None = None):
        self._rules: list[PolicyRuleFn] = list(rules or [])

    def add_rule(self, rule_fn: PolicyRuleFn) -> None:
        self._rules.append(rule_fn)

    def evaluate(self, proposal: ProposedDecision, context: DecisionContext | None = None) -> PolicyEvaluation:
        ctx = context or DecisionContext(subject_or_entity=proposal.intent)
        results: list[PolicyRuleResult] = []

        # Standard baseline check: verify all cited evidence IDs exist in ledger if ledger is supplied
        if ctx.evidence_ledger is not None and proposal.supporting_evidence_ids:
            missing_ids = [eid for eid in proposal.supporting_evidence_ids if ctx.evidence_ledger.get_by_id(eid) is None]
            if missing_ids:
                results.append(
                    PolicyRuleResult(
                        rule_name="evidence_existence_check",
                        passed=False,
                        status=PolicyStatus.BLOCKED,
                        message=f"Proposal cites non-existent evidence IDs: {missing_ids}",
                    )
                )

        # Evaluate registered deterministic rules
        for rule_fn in self._rules:
            try:
                res = rule_fn(proposal, ctx)
                results.append(res)
            except Exception as exc:
                results.append(
                    PolicyRuleResult(
                        rule_name=getattr(rule_fn, "__name__", "custom_rule"),
                        passed=False,
                        status=PolicyStatus.BLOCKED,
                        message=f"Rule execution raised error: {exc}",
                    )
                )

        # Determine final status by deterministic precedence:
        # BLOCKED > REQUIRES_APPROVAL > ALLOWED
        statuses = [r.status for r in results if not r.passed]
        if PolicyStatus.BLOCKED in statuses:
            final_status = PolicyStatus.BLOCKED
            enforced_action = "BLOCKED: Proposed action rejected due to policy violation."
        elif PolicyStatus.REQUIRES_APPROVAL in statuses:
            final_status = PolicyStatus.REQUIRES_APPROVAL
            enforced_action = "REQUIRES_APPROVAL: Proposed action requires human authorization."
        else:
            final_status = PolicyStatus.ALLOWED
            enforced_action = proposal.recommended_action

        audit_lines = [f"Rule '{r.rule_name}': {r.status.value.upper()} - {r.message}" for r in results]
        audit_notes = "; ".join(audit_lines) if audit_lines else "All policy checks passed."

        return PolicyEvaluation(
            final_status=final_status,
            rule_results=results,
            proposed_action=proposal.recommended_action,
            enforced_action=enforced_action,
            audit_notes=audit_notes,
        )
