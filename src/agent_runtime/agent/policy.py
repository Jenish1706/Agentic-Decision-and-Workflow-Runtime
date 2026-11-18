"""Deterministic decision and policy engine.

CENTRAL INVARIANT
    LLM proposes; deterministic components verify and constrain.

    The LLM may attach a `ProposedDecision` to its final answer. The
    `PolicyEngine` then evaluates that proposal against registered rule
    functions - plain Python, zero LLM calls - and produces one of three
    outcomes with strict precedence:

        BLOCKED  >  REQUIRES_APPROVAL  >  ALLOWED

    i.e. a single BLOCKED rule blocks the decision no matter how many rules
    allow it, and a single REQUIRES_APPROVAL rule escalates it to a human.

WHY "DENY OVERRIDES"
    Governance must fail safe. With deny-overrides precedence, adding a rule
    can only make the system *more* restrictive, never less, so a new rule
    cannot accidentally open a hole. A consequence worth knowing for
    interviews: a rule that returns ALLOWED is a no-op - it cannot override
    anything - which is why this project has no "auto-allow" rules.

FAIL-CLOSED DESIGN CHOICES IN THIS FILE
    - A rule that raises an exception counts as BLOCKED (a broken check must
      not silently pass).
    - A rule whose `passed` flag contradicts its `status` is resolved toward
      the more restrictive reading and flagged in the audit notes.
    - A proposal that cites evidence ids missing from the ledger is BLOCKED.
    - A proposal attached to an answer that FAILED deterministic verification
      (the loop's last-step best-effort acceptance) is escalated to
      REQUIRES_APPROVAL: a decision must not be auto-approved on numbers that
      did not reconcile with evidence.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .evidence import EvidenceLedger


class PolicyStatus(str, Enum):
    ALLOWED = "allowed"
    REQUIRES_APPROVAL = "requires_approval"
    BLOCKED = "blocked"


_ENTITY_EXTRACT_RE = re.compile(
    r"\b((?:acct|account)[_\s-]?\d+|(?:exp|experiment)[_\s-]?\d+|inv[_\s-][\w-]+|inv\d[\w-]*)\b",
    re.IGNORECASE,
)


def _canonicalize_entity(raw: str) -> str:
    """Canonical form for comparing entity identifiers: 'ACCT_101' == 'account 101' == 'acct-101'."""
    cleaned = re.sub(r"[_\s]+", "-", raw.strip().casefold())
    cleaned = re.sub(r"^(account|acct)[_\s-]*", "acct-", cleaned)
    cleaned = re.sub(r"^(experiment|exp)[_\s-]*", "exp-", cleaned)
    cleaned = re.sub(r"^inv[_\s-]*", "inv-", cleaned)
    return cleaned


# Higher number = more restrictive. Used to compute precedence without a
# chain of if/elif, and to resolve inconsistent rule results conservatively.
_RESTRICTIVENESS: dict[PolicyStatus, int] = {
    PolicyStatus.ALLOWED: 0,
    PolicyStatus.REQUIRES_APPROVAL: 1,
    PolicyStatus.BLOCKED: 2,
}


class ProposedDecision(BaseModel):
    """A structured, machine-checkable business decision proposed by the LLM.

    `parameters` should identify the target of the action (e.g. account_id,
    document_id, service_name) and any quantities that *are* the proposal
    (e.g. the traffic percentage of a rollout). Domain rules deliberately do
    NOT trust facts about the world placed here - see agent/facts.py.
    """

    intent: str
    recommended_action: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    reasoning: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0 (not a percentage).")

    @field_validator("confidence", mode="before")
    @classmethod
    def _rescale_percentage_confidence(cls, value: object) -> object:
        """Accept a percentage-style confidence (e.g. 85) as 0.85.

        Small local models routinely write 85 where the schema says 0-1, and
        because this model is nested inside the agent's `ActionDecision`, a
        rejected confidence would invalidate the *entire* step and could end
        the run after retries. Only 2-100 is rescaled: a value such as 1.5 is
        more plausibly a mistake on the 0-1 scale than "1.5 percent", so it is
        still rejected by the ge/le bounds rather than silently becoming
        0.015.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        if 2.0 <= value <= 100.0:
            return value / 100.0
        return value


@dataclass(frozen=True)
class PolicyRuleResult:
    rule_name: str
    passed: bool
    status: PolicyStatus
    message: str

    # Convenience constructors keep domain rules short and guarantee that
    # `passed` and `status` agree for results built through them.
    @classmethod
    def allowed(cls, rule_name: str, message: str) -> "PolicyRuleResult":
        return cls(rule_name=rule_name, passed=True, status=PolicyStatus.ALLOWED, message=message)

    @classmethod
    def requires_approval(cls, rule_name: str, message: str) -> "PolicyRuleResult":
        return cls(rule_name=rule_name, passed=False, status=PolicyStatus.REQUIRES_APPROVAL, message=message)

    @classmethod
    def blocked(cls, rule_name: str, message: str) -> "PolicyRuleResult":
        return cls(rule_name=rule_name, passed=False, status=PolicyStatus.BLOCKED, message=message)

    @property
    def effective_status(self) -> PolicyStatus:
        """The status used for precedence.

        `status` is authoritative, but a result that says passed=False while
        claiming ALLOWED is internally inconsistent - most likely a bug in the
        rule - so it is read as REQUIRES_APPROVAL rather than trusted.
        """
        if self.status is PolicyStatus.ALLOWED and not self.passed:
            return PolicyStatus.REQUIRES_APPROVAL
        return self.status

    @property
    def is_inconsistent(self) -> bool:
        return self.passed != (self.status is PolicyStatus.ALLOWED)


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
                    "rule_name": result.rule_name,
                    "passed": result.passed,
                    "status": result.status.value,
                    "message": result.message,
                }
                for result in self.rule_results
            ],
        }


@dataclass(frozen=True)
class DecisionContext:
    """Everything a rule may consult besides the proposal itself.

    `metadata["answer_verified"]` is set by the agent loop: False means the
    final answer did not pass deterministic verification.
    """

    subject_or_entity: str
    quantitative_facts: Mapping[str, Any] = field(default_factory=dict)
    evidence_ledger: EvidenceLedger | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# A policy rule is any pure function (proposal, context) -> PolicyRuleResult.
PolicyRuleFn = Callable[[ProposedDecision, DecisionContext], PolicyRuleResult]


class PolicyEngine:
    """Deterministic rule evaluator. Never calls an LLM.

    `thresholds` are the numeric constants the rules enforce (loaded from a
    domain manifest - see agent_runtime/domains.py). They are exposed so the
    agent loop can let an answer *mention* a policy threshold ("DTI exceeds
    the 43% ceiling") without numerical verification rejecting 43 as an
    ungrounded figure: the threshold is a real, citable fact, just not one
    that a tool returned.

    `guidance` is prompt text for the model: which action names and parameter
    keys this engine's rules understand, and which evidence they read. It
    never affects evaluation - a proposal that ignores it is still judged by
    the rules, which fail closed on anything they cannot interpret.
    """

    def __init__(
        self,
        rules: list[PolicyRuleFn] | None = None,
        thresholds: Mapping[str, float] | None = None,
        name: str = "policy",
        guidance: str = "",
    ):
        self._rules: list[PolicyRuleFn] = list(rules or [])
        self._thresholds: Mapping[str, float] = MappingProxyType(dict(thresholds or {}))
        self.name = name
        self.guidance = guidance

    @property
    def thresholds(self) -> Mapping[str, float]:
        return self._thresholds

    @property
    def rule_names(self) -> list[str]:
        """Function names of the registered rules. Domain rules are named
        after the rule they implement, which lets the domain loader check the
        manifest's documented rules against the code that actually runs."""
        return [getattr(rule_fn, "__name__", "custom_rule") for rule_fn in self._rules]

    def add_rule(self, rule_fn: PolicyRuleFn) -> None:
        self._rules.append(rule_fn)

    def evaluate(self, proposal: ProposedDecision, context: DecisionContext | None = None) -> PolicyEvaluation:
        ctx = context or DecisionContext(subject_or_entity=proposal.intent)
        results: list[PolicyRuleResult] = []

        # Built-in check 1: cited evidence must exist. A decision that points
        # at evidence the ledger never recorded is citing a fabrication.
        if ctx.evidence_ledger is not None and proposal.supporting_evidence_ids:
            missing_ids = [
                evidence_id
                for evidence_id in proposal.supporting_evidence_ids
                if ctx.evidence_ledger.get_by_id(evidence_id) is None
            ]
            if missing_ids:
                results.append(
                    PolicyRuleResult.blocked(
                        "evidence_existence_check",
                        f"Proposal cites non-existent evidence IDs: {missing_ids}",
                    )
                )

        # Built-in check 2: never auto-approve on an unverified answer.
        if ctx.metadata.get("answer_verified") is False:
            results.append(
                PolicyRuleResult.requires_approval(
                    "decision_verification_gate",
                    "The answer supporting this decision failed deterministic verification "
                    "(citations or numbers did not reconcile with evidence); human review required.",
                )
            )

        # Built-in check 3: Target entity binding (P0-2).
        # A decision is evaluated only when its target matches an entity identified in the user's request.
        # On a mismatch or an ambiguous request, the result is REQUIRES_APPROVAL.
        if ctx.subject_or_entity:
            from .facts import proposal_target
            target = proposal_target(
                proposal.parameters,
                "account_id", "account", "experiment_id", "experiment",
                "service_name", "service", "document_id", "invoice_id", "artifact_id",
            )
            if target is not None:
                norm_target = _canonicalize_entity(target)
                norm_subject = ctx.subject_or_entity.strip().casefold()
                matches = _ENTITY_EXTRACT_RE.findall(norm_subject)
                found_entities = {_canonicalize_entity(m) for m in matches}
                if found_entities:
                    if norm_target not in found_entities:
                        results.append(
                            PolicyRuleResult.requires_approval(
                                "target_entity_binding",
                                f"Proposed target '{target}' does not match entity in user request ({sorted(found_entities)}): '{ctx.subject_or_entity}'.",
                            )
                        )
                    elif len(found_entities) > 1:
                        results.append(
                            PolicyRuleResult.requires_approval(
                                "target_entity_binding",
                                f"User request is ambiguous across multiple entities ({sorted(found_entities)}); human approval required.",
                            )
                        )

        for rule_fn in self._rules:
            try:
                results.append(rule_fn(proposal, ctx))
            except Exception as exc:  # noqa: BLE001 - any rule failure must fail closed, not crash
                results.append(
                    PolicyRuleResult.blocked(
                        getattr(rule_fn, "__name__", "custom_rule"),
                        f"Rule execution raised error: {exc}",
                    )
                )

        final_status = max(
            (result.effective_status for result in results),
            key=_RESTRICTIVENESS.__getitem__,
            default=PolicyStatus.ALLOWED,
        )
        if final_status is PolicyStatus.BLOCKED:
            enforced_action = "BLOCKED: Proposed action rejected due to policy violation."
        elif final_status is PolicyStatus.REQUIRES_APPROVAL:
            enforced_action = "REQUIRES_APPROVAL: Proposed action requires human authorization."
        else:
            enforced_action = proposal.recommended_action

        audit_lines = []
        for result in results:
            line = f"Rule '{result.rule_name}': {result.effective_status.value.upper()} - {result.message}"
            if result.is_inconsistent:
                line += f" [inconsistent rule result: passed={result.passed}, status={result.status.value}]"
            audit_lines.append(line)
        audit_notes = "; ".join(audit_lines) if audit_lines else "All policy checks passed."

        return PolicyEvaluation(
            final_status=final_status,
            rule_results=results,
            proposed_action=proposal.recommended_action,
            enforced_action=enforced_action,
            audit_notes=audit_notes,
        )
