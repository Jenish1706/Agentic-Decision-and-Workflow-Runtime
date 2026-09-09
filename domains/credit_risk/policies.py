"""Credit Risk Underwriting Policy Rules.

Encodes deterministic regulatory and risk governance constraints:
1. minimum_credit_score: Score < 650 -> BLOCKED
2. hard_delinquency_cutoff: delinquencies_90d > 0 -> BLOCKED
3. dti_ceiling: DTI > 0.43 (43%) -> BLOCKED
4. high_limit_increase_approval: increase_amount > $5,000 -> REQUIRES_APPROVAL
"""

from typing import Any
from agent_runtime.agent.policy import (
    DecisionContext,
    PolicyEngine,
    PolicyRuleResult,
    PolicyStatus,
    ProposedDecision,
)


def create_credit_risk_policy_engine() -> PolicyEngine:
    """Builds the deterministic underwriting policy engine."""

    def _get_metric_from_proposal_or_evidence(
        metric_name: str, proposal: ProposedDecision, ctx: DecisionContext
    ) -> float | None:
        # First check proposal parameters
        if metric_name in proposal.parameters:
            val = proposal.parameters[metric_name]
            if val is not None:
                return float(val)

        # Then check evidence ledger facts
        if ctx.evidence_ledger:
            facts = ctx.evidence_ledger.extract_numerical_facts()
            for k, v in facts.items():
                if metric_name in k.lower():
                    return float(v)
        return None

    def minimum_credit_score_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        # If action is not an approval or limit increase, rule is not violated
        if proposal.recommended_action not in ("approve", "approve_increase", "increase_limit", "grant_increase"):
            return PolicyRuleResult(
                rule_name="minimum_credit_score",
                passed=True,
                status=PolicyStatus.ALLOWED,
                message="Non-approval action, score threshold not violated.",
            )

        score = _get_metric_from_proposal_or_evidence("credit_score", proposal, ctx)
        if score is not None and score < 650:
            return PolicyRuleResult(
                rule_name="minimum_credit_score",
                passed=False,
                status=PolicyStatus.BLOCKED,
                message=f"Credit score ({score:.0f}) is below the required 650 threshold. Increase blocked.",
            )
        return PolicyRuleResult(
            rule_name="minimum_credit_score",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="Credit score satisfies minimum requirements.",
        )

    def hard_delinquency_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        if proposal.recommended_action not in ("approve", "approve_increase", "increase_limit", "grant_increase"):
            return PolicyRuleResult(
                rule_name="hard_delinquency_cutoff",
                passed=True,
                status=PolicyStatus.ALLOWED,
                message="Non-approval action, delinquency check passed.",
            )

        delinq = _get_metric_from_proposal_or_evidence("delinquencies_90d", proposal, ctx)
        if delinq is not None and delinq > 0:
            return PolicyRuleResult(
                rule_name="hard_delinquency_cutoff",
                passed=False,
                status=PolicyStatus.BLOCKED,
                message=f"Account has {int(delinq)} severe (90+ day) delinquency on record. Increase blocked.",
            )
        return PolicyRuleResult(
            rule_name="hard_delinquency_cutoff",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="No severe 90-day delinquencies detected.",
        )

    def dti_ceiling_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        if proposal.recommended_action not in ("approve", "approve_increase", "increase_limit", "grant_increase"):
            return PolicyRuleResult(
                rule_name="dti_ceiling",
                passed=True,
                status=PolicyStatus.ALLOWED,
                message="Non-approval action, DTI check passed.",
            )

        dti = _get_metric_from_proposal_or_evidence("dti_ratio", proposal, ctx)
        if dti is not None:
            # Handle both fractional 0.48 and percentage 48.0
            normalized_dti = dti / 100.0 if dti > 1.0 else dti
            if normalized_dti > 0.43:
                return PolicyRuleResult(
                    rule_name="dti_ceiling",
                    passed=False,
                    status=PolicyStatus.BLOCKED,
                    message=f"Debt-to-Income ratio ({normalized_dti:.1%}) exceeds maximum allowable 43.0% ceiling. Increase blocked.",
                )
        return PolicyRuleResult(
            rule_name="dti_ceiling",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="Debt-to-income ratio is within safe limit.",
        )

    def high_limit_increase_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        inc = _get_metric_from_proposal_or_evidence("increase_amount", proposal, ctx)
        if inc is not None and inc > 5000:
            return PolicyRuleResult(
                rule_name="high_limit_increase_approval",
                passed=False,
                status=PolicyStatus.REQUIRES_APPROVAL,
                message=f"Requested increase amount of ${inc:,.2f} exceeds autonomous threshold ($5,000). Senior underwriter sign-off required.",
            )
        return PolicyRuleResult(
            rule_name="high_limit_increase_approval",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="Increase amount is within autonomous approval limits.",
        )

    return PolicyEngine(
        rules=[
            minimum_credit_score_rule,
            hard_delinquency_rule,
            dti_ceiling_rule,
            high_limit_increase_rule,
        ]
    )
