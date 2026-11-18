"""Credit-risk underwriting policy rules.

Rules (limits come from `thresholds` in domain_manifest.yaml):
    minimum_credit_score          credit_score < min_credit_score             -> BLOCKED
    hard_delinquency_cutoff       delinquencies_90d > 0                       -> BLOCKED
    dti_ceiling                   dti_ratio > max_dti_ratio                   -> BLOCKED
    high_limit_increase_approval  increase > max_autonomous_increase          -> REQUIRES_APPROVAL
    Any fact the evidence does not establish                                  -> REQUIRES_APPROVAL

WHERE FACTS COME FROM (see agent_runtime/agent/facts.py)
    - The proposal names the account (`account_id`) and the increase it would
      grant. Score, delinquencies and DTI are read from evidence scoped to
      that account. The previous rules read them from the proposal first, so
      {"credit_score": 700} in the model's own parameters overrode a 600 in
      the database, and a lookup that found nothing counted as "no violation".
    - Facts may restrict on weak grounds but permit only on conclusive ones:
      a violating value blocks even if it could not be attributed to the
      account, while an acceptable value approves only if it could.
    - The increase is the LARGER of the proposal's amount and the amounts
      requested for this account in evidence, so a proposal cannot slip under
      the approval threshold by understating the request.

WHICH ACTIONS ARE GOVERNED
    Every action that could grant credit. The old rules matched four exact
    strings, so "grant_limit_raise" skipped all of them; now only clearly
    non-granting actions ("decline", "refer_to_underwriter", ...) are exempt
    and anything unrecognised is governed.

DTI UNITS
    dti_ratio is stored as a fraction (0.48 = 48%) and is compared as one. The
    old rule divided any value above 1.0 by 100 to "handle percentages",
    which turned a genuine 120% DTI (1.2) into 1.2% and approved it.
"""

from collections.abc import Mapping

from agent_runtime.agent.facts import (
    ResolvedFact,
    is_governed_action,
    numeric_candidates,
    proposal_number,
    proposal_target,
    resolve_fact,
)
from agent_runtime.agent.policy import DecisionContext, PolicyEngine, PolicyRuleResult, ProposedDecision

REQUIRED_THRESHOLDS = ("min_credit_score", "max_dti_ratio", "max_autonomous_increase")

_GRANTING = frozenset({"approve", "grant", "increase", "raise", "expand", "extend", "accept", "authorize", "allow"})
_NON_GRANTING = frozenset(
    {"decline", "deny", "reject", "refuse", "hold", "defer", "refer", "escalate", "review", "investigate", "monitor", "flag"}
)


def _account(proposal: ProposedDecision) -> str | None:
    return proposal_target(proposal.parameters, "account_id", "account")


def _account_fact(ctx: DecisionContext, account: str, field: str) -> ResolvedFact:
    return resolve_fact(ctx.evidence_ledger, field, entity_key="account_id", entity_value=account)


def _preconditions(rule: str, proposal: ProposedDecision) -> PolicyRuleResult | str:
    """The account a rule should evaluate, or the result to return early."""
    if not is_governed_action(proposal.recommended_action, governed=_GRANTING, exempt=_NON_GRANTING):
        return PolicyRuleResult.allowed(rule, f"'{proposal.recommended_action}' does not grant credit; not applicable.")
    account = _account(proposal)
    if account is None:
        return PolicyRuleResult.requires_approval(rule, "The proposal does not identify an account_id to evaluate.")
    return account


def _unattributed(label: str, account: str) -> str:
    return (
        f"The {label} in evidence could not be attributed to {account} (the query did not return account_id), "
        "so it cannot support an approval."
    )


def create_policy_engine(thresholds: Mapping[str, float]) -> PolicyEngine:
    min_score = float(thresholds["min_credit_score"])
    max_dti = float(thresholds["max_dti_ratio"])
    max_increase = float(thresholds["max_autonomous_increase"])

    def minimum_credit_score(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "minimum_credit_score"
        target = _preconditions(rule, proposal)
        if isinstance(target, PolicyRuleResult):
            return target
        fact = _account_fact(ctx, target, "credit_score")
        scores = numeric_candidates(fact)
        if scores and min(scores) < min_score:
            return PolicyRuleResult.blocked(
                rule, f"Credit score ({min(scores):.0f}) is below the required {min_score:.0f} threshold. Increase blocked."
            )
        if not scores or not fact.found:
            # ACCT-105's NULL score (a thin-file borrower) lands here.
            return PolicyRuleResult.requires_approval(rule, f"Credit score for {target} is not established: {fact.describe()}.")
        if not fact.attributed:
            return PolicyRuleResult.requires_approval(rule, _unattributed("credit score", target))
        if not fact.can_permit:
            return PolicyRuleResult.requires_approval(
                rule, f"Credit score for {target} is from free-form SQL and cannot autonomously permit approval; verified record lookup required."
            )
        return PolicyRuleResult.allowed(rule, f"Credit score {scores[0]:.0f} meets the {min_score:.0f} minimum.")

    def hard_delinquency_cutoff(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "hard_delinquency_cutoff"
        target = _preconditions(rule, proposal)
        if isinstance(target, PolicyRuleResult):
            return target
        fact = _account_fact(ctx, target, "delinquencies_90d")
        counts = numeric_candidates(fact)
        if counts and max(counts) > 0:
            return PolicyRuleResult.blocked(
                rule, f"Account has {max(counts):.0f} severe (90+ day) delinquency record(s). Increase blocked."
            )
        if not counts or not fact.found:
            return PolicyRuleResult.requires_approval(
                rule, f"90-day delinquency history for {target} is not established: {fact.describe()}."
            )
        if not fact.attributed:
            return PolicyRuleResult.requires_approval(rule, _unattributed("delinquency count", target))
        if not fact.can_permit:
            return PolicyRuleResult.requires_approval(
                rule, f"90-day delinquency history for {target} is from free-form SQL and cannot autonomously permit approval; verified record lookup required."
            )
        return PolicyRuleResult.allowed(rule, "No severe 90-day delinquencies on record.")

    def dti_ceiling(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "dti_ceiling"
        target = _preconditions(rule, proposal)
        if isinstance(target, PolicyRuleResult):
            return target
        fact = _account_fact(ctx, target, "dti_ratio")
        ratios = numeric_candidates(fact)
        if ratios and max(ratios) > max_dti:
            return PolicyRuleResult.blocked(
                rule,
                f"Debt-to-Income ratio ({max(ratios):.1%}) exceeds maximum allowable {max_dti:.1%} ceiling. "
                "Increase blocked.",
            )
        if not ratios or not fact.found:
            return PolicyRuleResult.requires_approval(rule, f"DTI ratio for {target} is not established: {fact.describe()}.")
        if not fact.attributed:
            return PolicyRuleResult.requires_approval(rule, _unattributed("DTI ratio", target))
        if not fact.can_permit:
            return PolicyRuleResult.requires_approval(
                rule, f"DTI ratio for {target} is from free-form SQL and cannot autonomously permit approval; verified record lookup required."
            )
        return PolicyRuleResult.allowed(rule, f"DTI ratio {ratios[0]:.1%} is within the {max_dti:.1%} ceiling.")

    def high_limit_increase_approval(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "high_limit_increase_approval"
        target = _preconditions(rule, proposal)
        if isinstance(target, PolicyRuleResult):
            return target
        proposed = proposal_number(proposal.parameters, "increase_amount", "amount")
        requested = _account_fact(ctx, target, "increase_amount")
        amounts = [amount for amount in (proposed, *numeric_candidates(requested)) if amount is not None]
        if not amounts:
            return PolicyRuleResult.requires_approval(
                rule, f"The increase amount for {target} is neither stated in the proposal nor established by evidence."
            )
        increase = max(amounts)
        if increase > max_increase:
            return PolicyRuleResult.requires_approval(
                rule,
                f"Requested increase amount of ${increase:,.2f} exceeds autonomous threshold (${max_increase:,.0f}). "
                "Senior underwriter sign-off required.",
            )
        if proposed is None and not requested.attributed:
            return PolicyRuleResult.requires_approval(rule, _unattributed("increase amount", target))
        return PolicyRuleResult.allowed(
            rule, f"Increase of ${increase:,.2f} is within the ${max_increase:,.0f} autonomous limit."
        )

    return PolicyEngine(
        rules=[minimum_credit_score, hard_delinquency_cutoff, dti_ceiling, high_limit_increase_approval],
        thresholds=thresholds,
        name="credit_risk",
    )
