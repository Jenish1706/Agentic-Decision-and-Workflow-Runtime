"""Experiment-rollout policy rules for the product analytics domain.

Rules (limits come from `thresholds` in domain_manifest.yaml):
    statistical_significance_gate  p_value >= significance_alpha, or the
                                   treatment did not beat control            -> BLOCKED
    high_impact_rollout_approval   traffic percentage > max_autonomous_traffic_pct,
                                   or not stated                              -> REQUIRES_APPROVAL
    Significance not established by evidence                                  -> REQUIRES_APPROVAL

WHAT CHANGED AND WHY
    These rules used to live inside the evaluation harness
    (evaluation/product_analytics_experiment.py), separate from this domain's
    manifest - and the two had drifted: the manifest documented a 20% traffic
    threshold while the code enforced 50%. They now live with the domain, and
    the manifest thresholds are what the code reads.

    - The significance gate pooled every p-value in the ledger, whichever
      experiment it came from, and ignored direction. A significantly WORSE
      treatment also has p < 0.05, so "roll out the losing variant" passed.
      The gate now reads the p-value and signed difference of the experiment
      the proposal names, and requires the treatment to be better.
    - The p-value is compared with the policy's own alpha, not the tool's
      `significant` flag: the model chooses the `alpha` argument of
      test_significance, so trusting that flag would let it pick alpha=0.5.
    - A missing traffic percentage defaulted to 0, i.e. "tiny rollout,
      allowed". It now requires approval; a value such as "100%" (a string)
      is parsed instead of crashing the comparison.
"""

from collections.abc import Mapping

from agent_runtime.agent.facts import (
    is_governed_action,
    numeric_candidates,
    proposal_number,
    proposal_target,
    resolve_fact,
)
from agent_runtime.agent.policy import DecisionContext, PolicyEngine, PolicyRuleResult, ProposedDecision

REQUIRED_THRESHOLDS = ("significance_alpha", "max_autonomous_traffic_pct")

_ROLLOUT = frozenset(
    {"rollout", "roll", "deploy", "launch", "ship", "release", "enable", "expand", "ramp", "graduate", "promote", "adopt"}
)
_NON_ROLLOUT = frozenset(
    {"stop", "halt", "pause", "continue", "investigate", "monitor", "hold", "rollback", "revert", "reject", "keep",
     "analyze", "analyse", "review", "rerun", "iterate", "abandon", "archive", "conclude", "retest"}
)
_TRAFFIC_KEYS = ("traffic_percentage", "traffic_pct", "rollout_pct", "rollout_percentage")


def _not_a_rollout(rule: str, proposal: ProposedDecision) -> PolicyRuleResult | None:
    if is_governed_action(proposal.recommended_action, governed=_ROLLOUT, exempt=_NON_ROLLOUT):
        return None
    return PolicyRuleResult.allowed(rule, f"'{proposal.recommended_action}' is not a rollout; not applicable.")


def create_policy_engine(thresholds: Mapping[str, float]) -> PolicyEngine:
    alpha = float(thresholds["significance_alpha"])
    max_traffic = float(thresholds["max_autonomous_traffic_pct"])

    def statistical_significance_gate(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "statistical_significance_gate"
        if (skip := _not_a_rollout(rule, proposal)) is not None:
            return skip
        experiment = proposal_target(proposal.parameters, "experiment_id", "experiment")
        if experiment is None:
            return PolicyRuleResult.requires_approval(rule, "The rollout proposal does not identify its experiment_id.")

        p_value = resolve_fact(ctx.evidence_ledger, "p_value", entity_key="experiment_id", entity_value=experiment)
        difference = resolve_fact(ctx.evidence_ledger, "difference_pp", entity_key="experiment_id", entity_value=experiment)
        p_values, differences = numeric_candidates(p_value), numeric_candidates(difference)
        if p_values and max(p_values) >= alpha:
            return PolicyRuleResult.blocked(
                rule, f"Cannot roll out {experiment}: p-value {max(p_values):.4f} is not below alpha {alpha}."
            )
        if differences and min(differences) <= 0:
            return PolicyRuleResult.blocked(
                rule,
                f"Cannot roll out {experiment}: the treatment did not outperform control "
                f"(difference {min(differences):+.2f} pp).",
            )
        if not (p_value.conclusive and difference.conclusive and p_values and differences):
            return PolicyRuleResult.requires_approval(
                rule,
                f"Statistical significance of {experiment} is not established by evidence "
                f"({p_value.describe()}; {difference.describe()}); run test_significance.",
            )
        return PolicyRuleResult.allowed(
            rule, f"{experiment}: p-value {p_values[0]:.4f} < alpha {alpha}, treatment ahead by {differences[0]:+.2f} pp."
        )

    def high_impact_rollout_approval(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "high_impact_rollout_approval"
        if (skip := _not_a_rollout(rule, proposal)) is not None:
            return skip
        traffic = proposal_number(proposal.parameters, *_TRAFFIC_KEYS)
        if traffic is None:
            return PolicyRuleResult.requires_approval(
                rule, "The rollout proposal does not state its traffic percentage; human approval required."
            )
        if not 0.0 < traffic <= 100.0:
            return PolicyRuleResult.requires_approval(rule, f"Traffic percentage {traffic:g} is outside (0, 100].")
        if traffic > max_traffic:
            return PolicyRuleResult.requires_approval(
                rule,
                f"Rollout allocation of {traffic:g}% exceeds autonomous threshold ({max_traffic:g}%). "
                "Human approval required.",
            )
        return PolicyRuleResult.allowed(rule, f"Rollout allocation of {traffic:g}% is within the {max_traffic:g}% limit.")

    return PolicyEngine(
        rules=[statistical_significance_gate, high_impact_rollout_approval],
        thresholds=thresholds,
        name="product_analytics",
    )
