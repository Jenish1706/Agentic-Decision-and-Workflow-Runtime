"""Multimodal Document & Operational Governance Policy Rules.

Encodes deterministic audit and compliance constraints:
1. invoice_math_reconciliation: abs((subtotal + tax) - total) > 0.01 -> BLOCKED
2. duplicate_invoice_prevention: is_duplicate or already_paid -> BLOCKED
3. executive_expense_approval: total > $1,000.0 -> REQUIRES_APPROVAL
4. sla_breach_detection: min_uptime_pct < 99.90% -> REQUIRES_APPROVAL
"""

from typing import Any
from agent_runtime.agent.policy import (
    DecisionContext,
    PolicyEngine,
    PolicyRuleResult,
    PolicyStatus,
    ProposedDecision,
)


def create_multimodal_policy_engine() -> PolicyEngine:
    """Builds the deterministic multimodal investigation and expense audit policy engine."""

    def _get_metric(metric_name: str, proposal: ProposedDecision, ctx: DecisionContext) -> float | None:
        if metric_name in proposal.parameters:
            v = proposal.parameters[metric_name]
            if v is not None:
                return float(v)

        if ctx.evidence_ledger:
            facts = ctx.evidence_ledger.extract_numerical_facts()
            for k, v in facts.items():
                if metric_name in k.lower():
                    return float(v)
        return None

    def _get_bool(flag_name: str, proposal: ProposedDecision, ctx: DecisionContext) -> bool | None:
        if flag_name in proposal.parameters:
            return bool(proposal.parameters[flag_name])

        if ctx.evidence_ledger:
            for it in ctx.evidence_ledger.all():
                if isinstance(it.data, dict) and flag_name in it.data:
                    return bool(it.data[flag_name])
        return None

    def _get_string(field_name: str, proposal: ProposedDecision, ctx: DecisionContext) -> str | None:
        if field_name in proposal.parameters:
            return str(proposal.parameters[field_name])

        if ctx.evidence_ledger:
            for it in ctx.evidence_ledger.all():
                if isinstance(it.data, dict) and field_name in it.data:
                    return str(it.data[field_name])
        return None

    def invoice_math_reconciliation_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        subtotal = _get_metric("subtotal", proposal, ctx)
        tax = _get_metric("tax_amount", proposal, ctx)
        total = _get_metric("total_amount", proposal, ctx)

        if subtotal is not None and tax is not None and total is not None:
            expected_total = subtotal + tax
            diff = abs(expected_total - total)
            if diff > 0.01:
                return PolicyRuleResult(
                    rule_name="invoice_math_reconciliation",
                    passed=False,
                    status=PolicyStatus.BLOCKED,
                    message=f"Arithmetic mismatch: Subtotal (${subtotal:,.2f}) + Tax (${tax:,.2f}) = ${expected_total:,.2f}, but Total is billed as ${total:,.2f} (discrepancy of ${diff:,.2f}).",
                )

        return PolicyRuleResult(
            rule_name="invoice_math_reconciliation",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="Invoice arithmetic reconciles successfully.",
        )

    def duplicate_invoice_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        is_dup = _get_bool("is_duplicate", proposal, ctx)
        status = _get_string("settlement_status", proposal, ctx)

        is_settled = False
        if status:
            s_lower = status.lower()
            is_settled = (s_lower in ("already_paid", "paid", "settled")) or ("paid" in s_lower and "unpaid" not in s_lower)

        if is_dup is True or is_settled:
            return PolicyRuleResult(
                rule_name="duplicate_invoice_prevention",
                passed=False,
                status=PolicyStatus.BLOCKED,
                message="Duplicate invoice detected: document has already been settled or marked duplicate. Payment blocked.",
            )

        return PolicyRuleResult(
            rule_name="duplicate_invoice_prevention",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="No duplicate billing submission detected.",
        )

    def executive_expense_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        total = _get_metric("total_amount", proposal, ctx)
        if total is not None and total > 1000.0:
            return PolicyRuleResult(
                rule_name="executive_expense_approval",
                passed=False,
                status=PolicyStatus.REQUIRES_APPROVAL,
                message=f"Expense total (${total:,.2f}) exceeds autonomous $1,000 limit. Executive authorization required.",
            )

        return PolicyRuleResult(
            rule_name="executive_expense_approval",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="Expense is within autonomous approval limits.",
        )

    def sla_breach_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        min_uptime = _get_metric("min_uptime_pct", proposal, ctx)
        breach_flag = _get_bool("sla_breach_detected", proposal, ctx)

        if (min_uptime is not None and min_uptime < 99.90) or breach_flag is True:
            return PolicyRuleResult(
                rule_name="sla_breach_detection",
                passed=False,
                status=PolicyStatus.REQUIRES_APPROVAL,
                message=f"SLA breach detected (minimum uptime {min_uptime or 'unknown'}% < 99.90%). Commercial penalty review required.",
            )

        return PolicyRuleResult(
            rule_name="sla_breach_detection",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="SLA availability satisfies contractual commitments.",
        )

    return PolicyEngine(
        rules=[
            invoice_math_reconciliation_rule,
            duplicate_invoice_rule,
            executive_expense_rule,
            sla_breach_rule,
        ]
    )
