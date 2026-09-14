"""Document-audit and SLA policy rules for the multimodal investigation domain.

Rules (limits come from `thresholds` in domain_manifest.yaml):
  payment actions (pay / approve / disburse / settle ...)
    invoice_math_reconciliation   line items != subtotal, or subtotal + tax != total  -> BLOCKED
    duplicate_invoice_prevention  flagged duplicate, or already settled               -> BLOCKED
    executive_expense_approval    total > max_autonomous_expense                      -> REQUIRES_APPROVAL
  SLA penalty actions (penalty / service credit ...)
    sla_breach_detection          monthly average uptime < contractual threshold      -> REQUIRES_APPROVAL
                                  no breach (a penalty credit is not justified)       -> BLOCKED
  Any fact the evidence does not establish                                             -> REQUIRES_APPROVAL

WHAT CHANGED AND WHY
    - Facts come from the extracted document or chart, scoped to the
      document_id / artifact_id the proposal names. The old rules took
      subtotal, total, `is_duplicate` and `settlement_status` from the
      proposal first - a model could clear a duplicate by writing
      {"is_duplicate": false} - and otherwise from whichever invoice happened
      to appear first in the ledger.
    - Flags were read with `bool(value)`, and `bool("false")` is True: a
      duplicate flag serialized as the string "false" blocked a legitimate
      invoice (the same bug in the operations domain turned an override of
      "false" into an authorized one). Flags are parsed strictly now.
    - Settlement statuses are matched against explicit lists. The old
      substring test ("paid" in status and "unpaid" not in status) treated a
      missing status - which the old server defaulted to "unpaid" - as safe
      to pay.
    - Arithmetic uses a half-cent tolerance. The old `diff > 0.01` let an
      exact one-cent discrepancy through.
    - The SLA rule compares the MONTHLY average with the contractual
      threshold, which is what the contract's penalty clause is about. The old
      rule fired on the minimum weekly uptime, so one bad week would trigger a
      penalty review even when the month as a whole met the SLA.
    - Payment rules apply only to payment actions and the SLA rule only to
      penalty actions; previously every rule ran against every proposal.
"""

from collections.abc import Mapping

from agent_runtime.agent.facts import (
    action_tokens,
    as_bool,
    as_number,
    is_governed_action,
    iter_scoped_mappings,
    normalize_identifier,
    numeric_candidates,
    proposal_target,
    resolve_fact,
)
from agent_runtime.agent.policy import DecisionContext, PolicyEngine, PolicyRuleResult, ProposedDecision

REQUIRED_THRESHOLDS = ("max_autonomous_expense",)

_PAYMENT = frozenset({"pay", "payment", "disburse", "disbursement", "settle", "settlement", "reimburse", "remit", "approve"})
_NON_PAYMENT = frozenset(
    {"reject", "decline", "deny", "hold", "flag", "dispute", "investigate", "review", "escalate", "audit", "void",
     "cancel", "penalty", "sla", "credit", "credits"}
)
_PENALTY = frozenset({"penalty", "penalties", "sla", "credit", "credits", "compensation"})

_PAYABLE_STATUSES = frozenset({"unpaid", "open", "pending", "due", "overdue", "outstanding"})
_SETTLED_STATUSES = frozenset({"paid", "already_paid", "settled", "partially_paid"})
_HALF_CENT = 0.005


def _target_document(proposal: ProposedDecision) -> str | None:
    reference = proposal_target(proposal.parameters, "document_id", "invoice_id", "document_path")
    if reference is None:
        return None
    return reference[:-5] if reference.casefold().endswith(".json") else reference


def _target_artifact(proposal: ProposedDecision, ctx: DecisionContext) -> str | None:
    """The chart a penalty proposal is about. Accepts the artifact id or the
    chart's file name, mapping the latter to the id through chart evidence."""
    reference = proposal_target(proposal.parameters, "artifact_id", "chart_id", "artifact_name", "image_path")
    if reference is None:
        return None
    key = normalize_identifier(reference[:-5] if reference.casefold().endswith(".json") else reference)
    for _, _, mapping in iter_scoped_mappings(ctx.evidence_ledger, None):
        artifact_id = mapping.get("artifact_id")
        if artifact_id is not None and key in {
            normalize_identifier(artifact_id),
            normalize_identifier(mapping.get("artifact_name") or ""),
        }:
            return str(artifact_id)
    return reference


def _payment_preconditions(rule: str, proposal: ProposedDecision) -> PolicyRuleResult | str:
    if not is_governed_action(proposal.recommended_action, governed=_PAYMENT, exempt=_NON_PAYMENT):
        return PolicyRuleResult.allowed(rule, f"'{proposal.recommended_action}' is not a payment; not applicable.")
    document = _target_document(proposal)
    if document is None:
        return PolicyRuleResult.requires_approval(rule, "The proposal does not identify the document_id to be paid.")
    return document


def create_policy_engine(thresholds: Mapping[str, float]) -> PolicyEngine:
    expense_limit = float(thresholds["max_autonomous_expense"])

    def invoice_math_reconciliation(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "invoice_math_reconciliation"
        target = _payment_preconditions(rule, proposal)
        if isinstance(target, PolicyRuleResult):
            return target
        facts = {
            name: resolve_fact(ctx.evidence_ledger, name, entity_key="document_id", entity_value=target)
            for name in ("subtotal", "tax_amount", "total_amount", "line_items")
        }
        numbers = {name: as_number(facts[name].value) for name in ("subtotal", "tax_amount", "total_amount")}
        if any(value is None for value in numbers.values()):
            missing = "; ".join(facts[name].describe() for name, value in numbers.items() if value is None)
            return PolicyRuleResult.requires_approval(rule, f"Cannot reconcile {target}: {missing}.")
        subtotal, tax, total = numbers["subtotal"], numbers["tax_amount"], numbers["total_amount"]

        line_items = facts["line_items"].value if facts["line_items"].found else ()
        amounts = [as_number(item.get("amount")) if isinstance(item, Mapping) else None for item in line_items]
        if amounts and None not in amounts and abs(sum(amounts) - subtotal) > _HALF_CENT:
            return PolicyRuleResult.blocked(
                rule, f"Arithmetic mismatch: line items sum to ${sum(amounts):,.2f}, but the subtotal is ${subtotal:,.2f}."
            )
        if abs(subtotal + tax - total) > _HALF_CENT:
            return PolicyRuleResult.blocked(
                rule,
                f"Arithmetic mismatch: Subtotal (${subtotal:,.2f}) + Tax (${tax:,.2f}) = ${subtotal + tax:,.2f}, but "
                f"Total is billed as ${total:,.2f} (discrepancy of ${total - subtotal - tax:,.2f}).",
            )
        if not all(facts[name].conclusive for name in numbers):
            return PolicyRuleResult.requires_approval(
                rule, f"The invoice amounts in evidence could not be attributed to {target}, so they cannot support a payment."
            )
        return PolicyRuleResult.allowed(rule, f"{target} reconciles: ${subtotal:,.2f} + ${tax:,.2f} = ${total:,.2f}.")

    def duplicate_invoice_prevention(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "duplicate_invoice_prevention"
        target = _payment_preconditions(rule, proposal)
        if isinstance(target, PolicyRuleResult):
            return target
        duplicate = resolve_fact(ctx.evidence_ledger, "is_duplicate", entity_key="document_id", entity_value=target)
        status = resolve_fact(ctx.evidence_ledger, "settlement_status", entity_key="document_id", entity_value=target)

        if any(as_bool(value) is True for value in duplicate.candidates):
            return PolicyRuleResult.blocked(
                rule, f"Duplicate invoice detected: {target} is flagged as a duplicate submission. Payment blocked."
            )
        settled = sorted({str(value) for value in status.candidates if normalize_identifier(value) in _SETTLED_STATUSES})
        if settled:
            return PolicyRuleResult.blocked(
                rule, f"Duplicate payment prevented: {target} is already settled (status {settled}). Payment blocked."
            )
        if not (duplicate.conclusive and status.conclusive) or as_bool(duplicate.value) is not False:
            return PolicyRuleResult.requires_approval(
                rule, f"Duplicate/settlement status of {target} is not established ({duplicate.describe()}; {status.describe()})."
            )
        if normalize_identifier(status.value) not in _PAYABLE_STATUSES:
            return PolicyRuleResult.requires_approval(
                rule, f"Unrecognised settlement status {status.value!r} for {target}; human review required."
            )
        return PolicyRuleResult.allowed(rule, f"{target} is not a duplicate and is unsettled ({status.value}).")

    def executive_expense_approval(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "executive_expense_approval"
        target = _payment_preconditions(rule, proposal)
        if isinstance(target, PolicyRuleResult):
            return target
        total = resolve_fact(ctx.evidence_ledger, "total_amount", entity_key="document_id", entity_value=target)
        totals = numeric_candidates(total)
        if totals and max(totals) > expense_limit:
            return PolicyRuleResult.requires_approval(
                rule,
                f"Expense total (${max(totals):,.2f}) exceeds autonomous ${expense_limit:,.0f} limit. "
                "Executive authorization required.",
            )
        if not totals or not total.conclusive:
            return PolicyRuleResult.requires_approval(rule, f"The total of {target} is not established: {total.describe()}.")
        return PolicyRuleResult.allowed(rule, f"Total ${totals[0]:,.2f} is within the ${expense_limit:,.0f} autonomous limit.")

    def sla_breach_detection(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        rule = "sla_breach_detection"
        if not action_tokens(proposal.recommended_action) & _PENALTY:
            return PolicyRuleResult.allowed(rule, f"'{proposal.recommended_action}' is not an SLA penalty action; not applicable.")
        artifact = _target_artifact(proposal, ctx)
        if artifact is None:
            return PolicyRuleResult.requires_approval(rule, "The proposal does not identify the SLA chart (artifact_id) it relies on.")

        average = resolve_fact(ctx.evidence_ledger, "monthly_average_uptime_pct", entity_key="artifact_id", entity_value=artifact)
        threshold = resolve_fact(ctx.evidence_ledger, "contractual_sla_threshold_pct", entity_key="artifact_id", entity_value=artifact)
        average_pct, threshold_pct = as_number(average.value), as_number(threshold.value)
        # Both outcomes below restrict the decision, but they point in
        # opposite directions, so either one needs conclusive facts.
        if not (average.conclusive and threshold.conclusive) or average_pct is None or threshold_pct is None:
            return PolicyRuleResult.requires_approval(
                rule, f"SLA status of {artifact} is not established ({average.describe()}; {threshold.describe()})."
            )
        if average_pct < threshold_pct:
            return PolicyRuleResult.requires_approval(
                rule,
                f"SLA breach detected: monthly average uptime {average_pct:.2f}% is below the contractual "
                f"{threshold_pct:.2f}% threshold. Commercial penalty review required.",
            )
        return PolicyRuleResult.blocked(
            rule,
            f"No SLA breach: monthly average uptime {average_pct:.2f}% meets the contractual {threshold_pct:.2f}% "
            "threshold, so a penalty credit is not justified.",
        )

    return PolicyEngine(
        rules=[invoice_math_reconciliation, duplicate_invoice_prevention, executive_expense_approval, sla_breach_detection],
        thresholds=thresholds,
        name="multimodal_investigation",
    )
