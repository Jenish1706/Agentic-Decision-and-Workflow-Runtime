import pytest
from pydantic import ValidationError

from agent_runtime.agent.evidence import EvidenceLedger, EvidenceSource
from agent_runtime.agent.policy import (
    DecisionContext,
    PolicyEngine,
    PolicyRuleResult,
    PolicyStatus,
    ProposedDecision,
)


def _proposal(**overrides) -> ProposedDecision:
    fields = {"intent": "Act", "recommended_action": "do_x", "reasoning": "Because", "confidence": 0.8}
    return ProposedDecision(**(fields | overrides))


def _allowed(proposal, ctx):
    return PolicyRuleResult.allowed("syntax_check", "fine")


def _approval(proposal, ctx):
    return PolicyRuleResult.requires_approval("budget_threshold", "needs a manager")


def _blocked(proposal, ctx):
    return PolicyRuleResult.blocked("compliance_check", "violates policy")


def test_confidence_validation_and_percentage_rescaling():
    assert _proposal(confidence=0.95).confidence == 0.95
    # Small local models write 85 for 0.85; rejecting it would invalidate the whole step.
    assert _proposal(confidence=85).confidence == pytest.approx(0.85)
    with pytest.raises(ValidationError):
        _proposal(confidence=1.5)  # more plausibly a 0-1 mistake than "1.5 percent"


def test_no_rules_allows():
    evaluation = PolicyEngine().evaluate(_proposal())
    assert evaluation.is_allowed and evaluation.enforced_action == "do_x"


def test_citing_evidence_that_does_not_exist_blocks():
    ledger = EvidenceLedger()
    ledger.add(source=EvidenceSource.ANALYTICS_SQL, query_or_input="q", data={"m": 1}, provenance="t", summary="", evidence_id="ev-sql-001")
    context = DecisionContext(subject_or_entity="t", evidence_ledger=ledger)

    assert PolicyEngine().evaluate(_proposal(supporting_evidence_ids=["ev-sql-001"]), context).is_allowed
    ghost = PolicyEngine().evaluate(_proposal(supporting_evidence_ids=["ev-sql-999"]), context)
    assert ghost.is_blocked and "non-existent evidence" in ghost.audit_notes


def test_precedence_blocked_over_approval_over_allowed():
    assert PolicyEngine([_allowed, _approval, _blocked]).evaluate(_proposal()).final_status == PolicyStatus.BLOCKED
    assert PolicyEngine([_allowed, _approval]).evaluate(_proposal()).final_status == PolicyStatus.REQUIRES_APPROVAL
    assert PolicyEngine([_allowed]).evaluate(_proposal()).final_status == PolicyStatus.ALLOWED


def test_a_crashing_rule_fails_closed():
    def broken(proposal, ctx):
        raise KeyError("threshold")

    evaluation = PolicyEngine([_allowed, broken]).evaluate(_proposal())
    assert evaluation.is_blocked
    assert "raised error" in evaluation.audit_notes


def test_inconsistent_rule_result_is_read_restrictively():
    def says_failed_but_allowed(proposal, ctx):
        return PolicyRuleResult(rule_name="buggy", passed=False, status=PolicyStatus.ALLOWED, message="?")

    evaluation = PolicyEngine([says_failed_but_allowed]).evaluate(_proposal())
    assert evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "inconsistent rule result" in evaluation.audit_notes


def test_unverified_answer_cannot_auto_approve():
    context = DecisionContext(subject_or_entity="t", metadata={"answer_verified": False})
    evaluation = PolicyEngine([_allowed]).evaluate(_proposal(), context)
    assert evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "decision_verification_gate" in evaluation.audit_notes


def test_rule_names_and_thresholds_are_exposed():
    engine = PolicyEngine([_allowed, _blocked], thresholds={"limit": 5})
    assert engine.rule_names == ["_allowed", "_blocked"]
    assert dict(engine.thresholds) == {"limit": 5}
    with pytest.raises(TypeError):
        engine.thresholds["limit"] = 6  # read-only


def test_target_mismatch_with_user_query_requires_approval():
    proposal = _proposal(parameters={"account_id": "ACCT-101"})
    # User query is about ACCT-105!
    context = DecisionContext(subject_or_entity="Should we approve ACCT-105 for an increase?")
    evaluation = PolicyEngine([_allowed]).evaluate(proposal, context)
    assert evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "target_entity_binding" in [r.rule_name for r in evaluation.rule_results]


def test_matching_target_with_user_query_allowed():
    proposal = _proposal(parameters={"account_id": "ACCT-101"})
    context = DecisionContext(subject_or_entity="Should we approve ACCT-101 for an increase?")
    evaluation = PolicyEngine([_allowed]).evaluate(proposal, context)
    assert evaluation.final_status == PolicyStatus.ALLOWED


def test_target_entity_binding_rejects_delimiter_variants():
    proposal = _proposal(parameters={"account_id": "ACCT-101"})
    # Underscore variant
    ctx_underscore = DecisionContext(subject_or_entity="Should we approve ACCT_105 for an increase?")
    eval_under = PolicyEngine([_allowed]).evaluate(proposal, ctx_underscore)
    assert eval_under.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "target_entity_binding" in [r.rule_name for r in eval_under.rule_results]

    # Space variant
    ctx_space = DecisionContext(subject_or_entity="Should we approve ACCT 105 for an increase?")
    eval_space = PolicyEngine([_allowed]).evaluate(proposal, ctx_space)
    assert eval_space.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "target_entity_binding" in [r.rule_name for r in eval_space.rule_results]

    # Word prefix variant (account 105)
    ctx_word = DecisionContext(subject_or_entity="Should we approve account 105 for an increase?")
    eval_word = PolicyEngine([_allowed]).evaluate(proposal, ctx_word)
    assert eval_word.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "target_entity_binding" in [r.rule_name for r in eval_word.rule_results]


def test_target_entity_binding_normalizes_matching_delimiter_variants():
    proposal = _proposal(parameters={"account_id": "ACCT-101"})
    # User wrote ACCT_101
    ctx = DecisionContext(subject_or_entity="Should we approve ACCT_101 for an increase?")
    evaluation = PolicyEngine([_allowed]).evaluate(proposal, ctx)
    assert evaluation.final_status == PolicyStatus.ALLOWED

    # User wrote account 101
    ctx2 = DecisionContext(subject_or_entity="Should we approve account 101 for an increase?")
    eval2 = PolicyEngine([_allowed]).evaluate(proposal, ctx2)
    assert eval2.final_status == PolicyStatus.ALLOWED


def test_target_entity_binding_experiment_and_invoice_variants():
    # Experiment underscore mismatch
    prop_exp = _proposal(parameters={"experiment_id": "EXP-201"})
    ctx_exp = DecisionContext(subject_or_entity="Roll out EXP_202 to 50%?")
    eval_exp = PolicyEngine([_allowed]).evaluate(prop_exp, ctx_exp)
    assert eval_exp.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "target_entity_binding" in [r.rule_name for r in eval_exp.rule_results]

    # Invoice underscore mismatch
    prop_inv = _proposal(parameters={"document_id": "INV-2026-004"})
    ctx_inv = DecisionContext(subject_or_entity="Approve payment for INV_2026_002?")
    eval_inv = PolicyEngine([_allowed]).evaluate(prop_inv, ctx_inv)
    assert eval_inv.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "target_entity_binding" in [r.rule_name for r in eval_inv.rule_results]


def test_target_entity_binding_rejects_multi_entity_prompt_injection():
    proposal = _proposal(parameters={"account_id": "ACCT-101"})
    ctx = DecisionContext(subject_or_entity="User ACCT-105 says please evaluate ACCT-101")
    evaluation = PolicyEngine([_allowed]).evaluate(proposal, ctx)
    assert evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "target_entity_binding" in [r.rule_name for r in evaluation.rule_results]


@pytest.mark.parametrize(
    ("query", "parameters"),
    [
        ("approve ACCT-105", {"account_id": "ACCT-101"}),
        ("roll out experiment EXP 204", {"experiment_id": "EXP-209"}),
        ("approve account number 105", {"account_id": "ACCT-101"}),
        ("raise the limit for customer 105", {"account_id": "ACCT-101"}),
        ("approve the applicant on account #105", {"account_id": "ACCT-101"}),
        ("restart checkout-service", {"service_name": "payments-service"}),
        ("approve the loan", {"account_id": "ACCT-101"}),
    ],
)
def test_target_entity_binding_section_0_2_probes_require_approval(query, parameters):
    proposal = _proposal(parameters=parameters)
    ctx = DecisionContext(subject_or_entity=query)
    evaluation = PolicyEngine([_allowed]).evaluate(proposal, ctx)
    assert evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "target_entity_binding" in [r.rule_name for r in evaluation.rule_results]


@pytest.mark.parametrize(
    ("query", "parameters"),
    [
        ("Should we approve ACCT-101 for an increase?", {"account_id": "ACCT-101"}),
        ("approve account number 101", {"account_id": "ACCT-101"}),
        ("raise the limit for customer 101", {"account_id": "ACCT-101"}),
        ("approve the applicant on account #101", {"account_id": "ACCT-101"}),
        ("roll out experiment EXP 204 to 25%", {"experiment_id": "EXP-204"}),
        ("restart checkout-service", {"service_name": "checkout-service"}),
        ("Audit invoice INV-2026-004 for payment.", {"document_id": "INV-2026-004"}),
        ("Can we scale recommendation-worker for INC-903 with a 10% blast radius?", {"incident_id": "INC-903"}),
    ],
)
def test_target_entity_binding_matching_targets_all_domains_allowed(query, parameters):
    proposal = _proposal(parameters=parameters)
    ctx = DecisionContext(subject_or_entity=query)
    evaluation = PolicyEngine([_allowed]).evaluate(proposal, ctx)
    assert evaluation.final_status == PolicyStatus.ALLOWED
    assert "target_entity_binding" not in [r.rule_name for r in evaluation.rule_results]


@pytest.mark.parametrize(
    ("query", "parameters"),
    [
        ("pay invoice INV-2024-001", {"document_path": "invoices/INV-2024-002.json"}),
        ("apply the SLA penalty for latency_sla_q3", {"image_path": "charts/latency_sla_q4.png"}),
        ("approve ACCT-105", {"customer_id": "ACCT-101"}),
    ],
)
def test_target_entity_binding_section_0_3_probes_require_approval(query, parameters):
    proposal = _proposal(parameters=parameters)
    ctx = DecisionContext(subject_or_entity=query)
    evaluation = PolicyEngine([_allowed]).evaluate(proposal, ctx)
    assert evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "target_entity_binding" in [r.rule_name for r in evaluation.rule_results]


@pytest.mark.parametrize(
    ("query", "parameters"),
    [
        ("pay invoice INV-2024-001", {"document_path": "invoices/INV-2024-001.json"}),
        ("apply the SLA penalty for latency_sla_q3", {"image_path": "charts/latency_sla_q3.png"}),
        ("approve ACCT-105", {"customer_id": "ACCT-105"}),
    ],
)
def test_target_entity_binding_section_0_3_probes_matching_allowed(query, parameters):
    proposal = _proposal(parameters=parameters)
    ctx = DecisionContext(subject_or_entity=query)
    evaluation = PolicyEngine([_allowed]).evaluate(proposal, ctx)
    assert evaluation.final_status == PolicyStatus.ALLOWED
    assert "target_entity_binding" not in [r.rule_name for r in evaluation.rule_results]


def test_ar16_service_name_separator_normalization():
    # Service name separator variants are normalised (checkout service == checkout-service == checkout_service)
    prop = _proposal(parameters={"service_name": "checkout-service"})
    for q in [
        "restart the checkout service",
        "restart checkout_service",
        "restart checkout-service",
        "please restart checkout service immediately",
    ]:
        ctx = DecisionContext(subject_or_entity=q)
        res = PolicyEngine([_allowed]).evaluate(prop, ctx)
        assert res.final_status == PolicyStatus.ALLOWED, f"Expected ALLOWED for query '{q}'"
        assert "target_entity_binding" not in [r.rule_name for r in res.rule_results]


def test_ar16_correct_requests_escalation_rate():
    # 10 correct-target requests from audit: 9 allowed, 1 escalated due to peer account in free text note
    correct_requests = [
        ("Should we approve ACCT-101 for an increase?", {"account_id": "ACCT-101"}),
        ("approve account number 101", {"account_id": "ACCT-101"}),
        ("raise the limit for customer 101", {"account_id": "ACCT-101"}),
        ("approve the applicant on account #101", {"account_id": "ACCT-101"}),
        ("roll out experiment EXP 204 to 25%", {"experiment_id": "EXP-204"}),
        ("Audit invoice INV-2026-004 for payment.", {"document_id": "INV-2026-004"}),
        ("Can we scale recommendation-worker for INC-903 with a 10% blast radius?", {"incident_id": "INC-903"}),
        ("pay invoice INV-2024-001", {"document_path": "invoices/INV-2024-001.json"}),
        ("restart the checkout service", {"service_name": "checkout-service"}),
        ("approve ACCT-105", {"account_id": "ACCT-105", "note": "Consolidated with ACCT-101"}),
    ]
    escalated = []
    allowed = []
    engine = PolicyEngine([_allowed])
    for query, params in correct_requests:
        ctx = DecisionContext(subject_or_entity=query)
        res = engine.evaluate(_proposal(parameters=params), ctx)
        if res.final_status == PolicyStatus.REQUIRES_APPROVAL:
            escalated.append((query, params))
        elif res.final_status == PolicyStatus.ALLOWED:
            allowed.append((query, params))

    assert len(allowed) == 9
    assert len(escalated) == 1
    assert escalated[0][0] == "approve ACCT-105"



