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


def test_proposed_decision_validation():
    # Valid proposal
    p = ProposedDecision(
        intent="Rollout feature",
        recommended_action="Deploy variant B to 100% traffic",
        parameters={"variant": "B", "rollout_pct": 100},
        reasoning="Variant B showed 12% lift with p=0.002",
        supporting_evidence_ids=["ev-sql-001"],
        confidence=0.95,
    )
    assert p.confidence == 0.95

    # Out of range confidence should fail
    with pytest.raises(ValidationError):
        ProposedDecision(
            intent="Test",
            recommended_action="Do X",
            reasoning="Reason",
            confidence=1.5,
        )


def test_policy_engine_allowed_flow():
    engine = PolicyEngine()
    ledger = EvidenceLedger()
    ledger.add(
        source=EvidenceSource.ANALYTICS_SQL,
        query_or_input="q",
        data={"metric": 1},
        provenance="table",
        summary="",
        evidence_id="ev-sql-001",
    )

    proposal = ProposedDecision(
        intent="Optimize checkout",
        recommended_action="Update checkout button color",
        reasoning="Evidence supports change",
        supporting_evidence_ids=["ev-sql-001"],
        confidence=0.85,
    )
    context = DecisionContext(subject_or_entity="checkout", evidence_ledger=ledger)
    evaluation = engine.evaluate(proposal, context)

    assert evaluation.is_allowed
    assert evaluation.final_status == PolicyStatus.ALLOWED
    assert evaluation.enforced_action == proposal.recommended_action


def test_policy_engine_missing_evidence_blocks():
    engine = PolicyEngine()
    ledger = EvidenceLedger()  # Empty ledger

    proposal = ProposedDecision(
        intent="Action",
        recommended_action="Action X",
        reasoning="Reasoning citing ghost evidence",
        supporting_evidence_ids=["ev-nonexistent-999"],
        confidence=0.8,
    )
    context = DecisionContext(subject_or_entity="test", evidence_ledger=ledger)
    evaluation = engine.evaluate(proposal, context)

    assert evaluation.is_blocked
    assert evaluation.final_status == PolicyStatus.BLOCKED
    assert any("non-existent evidence" in r.message for r in evaluation.rule_results)


def test_policy_precedence_blocked_over_approval_over_allowed():
    def approval_rule(prop, ctx):
        return PolicyRuleResult(
            rule_name="budget_threshold",
            passed=False,
            status=PolicyStatus.REQUIRES_APPROVAL,
            message="Budget change over 10% requires manager approval",
        )

    def blocked_rule(prop, ctx):
        return PolicyRuleResult(
            rule_name="compliance_check",
            passed=False,
            status=PolicyStatus.BLOCKED,
            message="Violates hard security policy",
        )

    def passing_rule(prop, ctx):
        return PolicyRuleResult(
            rule_name="syntax_check",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="Syntax is valid",
        )

    # When both approval and blocked fail, BLOCKED takes precedence
    engine = PolicyEngine(rules=[passing_rule, approval_rule, blocked_rule])
    proposal = ProposedDecision(
        intent="High stakes action",
        recommended_action="Execute transfer",
        reasoning="Reason",
        confidence=0.9,
    )
    eval_blocked = engine.evaluate(proposal)
    assert eval_blocked.is_blocked
    assert eval_blocked.final_status == PolicyStatus.BLOCKED

    # When only approval fails, REQUIRES_APPROVAL takes precedence over ALLOWED
    engine_approval = PolicyEngine(rules=[passing_rule, approval_rule])
    eval_approval = engine_approval.evaluate(proposal)
    assert eval_approval.requires_approval
    assert eval_approval.final_status == PolicyStatus.REQUIRES_APPROVAL
