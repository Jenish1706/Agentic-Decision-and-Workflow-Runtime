import pytest
from agent_runtime.agent.evidence import EvidenceLedger, EvidenceSource
from agent_runtime.agent.verification import (
    extract_quantitative_claims,
    verify_numerical_grounding,
)


def test_claim_extraction_filters_ids_and_dates():
    text = (
        "In Q3 2024, experiment EXP-204 was evaluated in Step 1. "
        "Conversion fell by 4.2 percentage points to 18.4% across 1,050 sessions, "
        "costing $5,000 in revenue."
    )
    claims = extract_quantitative_claims(text)
    raw_texts = [c.raw_text for c in claims]

    # Non-metric tokens must be excluded
    assert not any("2024" in r for r in raw_texts)
    assert not any("EXP-204" in r for r in raw_texts)
    assert not any("Step 1" in r for r in raw_texts)
    assert not any("Q3" in r for r in raw_texts)

    # Metric tokens must be detected
    assert any("4.2 percentage points" in r for r in raw_texts)
    assert any("18.4%" in r for r in raw_texts)
    assert any("1,050 sessions" in r for r in raw_texts)
    assert any("$5,000" in r for r in raw_texts)


def test_verify_direct_and_derived_numbers():
    ledger = EvidenceLedger()
    ledger.add(
        source=EvidenceSource.ANALYTICS_SQL,
        query_or_input="SELECT baseline, treatment FROM metrics",
        data={"baseline": 0.421, "treatment": 0.379, "revenue": 5000.0},
        provenance="analytics_db",
        summary="Baseline 42.1%, Treatment 37.9%",
    )

    # 1. Valid answer using direct fact (37.9%) and derived difference (4.2 percentage points)
    valid_answer = (
        "During experiment EXP-204, conversion fell by 4.2 percentage points to 37.9% "
        "representing $5,000."
    )
    res = verify_numerical_grounding(valid_answer, ledger)
    assert res.valid
    assert len(res.unverified_claims) == 0


def test_verify_catches_fabricated_numbers():
    ledger = EvidenceLedger()
    ledger.add(
        source=EvidenceSource.ANALYTICS_SQL,
        query_or_input="SELECT conv FROM metrics",
        data={"conv": 0.184},
        provenance="db",
        summary="Conversion 18.4%",
    )

    # Answer fabricates an 85.5% figure
    bad_answer = "Conversion rose to 85.5% with 9,999 users."
    res = verify_numerical_grounding(bad_answer, ledger)
    assert not res.valid
    assert len(res.unverified_claims) > 0
    assert any("85.5%" in c.raw_text for c in res.unverified_claims)


def test_refusal_is_exempt_from_numerical_check():
    ledger = EvidenceLedger()
    refusal = "The knowledge base does not contain sufficient evidence to determine conversion."
    res = verify_numerical_grounding(refusal, ledger)
    assert res.valid
