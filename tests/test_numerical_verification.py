from agent_runtime.agent.evidence import EvidenceLedger, EvidenceSource
from agent_runtime.agent.verification import extract_quantitative_claims, verify_numerical_grounding


def _ledger(data: dict, summary: str = "") -> EvidenceLedger:
    ledger = EvidenceLedger()
    ledger.add(source=EvidenceSource.ANALYTICS_SQL, query_or_input="q", data=data, provenance="db", summary=summary)
    return ledger


def test_claim_extraction_filters_ids_and_dates():
    text = (
        "In Q3 2024, experiment EXP-204 was evaluated in Step 1. "
        "Conversion fell by 4.2 percentage points to 18.4% across 1,050 sessions, "
        "costing $5,000 in revenue."
    )
    raw_texts = [claim.raw_text for claim in extract_quantitative_claims(text)]

    assert not any(token in raw for raw in raw_texts for token in ("2024", "EXP-204", "Step 1", "Q3"))
    assert any("4.2 percentage points" in raw for raw in raw_texts)
    assert any("18.4%" in raw for raw in raw_texts)
    assert any("1,050 sessions" in raw for raw in raw_texts)
    assert any("$5,000" in raw for raw in raw_texts)


def test_direct_and_derived_numbers_are_verified():
    ledger = _ledger({"conversion_rate_before": 0.421, "conversion_rate_after": 0.379, "revenue": 5000.0})
    result = verify_numerical_grounding("Conversion fell by 4.2 percentage points to 37.9%, representing $5,000.", ledger)
    assert result.valid and not result.unverified_claims


def test_fabricated_numbers_are_caught():
    result = verify_numerical_grounding("Conversion rose to 85.5% with 9,999 users.", _ledger({"conv": 0.184}))
    assert not result.valid
    assert any("85.5%" in claim.raw_text for claim in result.unverified_claims)


def test_rounding_tolerance_follows_the_written_precision():
    ledger = _ledger({"rate_b": 0.0293})
    assert verify_numerical_grounding("Variant B converted at 2.9%.", ledger).valid
    assert verify_numerical_grounding("Variant B converted at 2.93%.", ledger).valid
    assert not verify_numerical_grounding("Variant B converted at 3.1%.", ledger).valid


def test_a_percentage_is_not_grounded_by_a_count():
    # Regression: unit-blind matching let "43%" be "grounded" by 43 purchases.
    assert not verify_numerical_grounding("Conversion was 43%.", _ledger({"purchases": 43})).valid


def test_a_calculator_result_can_ground_a_percentage():
    # Found in a real run: the model computed "3000 / 5000 * 100" with the
    # calculator, as the prompt instructs, and was rejected for writing "60%".
    ledger = EvidenceLedger()
    ledger.add(
        source=EvidenceSource.CALCULATOR,
        query_or_input='{"expression": "3000 / 5000 * 100"}',
        data={"expression": "3000 / 5000 * 100", "result": 60.0},
        provenance="tool:calculator",
        summary="3000 / 5000 * 100 = 60.0",
    )
    assert verify_numerical_grounding("The increase is 60% of the current limit.", ledger).valid
    assert verify_numerical_grounding("The increase is 60 dollars.", ledger).valid


def test_direction_reversal_is_rejected():
    ledger = _ledger({"difference_pp": -1.3})
    assert verify_numerical_grounding("Conversion fell by 1.3 percentage points.", ledger).valid
    assert not verify_numerical_grounding("Conversion rose by 1.3 percentage points.", ledger).valid


def test_numbers_from_the_question_and_policy_thresholds_are_grounded():
    ledger = _ledger({"dti_ratio": 0.48})
    answer = "The 48% DTI exceeds the 43% ceiling for the $6,000 request."
    assert not verify_numerical_grounding(answer, ledger).valid
    assert verify_numerical_grounding(
        answer, ledger, query="Can she get $6,000?", reference_values={"max_dti_ratio": 0.43}
    ).valid


def test_a_genuine_refusal_passes():
    refusal = "The knowledge base does not contain sufficient evidence to determine conversion."
    assert verify_numerical_grounding(refusal, EvidenceLedger()).valid


def test_a_refusal_phrase_does_not_exempt_fabricated_numbers():
    # Regression: any answer containing a refusal phrase skipped numerical verification.
    answer = "Revenue was $9,999,999, but the iOS breakdown is not available."
    assert not verify_numerical_grounding(answer, _ledger({"revenue": 5000.0})).valid


def test_numbers_in_field_names_are_grounded_as_plain_numbers():
    # Found in a real run: "no delinquencies in the last 90 days" was rejected
    # although the evidence field is literally `delinquencies_90d`.
    ledger = _ledger({"account_id": "ACCT-101", "credit_history": {"delinquencies_90d": 0}})
    assert verify_numerical_grounding("There were no delinquencies in the last 90 days.", ledger).valid
    assert not verify_numerical_grounding("Utilization is 90%.", ledger).valid  # never a percentage


def test_request_text_does_not_ground_numbers():
    # "LIMIT 50" in the SQL a tool echoes back is not a fact about the data.
    ledger = _ledger({"query": "SELECT * FROM t LIMIT 50", "rows": [{"n": 7}]})
    assert not verify_numerical_grounding("There were 50 orders.", ledger).valid


def test_entity_swap_is_rejected():
    ledger = EvidenceLedger()
    ledger.add(
        source=EvidenceSource.ANALYTICS_SQL,
        query_or_input="SELECT ...",
        data={
            "rows": [
                {"device": "desktop", "conversion_rate": 0.0408},
                {"device": "mobile", "conversion_rate": 0.028},
            ]
        },
        provenance="db",
        summary="device conversion rates",
    )
    # Claiming desktop's rate for mobile must fail
    assert not verify_numerical_grounding("Mobile conversion was 4.1%.", ledger).valid
    # Claiming mobile's own rate passes
    assert verify_numerical_grounding("Mobile conversion was 2.8%.", ledger).valid


def test_version_numbers_are_not_treated_as_claims():
    ledger = EvidenceLedger()
    ledger.add(
        source=EvidenceSource.ANALYTICS_SQL,
        query_or_input="SELECT ...",
        data={"rows": [{"device": "mobile", "conversion_rate": 0.016}]},
        provenance="db",
        summary="",
    )
    # "iOS 17" should not be treated as an unsubstantiated quantitative claim
    result = verify_numerical_grounding("Mobile conversion dropped to 1.6% because of the iOS 17 update.", ledger)
    assert result.valid, f"Issues: {result.issues}"


def test_metric_mismatch_is_rejected():
    ledger = EvidenceLedger()
    ledger.add(
        source=EvidenceSource.RECORD_LOOKUP,
        query_or_input="ACCT-101",
        data={"account_id": "ACCT-101", "credit_score": 740, "dti_ratio": 0.28},
        provenance="db",
        summary="",
    )
    # Claiming DTI percentage as credit score must fail
    assert not verify_numerical_grounding("Credit score was 28%.", ledger).valid
    assert verify_numerical_grounding("DTI ratio was 28%.", ledger).valid


def test_audit_section_0_2_probes_are_rejected():
    from scripts.measure_verifier import build_ledger
    ledger = build_ledger()
    # 1. Entity swap: claiming mobile's rate (1.6%) for desktop
    assert not verify_numerical_grounding("Desktop users converted at just 1.6% last week.", ledger).valid
    # 2. Synonym swap: phone is mobile, but desktop converted at 4.1%
    assert not verify_numerical_grounding("The phone checkout converted at 4.1%.", ledger).valid
    # 3. Metric swap: 2,700 is checkouts started, not purchases completed
    assert not verify_numerical_grounding("Mobile purchases completed totalled 2,700 on one day.", ledger).valid


def test_correct_answers_remain_fully_accepted():
    from scripts.measure_verifier import build_ledger
    ledger = build_ledger()
    correct = [
        "Mobile conversion dropped from 2.8% to 1.6% last week.",
        "Variant B converted at 2.9% vs 1.6% for A, a 1.33 pp improvement (p = 0.014).",
        "Mobile checkouts started totalled 2,700 on one day.",
    ]
    for ans in correct:
        res = verify_numerical_grounding(ans, ledger)
        assert res.valid, f"Failed on: '{ans}', issues: {res.issues}"


def test_heldout_adversarial_probes_sets_1_and_2_rejected():
    from scripts.measure_verifier import (
        build_ledger,
        heldout_set1_wrong_claims,
        heldout_set2_wrong_claims,
    )
    ledger = build_ledger()
    for probe in heldout_set1_wrong_claims() + heldout_set2_wrong_claims():
        result = verify_numerical_grounding(probe, ledger)
        assert not result.valid, f"Expected held-out probe to be rejected: '{probe}'"


def test_heldout_correct_paraphrases_sets_1_and_2_accepted():
    from scripts.measure_verifier import (
        build_ledger,
        heldout_set1_right_claims,
        heldout_set2_right_claims,
    )
    ledger = build_ledger()
    for probe in heldout_set1_right_claims() + heldout_set2_right_claims():
        result = verify_numerical_grounding(probe, ledger)
        assert result.valid, f"Expected correct paraphrase to be accepted: '{probe}', issues: {result.issues}"



