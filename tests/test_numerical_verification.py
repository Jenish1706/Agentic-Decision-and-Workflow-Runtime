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
