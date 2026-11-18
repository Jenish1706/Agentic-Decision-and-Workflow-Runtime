from agent_runtime.agent.evidence import EvidenceLedger, EvidenceSource
from agent_runtime.agent.facts import (
    FactStatus,
    as_bool,
    as_number,
    find_entity_records,
    is_governed_action,
    proposal_number,
    resolve_fact,
)


def _ledger(*payloads: dict, source: EvidenceSource = EvidenceSource.RECORD_LOOKUP) -> EvidenceLedger:
    ledger = EvidenceLedger()
    for payload in payloads:
        ledger.add(source=source, query_or_input="q", data=payload, provenance="t", summary="")
    return ledger


ROWS = {
    "rows": [
        {"account_id": "ACCT-101", "credit_score": 740},
        {"account_id": "ACCT-102", "credit_score": 610},
    ]
}


def test_values_are_scoped_to_the_requested_entity():
    fact = resolve_fact(_ledger(ROWS), "credit_score", entity_key="account_id", entity_value="acct-102 ")
    assert fact.status is FactStatus.FOUND and fact.value == 610
    assert fact.conclusive


def test_nested_records_inherit_the_enclosing_entity():
    profile = {"account_id": "ACCT-101", "credit_history": {"delinquencies_90d": 0}}
    fact = resolve_fact(_ledger(profile), "delinquencies_90d", entity_key="account_id", entity_value="ACCT-101")
    assert fact.found and fact.attributed


def test_other_entities_values_are_ignored_not_used_as_fallback():
    fact = resolve_fact(_ledger(ROWS), "credit_score", entity_key="account_id", entity_value="ACCT-103")
    assert fact.status is FactStatus.MISSING


def test_unattributed_values_are_a_fallback_that_is_not_conclusive():
    fact = resolve_fact(_ledger({"credit_score": 700}), "credit_score", entity_key="account_id", entity_value="ACCT-101")
    assert fact.found and not fact.attributed and not fact.conclusive


def test_attributed_values_win_over_unattributed_ones():
    fact = resolve_fact(_ledger({"credit_score": 500}, ROWS), "credit_score", entity_key="account_id", entity_value="ACCT-101")
    assert fact.value == 740 and fact.attributed


def test_disagreeing_values_are_a_conflict_with_all_candidates():
    ledger = _ledger({"account_id": "ACCT-101", "credit_score": 740}, {"account_id": "ACCT-101", "credit_score": 745})
    fact = resolve_fact(ledger, "credit_score", entity_key="account_id", entity_value="ACCT-101")
    assert fact.status is FactStatus.CONFLICT and fact.value is None
    assert set(fact.candidates) == {740, 745}
    assert "contradictory" in fact.describe()


def test_equal_values_in_different_representations_agree():
    ledger = _ledger({"account_id": "A", "tier": "Gold"}, {"account_id": "A", "tier": "gold "}, {"account_id": "A", "n": 1}, {"account_id": "A", "n": 1.0})
    assert resolve_fact(ledger, "tier", entity_key="account_id", entity_value="A").found
    assert resolve_fact(ledger, "n", entity_key="account_id", entity_value="A").found


def test_null_values_count_as_missing():
    fact = resolve_fact(_ledger({"account_id": "ACCT-105", "credit_score": None}), "credit_score", entity_key="account_id", entity_value="ACCT-105")
    assert fact.status is FactStatus.MISSING


def test_find_entity_records():
    listing = {"service_name": "svc", "incidents": [{"incident_id": "I-1", "severity": "P0"}, {"incident_id": "I-2", "severity": "P2"}]}
    records = find_entity_records(_ledger(listing), entity_key="service_name", entity_value="SVC", required_fields=("incident_id",))
    assert [record["incident_id"] for _, record in records] == ["I-1", "I-2"]


def test_as_bool_is_strict():
    # bool("false") is True - the bug that turned a "false" override into an authorized one.
    assert as_bool("false") is False and as_bool("TRUE") is True and as_bool(0) is False
    assert as_bool("maybe") is None and as_bool(2) is None and as_bool(None) is None


def test_as_number_is_strict():
    assert as_number("$5,000") == 5000.0 and as_number("48%") == 48.0 and as_number(3) == 3.0
    assert as_number(True) is None and as_number("n/a") is None and as_number(float("nan")) is None


def test_proposal_number_takes_the_first_parseable_key():
    assert proposal_number({"traffic_pct": "25%"}, "traffic_percentage", "traffic_pct") == 25.0
    assert proposal_number({"traffic_percentage": "all"}, "traffic_percentage") is None


def test_governed_actions_are_decided_conservatively():
    granting, exempt = frozenset({"approve", "increase"}), frozenset({"decline", "refer"})
    assert is_governed_action("approve_increase", governed=granting, exempt=exempt)
    assert is_governed_action("grant_limit_raise", governed=granting, exempt=exempt)  # unrecognised -> governed
    assert is_governed_action("decline_increase", governed=granting, exempt=exempt)  # mentions a governed word
    assert not is_governed_action("Refer-To-Underwriter", governed=granting, exempt=exempt)


def test_facts_from_free_form_sql_cannot_be_conclusive_to_permit():
    sql_ledger = EvidenceLedger()
    sql_ledger.add(
        source=EvidenceSource.ANALYTICS_SQL,
        query_or_input="SELECT 'ACCT-101' AS account_id, 740 AS credit_score",
        data={"rows": [{"account_id": "ACCT-101", "credit_score": 740}]},
        provenance="tool:execute_sql",
        summary="",
    )
    fact = resolve_fact(sql_ledger, "credit_score", entity_key="account_id", entity_value="ACCT-101")
    assert fact.found
    assert fact.attributed
    assert not fact.can_permit
    assert not fact.conclusive


def test_facts_from_record_lookup_can_be_conclusive_to_permit():
    rec_ledger = EvidenceLedger()
    rec_ledger.add(
        source=EvidenceSource.RECORD_LOOKUP,
        query_or_input="ACCT-101",
        data={"account_id": "ACCT-101", "credit_score": 740},
        provenance="tool:get_account_profile",
        summary="",
    )
    fact = resolve_fact(rec_ledger, "credit_score", entity_key="account_id", entity_value="ACCT-101")
    assert fact.found
    assert fact.attributed
    assert fact.can_permit
    assert fact.conclusive
