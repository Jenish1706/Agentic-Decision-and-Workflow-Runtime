import pytest

from agent_runtime.agent.evidence import EvidenceItem, EvidenceKind, EvidenceLedger, EvidenceSource


def test_evidence_item_creation_and_serialization():
    item = EvidenceItem(
        evidence_id="ev-sql-001",
        source=EvidenceSource.ANALYTICS_SQL,
        kind=EvidenceKind.OBSERVATION,
        query_or_input="SELECT COUNT(*) FROM users",
        data={"user_count": 1050},
        provenance="users_table",
        summary="Total user count is 1050",
    )
    assert item.evidence_id == "ev-sql-001"
    assert item.source == EvidenceSource.ANALYTICS_SQL
    assert item.kind == EvidenceKind.OBSERVATION
    assert item.data["user_count"] == 1050

    d = item.to_dict()
    assert d["evidence_id"] == "ev-sql-001"
    assert d["data"] == {"user_count": 1050}

    reconstructed = EvidenceItem.from_dict(d)
    assert reconstructed.evidence_id == item.evidence_id
    assert reconstructed.data == item.data


def test_ledger_append_only_and_unique_ids():
    ledger = EvidenceLedger(run_id="run-test-123")
    assert ledger.run_id == "run-test-123"
    assert len(ledger) == 0

    item1 = ledger.add(
        source=EvidenceSource.ANALYTICS_SQL,
        query_or_input="SELECT conversion FROM mobile",
        data={"mobile_conversion": 0.184},
        provenance="events",
        summary="Mobile conversion is 18.4%",
    )
    assert item1.evidence_id == "ev-sql-001"
    assert len(ledger) == 1

    # Attempting to duplicate ID should raise ValueError
    with pytest.raises(ValueError, match="already exists"):
        ledger.add(
            source=EvidenceSource.ANALYTICS_SQL,
            query_or_input="SELECT 1",
            data={},
            provenance="",
            summary="",
            evidence_id="ev-sql-001",
        )


def test_ledger_get_and_find_by_source_and_kind():
    ledger = EvidenceLedger()
    it1 = ledger.add(
        source=EvidenceSource.ANALYTICS_SQL,
        query_or_input="SELECT ...",
        data={"metric": 10},
        provenance="db",
        summary="SQL metric",
    )
    it2 = ledger.add(
        source=EvidenceSource.KNOWLEDGE_RAG,
        query_or_input="search doc",
        data={"doc": "policy"},
        provenance="doc_1",
        summary="Policy document",
        kind=EvidenceKind.RETRIEVAL,
    )

    assert ledger.get_by_id(it1.evidence_id) == it1
    assert ledger.get_by_id(it2.evidence_id) == it2
    assert ledger.get_by_id("non-existent") is None

    assert ledger.find_by_source(EvidenceSource.ANALYTICS_SQL) == [it1]
    assert ledger.find_by_source(EvidenceSource.KNOWLEDGE_RAG) == [it2]
    assert ledger.find_by_kind(EvidenceKind.RETRIEVAL) == [it2]


def test_derived_evidence_and_parents():
    ledger = EvidenceLedger()
    e1 = ledger.add(
        source=EvidenceSource.ANALYTICS_SQL,
        query_or_input="q1",
        data={"desktop_conv": 0.421},
        provenance="events",
        summary="Desktop conversion is 42.1%",
    )
    e2 = ledger.add(
        source=EvidenceSource.ANALYTICS_SQL,
        query_or_input="q2",
        data={"mobile_conv": 0.184},
        provenance="events",
        summary="Mobile conversion is 18.4%",
    )

    # Derived calculation referencing parents
    e3 = ledger.add(
        source=EvidenceSource.STATISTICS,
        query_or_input="desktop_conv - mobile_conv",
        data={"conv_gap": 0.237},
        provenance="calculated",
        summary="Desktop to mobile conversion gap is 23.7 percentage points",
        kind=EvidenceKind.CALCULATION,
        parent_evidence_ids=[e1.evidence_id, e2.evidence_id],
    )
    assert e3.parent_evidence_ids == (e1.evidence_id, e2.evidence_id)

    # Adding derived evidence with non-existent parent should fail
    with pytest.raises(ValueError, match="does not exist in ledger"):
        ledger.add(
            source=EvidenceSource.STATISTICS,
            query_or_input="calc",
            data={},
            provenance="",
            summary="",
            parent_evidence_ids=["ev-fake-999"],
        )


def test_evidence_cannot_be_rewritten_after_it_is_recorded():
    # Regression: items were `frozen=True` but kept a plain dict inside, so
    # `item.data["p_value"] = 0.9` silently succeeded.
    payload = {"p_value": 0.01, "rows": [{"n": 1}]}
    ledger = EvidenceLedger()
    item = ledger.add(source=EvidenceSource.STATISTICS, query_or_input="q", data=payload, provenance="t", summary="")

    with pytest.raises(TypeError):
        item.data["p_value"] = 0.9  # type: ignore[index]
    with pytest.raises(AttributeError):
        item.data["rows"].append({"n": 2})  # lists are frozen into tuples

    payload["p_value"] = 0.9  # the tool mutating its own result afterwards
    assert ledger.all()[0].data["p_value"] == 0.01

    exported = item.to_dict()["data"]
    exported["p_value"] = 0.9  # exports are copies
    assert item.data["p_value"] == 0.01


def test_automatic_ids_skip_ids_supplied_by_callers():
    # Regression: automatic ids were recounted from the item list and could
    # collide with a caller-supplied id, raising "already exists" mid-run.
    ledger = EvidenceLedger()
    ledger.add(source=EvidenceSource.ANALYTICS_SQL, query_or_input="q", data={}, provenance="", summary="", evidence_id="ev-sql-002")
    first = ledger.add(source=EvidenceSource.ANALYTICS_SQL, query_or_input="q", data={}, provenance="", summary="")
    second = ledger.add(source=EvidenceSource.ANALYTICS_SQL, query_or_input="q", data={}, provenance="", summary="")
    assert (first.evidence_id, second.evidence_id) == ("ev-sql-001", "ev-sql-003")


def test_extract_numerical_facts_skips_booleans():
    ledger = EvidenceLedger()
    ledger.add(source=EvidenceSource.STATISTICS, query_or_input="q", data={"significant": True, "p": 0.01}, provenance="", summary="")
    assert ledger.extract_numerical_facts() == {"ev-stat-001.p": 0.01}


def test_extract_numerical_facts():
    ledger = EvidenceLedger()
    ledger.add(
        source=EvidenceSource.ANALYTICS_SQL,
        query_or_input="q1",
        data={"a": 10.5, "b": 20, "nested": {"rate": 0.05}},
        provenance="db",
        summary="",
    )
    facts = ledger.extract_numerical_facts()
    assert facts["ev-sql-001.a"] == 10.5
    assert facts["ev-sql-001.b"] == 20.0
    assert facts["ev-sql-001.nested.rate"] == 0.05
