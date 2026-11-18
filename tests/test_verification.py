from agent_runtime.agent.verification import verify_citations
from agent_runtime.rag.documents import Chunk


def _chunk(doc_id: str, text: str = "The rate limit is 5000 confirmed by policy") -> Chunk:
    return Chunk(chunk_id=f"{doc_id}::0", doc_id=doc_id, doc_title=doc_id, position=0, text=text)


def test_answer_citing_known_tags_is_grounded():
    citation_map = {"[1]": _chunk("doc-a"), "[2]": _chunk("doc-b")}
    result = verify_citations("The limit is 5000 [1], confirmed by policy [2].", citation_map)
    assert result.grounded
    assert result.issues == []


def test_answer_citing_unknown_tag_is_not_grounded():
    citation_map = {"[1]": _chunk("doc-a")}
    result = verify_citations("The limit is 5000 [3].", citation_map)
    assert not result.grounded
    assert "[3]" in result.issues[0]


def test_answer_with_no_citations_despite_evidence_is_flagged():
    citation_map = {"[1]": _chunk("doc-a")}
    result = verify_citations("The limit is 5000.", citation_map)
    assert not result.grounded


def test_refusal_with_no_citations_is_not_flagged():
    citation_map = {"[1]": _chunk("doc-a")}
    result = verify_citations("The knowledge base does not contain this information.", citation_map)
    assert result.grounded


def test_answer_with_no_evidence_and_no_citations_is_not_flagged():
    # nothing was retrieved at all; an unsupported-sounding answer here isn't
    # a citation problem specifically (there was nothing to cite)
    result = verify_citations("I don't know.", {})
    assert result.grounded


def test_unsupported_citation_is_flagged():
    aurora_chunk = Chunk(
        chunk_id="arch-aurora::0",
        doc_id="arch-aurora",
        doc_title="Aurora Architecture Overview",
        position=0,
        text="Aurora is Nimbus's event ingestion service. Aurora enforces a default rate limit of 5,000 events/s.",
    )
    citation_map = {"[1]": aurora_chunk}
    result = verify_citations("The Eiffel Tower was completed in 1889 in Paris, France [1].", citation_map)
    assert not result.grounded
    assert any("does not support" in issue for issue in result.issues)

