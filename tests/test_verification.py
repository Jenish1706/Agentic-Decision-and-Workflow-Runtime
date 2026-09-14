from agent_runtime.agent.verification import verify_citations
from agent_runtime.rag.documents import Chunk


def _chunk(doc_id: str) -> Chunk:
    return Chunk(chunk_id=f"{doc_id}::0", doc_id=doc_id, doc_title=doc_id, position=0, text="some text")


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
