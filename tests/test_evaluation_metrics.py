from agent_runtime.evaluation.agent_metrics import count_unnecessary_tool_calls, score_tool_selection
from agent_runtime.evaluation.rag_metrics import RetrievalScore, score_citation_correctness


def test_tool_selection_none_when_no_tool_expected():
    assert score_tool_selection(None, ["calculator"]) is None


def test_tool_selection_true_when_expected_tool_was_called():
    assert score_tool_selection("calculator", ["records_lookup", "calculator"]) is True


def test_tool_selection_false_when_expected_tool_was_not_called():
    assert score_tool_selection("calculator", ["records_lookup"]) is False


def test_unnecessary_tool_calls_counts_only_when_no_tool_was_expected():
    assert count_unnecessary_tool_calls(None, ["calculator", "calculator"]) == 2
    assert count_unnecessary_tool_calls("calculator", ["calculator"]) == 0


def test_citation_correctness_none_without_expected_docs_or_citations():
    assert score_citation_correctness(["[1]"], {"[1]": "doc-a"}, []) is None
    assert score_citation_correctness([], {}, ["doc-a"]) is None


def test_citation_correctness_scores_fraction_matching_expected_docs():
    citation_sources = {"[1]": "doc-a", "[2]": "doc-b"}
    score = score_citation_correctness(["[1]", "[2]"], citation_sources, ["doc-a"])
    assert score == 0.5


def test_citation_correctness_perfect_when_all_match():
    citation_sources = {"[1]": "doc-a"}
    score = score_citation_correctness(["[1]"], citation_sources, ["doc-a"])
    assert score == 1.0


def test_retrieval_score_is_a_plain_dataclass():
    score = RetrievalScore(recall_at_k=0.5, reciprocal_rank=1.0)
    assert score.recall_at_k == 0.5
