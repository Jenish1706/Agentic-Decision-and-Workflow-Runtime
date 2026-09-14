from agent_runtime.agent.state import AgentState, TerminationReason
from agent_runtime.evaluation.agent_metrics import (
    count_unnecessary_tool_calls,
    score_expected_numbers,
    score_tool_selection,
    single_agent_failed,
)
from agent_runtime.evaluation.rag_metrics import RetrievalScore, rank_metrics, score_citation_correctness


def test_expected_numbers_match_at_the_coarser_precision():
    answer = "ACCT-101 has a $5,000 limit, a score of 740 and a DTI of 28%; p = 0.014, lift 2.93%."
    assert score_expected_numbers([5000.0, 740, 28.0], answer) == 1.0
    assert score_expected_numbers([2.9], answer) == 1.0  # "2.93%" read at the benchmark's 1 decimal
    assert score_expected_numbers([0.015], answer) == 0.0  # 0.014 is a different 3-decimal value
    assert score_expected_numbers([740, 999], answer) == 0.5


def test_expected_numbers_ignore_digits_inside_identifiers():
    assert score_expected_numbers([101], "Account ACCT-101 in Q3.") == 0.0
    assert score_expected_numbers([2026], "Invoice INV-2026-004.") == 0.0


def test_expected_numbers_none_without_expectations():
    assert score_expected_numbers([], "Anything at all: 42.") is None


def test_unverified_best_effort_answer_counts_as_a_failure():
    state = AgentState(query="q")
    state.termination_reason = TerminationReason.FINAL_ANSWER
    state.answer_verified = True
    assert not single_agent_failed(state)

    state.answer_verified = False  # accepted on the last step without passing verification
    assert single_agent_failed(state)

    state.termination_reason = TerminationReason.MAX_STEPS_REACHED
    state.answer_verified = None
    assert single_agent_failed(state)


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


def test_rank_metrics():
    assert rank_metrics(["x", "doc-b", "doc-a"], ["doc-a", "doc-b"]) == RetrievalScore(recall_at_k=1.0, reciprocal_rank=0.5)
    assert rank_metrics(["x", "y"], ["doc-a"]) == RetrievalScore(recall_at_k=0.0, reciprocal_rank=0.0)
    assert rank_metrics(["doc-a"], []) == RetrievalScore(recall_at_k=None, reciprocal_rank=None)
