from agent_runtime.evaluation.retrieval_metrics import score_search_efficiency


def test_no_retrieval_reports_zero_and_undefined_redundancy():
    e = score_search_efficiency([], [])
    assert e.retrieval_calls == 0
    assert e.retrieval_chunks_consumed == 0
    # None, not 0.0 - a run that never searched has no redundancy to report,
    # and scoring it 0.0 would flatter the average as "perfectly efficient".
    assert e.retrieval_redundancy is None


def test_single_retrieval_has_zero_redundancy():
    e = score_search_efficiency(["aurora rate limit"], ["a::0", "b::0", "c::0"])
    assert e.retrieval_calls == 1
    assert e.unique_retrieval_queries == 1
    assert e.retrieval_chunks_consumed == 3
    assert e.retrieval_redundancy == 0.0
    assert e.repeated_query_calls == 0


def test_fully_redundant_second_search_scores_half():
    # Same query twice, identical chunks back: 6 consumed, 3 unique.
    e = score_search_efficiency(
        ["aurora rate limit", "aurora rate limit"],
        ["a::0", "b::0", "c::0", "a::0", "b::0", "c::0"],
    )
    assert e.retrieval_calls == 2
    assert e.unique_retrieval_queries == 1
    assert e.repeated_query_calls == 1
    assert e.retrieval_redundancy == 0.5


def test_query_normalization_treats_trivial_variants_as_repeats():
    e = score_search_efficiency(["Aurora rate limit?", "  aurora rate limit  "], ["a::0", "a::0"])
    assert e.unique_retrieval_queries == 1
    assert e.repeated_query_calls == 1


def test_paraphrases_are_deliberately_counted_as_distinct():
    # Shallow normalization only - this UNDERcounts redundancy rather than
    # overcounting it, which is the safer direction (documented in the module).
    e = score_search_efficiency(["cascade escalation", "cascade escalation process"], ["a::0", "a::0"])
    assert e.unique_retrieval_queries == 2
    assert e.repeated_query_calls == 0
    # chunk-level redundancy still catches it
    assert e.retrieval_redundancy == 0.5


def test_partial_chunk_overlap():
    e = score_search_efficiency(["q1", "q2"], ["a::0", "b::0", "b::0", "c::0"])
    assert e.retrieval_chunks_consumed == 4
    assert e.unique_chunks_consumed == 3
    assert e.retrieval_redundancy == 0.25
