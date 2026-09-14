import dataclasses

import pytest

from agent_runtime.agent.budget import (
    ActionBudget,
    BudgetController,
    BudgetMode,
    TaskComplexity,
    allocate_budget,
    classify_task_complexity,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is Beacon's uptime SLO?", TaskComplexity.FACTUAL_LOOKUP),
        ("What was the mobile checkout conversion rate last week?", TaskComplexity.STRUCTURED_ANALYTIC),
        ("Did variant B in experiment EXP-204 statistically outperform variant A?", TaskComplexity.MULTI_HOP_ANALYSIS),
        ("Why did conversion decline and what should the product team recommend?", TaskComplexity.INVESTIGATION),
    ],
)
def test_classify_task_complexity(question, expected):
    assert classify_task_complexity(question) == expected


def test_keywords_match_whole_words_only():
    # Regression: raw substring matching read "count" inside "account" and
    # "rate" inside "generate", so these were classified as analytics tasks.
    assert classify_task_complexity("What is the current limit on account ACCT-101?") == TaskComplexity.FACTUAL_LOOKUP
    assert classify_task_complexity("Generate the vendor list.") == TaskComplexity.FACTUAL_LOOKUP


def test_allocate_budget_per_mode():
    factual = allocate_budget(TaskComplexity.FACTUAL_LOOKUP, mode="adaptive")
    assert (factual.max_retrievals, factual.max_sql_queries) == (1, 1)

    investigation = allocate_budget(TaskComplexity.INVESTIGATION, mode=BudgetMode.ADAPTIVE)
    assert (investigation.max_steps, investigation.max_sql_queries) == (8, 5)

    # Fixed and unconstrained ignore complexity.
    assert allocate_budget(TaskComplexity.FACTUAL_LOOKUP, mode="fixed") == allocate_budget(
        TaskComplexity.INVESTIGATION, mode="fixed"
    )
    assert allocate_budget(TaskComplexity.FACTUAL_LOOKUP, mode=" Unconstrained ").max_steps == 12


def test_unknown_mode_raises_instead_of_running_as_adaptive():
    with pytest.raises(ValueError):
        allocate_budget(TaskComplexity.FACTUAL_LOOKUP, mode="adaptiv")


def test_budget_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ActionBudget().max_steps = 99  # type: ignore[misc]


def test_controller_tracks_and_exhausts_each_resource():
    controller = BudgetController(ActionBudget(max_steps=2, max_retrievals=1, max_sql_queries=1, max_vision_calls=0))
    assert controller.can_step() and controller.can_retrieve() and controller.can_query_sql()
    assert not controller.can_call_vision()  # 0 means not allowed at all

    controller.record_step()
    controller.record_retrieval()
    assert not controller.can_retrieve()
    assert controller.can_query_sql()

    controller.record_step()
    assert not controller.can_step()
    assert controller.usage.to_dict() == {
        "steps_used": 2,
        "retrievals_used": 1,
        "sql_queries_used": 0,
        "vision_calls_used": 0,
    }
