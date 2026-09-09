import pytest
from agent_runtime.agent.budget import (
    ActionBudget,
    BudgetController,
    TaskComplexity,
    allocate_budget,
    classify_task_complexity,
)


def test_classify_task_complexity():
    assert classify_task_complexity("What is Beacon's uptime SLO?") == TaskComplexity.FACTUAL_LOOKUP
    assert classify_task_complexity("What was the mobile checkout conversion rate last week?") == TaskComplexity.STRUCTURED_ANALYTIC
    assert classify_task_complexity("Did variant B in experiment EXP-204 statistically outperform variant A?") == TaskComplexity.MULTI_HOP_ANALYSIS
    assert classify_task_complexity("Why did conversion decline and what should the product team recommend?") == TaskComplexity.INVESTIGATION


def test_allocate_budget_modes():
    # Adaptive
    b_fact = allocate_budget(TaskComplexity.FACTUAL_LOOKUP, mode="adaptive")
    assert b_fact.max_retrievals == 1
    assert b_fact.max_sql_queries == 1
    assert not b_fact.allow_decision

    b_inv = allocate_budget(TaskComplexity.INVESTIGATION, mode="adaptive")
    assert b_inv.max_steps == 8
    assert b_inv.max_sql_queries == 5
    assert b_inv.allow_decision

    # Fixed
    b_fixed = allocate_budget(TaskComplexity.FACTUAL_LOOKUP, mode="fixed")
    assert b_fixed.max_steps == 6
    assert b_fixed.max_sql_queries == 3

    # Unconstrained
    b_uncon = allocate_budget(TaskComplexity.FACTUAL_LOOKUP, mode="unconstrained")
    assert b_uncon.max_steps == 12
    assert b_uncon.max_sql_queries == 8


def test_budget_controller_tracking_and_exhaustion():
    budget = ActionBudget(max_steps=2, max_retrievals=1, max_sql_queries=1)
    ctrl = BudgetController(budget)

    assert ctrl.can_step()
    assert ctrl.can_retrieve()
    assert ctrl.can_query_sql()

    ctrl.record_step()
    ctrl.record_retrieve = ctrl.record_retrieval()  # type: ignore
    assert not ctrl.can_retrieve()
    assert ctrl.can_step()
    assert ctrl.can_query_sql()

    ctrl.record_step()
    assert not ctrl.can_step()

    usage_dict = ctrl.usage.to_dict()
    assert usage_dict["steps_used"] == 2
    assert usage_dict["retrievals_used"] == 1
    assert usage_dict["sql_queries_used"] == 0
