"""Adaptive and Fixed Budget Controller.

Regulates exploration across heterogeneous tools (RAG retrievals, SQL queries,
vision calls, agent steps) based on task complexity.
Supports three evaluation regimes:
1. UNCONSTRAINED: No proactive caps; bounded only by global safety limits.
2. FIXED: Uniform resource caps across all queries.
3. ADAPTIVE: Complexity-dependent resource allocation.
"""

from dataclasses import dataclass, field
from enum import Enum
import re


class TaskComplexity(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    STRUCTURED_ANALYTIC = "structured_analytic"
    MULTI_HOP_ANALYSIS = "multi_hop_analysis"
    INVESTIGATION = "investigation"


@dataclass(frozen=True)
class ActionBudget:
    max_steps: int = 8
    max_retrievals: int = 2
    max_sql_queries: int = 4
    max_vision_calls: int = 0
    allow_decision: bool = True

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "max_steps": self.max_steps,
            "max_retrievals": self.max_retrievals,
            "max_sql_queries": self.max_sql_queries,
            "max_vision_calls": self.max_vision_calls,
            "allow_decision": self.allow_decision,
        }


@dataclass
class BudgetUsage:
    steps_used: int = 0
    retrievals_used: int = 0
    sql_queries_used: int = 0
    vision_calls_used: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "steps_used": self.steps_used,
            "retrievals_used": self.retrievals_used,
            "sql_queries_used": self.sql_queries_used,
            "vision_calls_used": self.vision_calls_used,
        }


class BudgetController:
    """Tracks resource consumption against an allocated ActionBudget."""

    def __init__(self, budget: ActionBudget | None = None):
        self.budget = budget or ActionBudget()
        self.usage = BudgetUsage()

    def can_step(self) -> bool:
        return self.usage.steps_used < self.budget.max_steps

    def can_retrieve(self) -> bool:
        return self.usage.retrievals_used < self.budget.max_retrievals

    def can_query_sql(self) -> bool:
        return self.usage.sql_queries_used < self.budget.max_sql_queries

    def can_call_vision(self) -> bool:
        return self.usage.vision_calls_used < self.budget.max_vision_calls

    def can_propose_decision(self) -> bool:
        return self.budget.allow_decision

    def record_step(self) -> None:
        self.usage.steps_used += 1

    def record_retrieval(self) -> None:
        self.usage.retrievals_used += 1

    def record_sql_query(self) -> None:
        self.usage.sql_queries_used += 1

    def record_vision_call(self) -> None:
        self.usage.vision_calls_used += 1


def classify_task_complexity(query: str) -> TaskComplexity:
    """Deterministic, rule-based task complexity classifier.

    Observable and predictable; does not rely on opaque LLM calls for budgeting.
    """
    q = query.lower()

    # Investigation: anomaly, why, root cause, recommendation, policy check
    if any(k in q for k in ("why", "driver", "root cause", "investigate", "recommend", "should we", "action")):
        return TaskComplexity.INVESTIGATION

    # Multi-hop: experiment comparison, statistical test, correlation
    if any(k in q for k in ("experiment", "exp-", "variant", "significant", "statistically", "compare", "versus", "vs")):
        return TaskComplexity.MULTI_HOP_ANALYSIS

    # Structured analytic: aggregation, metrics, funnel, segmentation
    if any(k in q for k in ("conversion", "rate", "count", "average", "avg", "segment", "funnel", "breakdown", "by device", "last week", "drop")):
        return TaskComplexity.STRUCTURED_ANALYTIC

    # Factual lookup: simple docs, single record lookup, direct FAQ
    return TaskComplexity.FACTUAL_LOOKUP


def allocate_budget(complexity: TaskComplexity, mode: str = "adaptive") -> ActionBudget:
    """Allocates ActionBudget based on complexity and allocation mode.

    Modes:
    - 'unconstrained': High safety boundaries only.
    - 'fixed': Static limits across all complexity levels.
    - 'adaptive': Dynamic limits tuned to task complexity.
    """
    m = mode.lower()
    if m == "unconstrained":
        return ActionBudget(
            max_steps=12,
            max_retrievals=8,
            max_sql_queries=8,
            max_vision_calls=2,
            allow_decision=True,
        )

    if m == "fixed":
        return ActionBudget(
            max_steps=6,
            max_retrievals=2,
            max_sql_queries=3,
            max_vision_calls=0,
            allow_decision=True,
        )

    # Adaptive mode
    if complexity == TaskComplexity.FACTUAL_LOOKUP:
        return ActionBudget(
            max_steps=4,
            max_retrievals=1,
            max_sql_queries=1,
            max_vision_calls=0,
            allow_decision=False,
        )
    elif complexity == TaskComplexity.STRUCTURED_ANALYTIC:
        return ActionBudget(
            max_steps=5,
            max_retrievals=0,
            max_sql_queries=3,
            max_vision_calls=0,
            allow_decision=False,
        )
    elif complexity == TaskComplexity.MULTI_HOP_ANALYSIS:
        return ActionBudget(
            max_steps=6,
            max_retrievals=1,
            max_sql_queries=4,
            max_vision_calls=0,
            allow_decision=True,
        )
    else:  # INVESTIGATION
        return ActionBudget(
            max_steps=8,
            max_retrievals=2,
            max_sql_queries=5,
            max_vision_calls=0,
            allow_decision=True,
        )
