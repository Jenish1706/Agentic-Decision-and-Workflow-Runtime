"""Action budgets: how many steps, searches, SQL queries and extraction calls
one agent run may spend.

WHAT
    `ActionBudget` is an immutable allocation. `BudgetController` counts
    consumption against it for exactly ONE run. `classify_task_complexity`
    and `allocate_budget` pick an allocation for a question.

WHY
    This project's retrieval-budget experiment (EVALUATION.md) found that
    giving an agent more freedom to search made answers worse, not better:
    more redundant searches, more accumulated context, more chances to
    confabulate. Budgets turn that finding into a control.

THREE REGIMES (compared in the product analytics experiment)
    UNCONSTRAINED  generous safety ceilings only
    FIXED          the same moderate caps for every question
    ADAPTIVE       caps chosen from the question's classified complexity

BUGS FIXED IN THIS VERSION (kept here because they are instructive)
    - A single controller used to be shared across every run of an
      AgentRuntime and was never reset, so after enough total steps every
      later question started with its budget already spent. The loop now
      creates a fresh BudgetController per run (agent/loop.py).
    - The classifier matched keywords as raw substrings: "count" matched
      "account" (credit-risk questions got a zero-retrieval budget) and
      "action" matched "transaction". Matching is now on word boundaries.
    - The vision budget existed but was never enforced; it now is.
    - `allow_decision` was removed. It was never enforced, and "may this task
      produce a decision" is authorization, which the PolicyEngine governs -
      not a resource to be budgeted.
    - An unknown mode string silently fell through to "adaptive"; a typo in an
      experiment config would have mislabeled results. It now raises.

FAILURE MODE
    The classifier is a transparent keyword heuristic, chosen over an LLM call
    so budgeting stays deterministic and inspectable. It will misroute
    unusually phrased questions; the cost of that is a tighter or looser
    budget, never a policy bypass.
"""

import re
from dataclasses import dataclass
from enum import Enum


class TaskComplexity(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    STRUCTURED_ANALYTIC = "structured_analytic"
    MULTI_HOP_ANALYSIS = "multi_hop_analysis"
    INVESTIGATION = "investigation"


class BudgetMode(str, Enum):
    UNCONSTRAINED = "unconstrained"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class ActionBudget:
    """An immutable per-run allocation. 0 means "not allowed at all"."""

    max_steps: int = 8
    max_retrievals: int = 2
    max_sql_queries: int = 4
    max_vision_calls: int = 2

    def to_dict(self) -> dict[str, int]:
        return {
            "max_steps": self.max_steps,
            "max_retrievals": self.max_retrievals,
            "max_sql_queries": self.max_sql_queries,
            "max_vision_calls": self.max_vision_calls,
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
    """Counts one run's consumption against an ActionBudget.

    Create a new controller for every run - its `usage` is intentionally not
    resettable, so reusing one across runs is visibly wrong rather than subtly
    wrong.
    """

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

    def record_step(self) -> None:
        self.usage.steps_used += 1

    def record_retrieval(self) -> None:
        self.usage.retrievals_used += 1

    def record_sql_query(self) -> None:
        self.usage.sql_queries_used += 1

    def record_vision_call(self) -> None:
        self.usage.vision_calls_used += 1


def _phrases(*phrases: str) -> re.Pattern[str]:
    """Compile phrases into one case-insensitive, word-bounded pattern.

    `\\b` on both sides is the whole fix for the substring bug: "count" no
    longer fires inside "account", nor "rate" inside "generate".
    """
    return re.compile(r"\b(?:" + "|".join(phrases) + r")\b", re.IGNORECASE)


# Checked in this order; the first category that matches wins, so the more
# demanding categories come first.
_INVESTIGATION = _phrases(
    "why", "root cause", "drivers?", "investigat(?:e|ion|ing)", "recommend(?:ation|ed)?",
    "should we", "what should", "approv(?:e|al|ed)", "grant(?:ed)?", "authori[sz](?:e|ation|ed)",
    "evaluat(?:e|ion)", "audit", "eligib(?:le|ility)", "qualif(?:y|ies)", "remediat(?:e|ion)",
    "can we", "roll ?out", "decid(?:e|ing)", "decision",
)
_MULTI_HOP = _phrases(
    "experiments?", r"exp-\d+", "variants?", "significan(?:t|ce)", "statistically",
    "compar(?:e|ison|ing)", "versus", r"vs\.?", "a/b",
)
_STRUCTURED = _phrases(
    "conversion", "rates?", "count", "average", "avg", "segments?", "segmentation", "funnel",
    "breakdown", "by device", "last week", "drop(?:ped|-off|off)?", "total", "sum", "trend",
)


def classify_task_complexity(query: str) -> TaskComplexity:
    """Deterministic, rule-based complexity classifier (no LLM involved)."""
    if _INVESTIGATION.search(query):
        return TaskComplexity.INVESTIGATION
    if _MULTI_HOP.search(query):
        return TaskComplexity.MULTI_HOP_ANALYSIS
    if _STRUCTURED.search(query):
        return TaskComplexity.STRUCTURED_ANALYTIC
    return TaskComplexity.FACTUAL_LOOKUP


_ADAPTIVE_BUDGETS: dict[TaskComplexity, ActionBudget] = {
    # One lookup and an answer. Retrieval allowed once in case the fact lives
    # in documentation.
    TaskComplexity.FACTUAL_LOOKUP: ActionBudget(max_steps=4, max_retrievals=1, max_sql_queries=1, max_vision_calls=1),
    # Aggregations over structured data; documents are not expected to help,
    # so retrieval is disallowed outright (this is the experiment's hypothesis,
    # not an oversight).
    TaskComplexity.STRUCTURED_ANALYTIC: ActionBudget(max_steps=5, max_retrievals=0, max_sql_queries=3, max_vision_calls=1),
    TaskComplexity.MULTI_HOP_ANALYSIS: ActionBudget(max_steps=6, max_retrievals=1, max_sql_queries=4, max_vision_calls=2),
    TaskComplexity.INVESTIGATION: ActionBudget(max_steps=8, max_retrievals=2, max_sql_queries=5, max_vision_calls=3),
}


def allocate_budget(complexity: TaskComplexity, mode: BudgetMode | str = BudgetMode.ADAPTIVE) -> ActionBudget:
    """Choose an ActionBudget for one question.

    Raises ValueError for an unknown mode instead of guessing, so a mistyped
    experiment regime cannot silently run as a different one.
    """
    regime = mode if isinstance(mode, BudgetMode) else BudgetMode(mode.strip().lower())
    if regime is BudgetMode.UNCONSTRAINED:
        return ActionBudget(max_steps=12, max_retrievals=8, max_sql_queries=8, max_vision_calls=6)
    if regime is BudgetMode.FIXED:
        return ActionBudget(max_steps=6, max_retrievals=2, max_sql_queries=3, max_vision_calls=2)
    return _ADAPTIVE_BUDGETS[complexity]
