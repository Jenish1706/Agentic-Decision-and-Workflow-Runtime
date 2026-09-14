"""Product analytics domain through the real agent loop:

    question -> SQL / significance test -> evidence ledger -> numerical
    verification -> rollout policy -> audit trace

Real server functions and the domain's own policies; only the LLM is scripted.
"""

import pytest

from agent_runtime.agent.actions import ActionDecision, ActionType
from agent_runtime.agent.budget import ActionBudget
from agent_runtime.agent.evidence import EvidenceKind, EvidenceLedger, EvidenceSource
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.policy import DecisionContext, PolicyStatus, ProposedDecision
from agent_runtime.agent.state import TerminationReason
from agent_runtime.agent.tools.base import ToolCategory
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.domains import load_domain
from domains.product_analytics.server import describe_tables, execute_sql
from domains.product_analytics.server import test_significance as significance_test

from .fakes import FakeLLMClient, FunctionTool


def _engine():
    return load_domain("product_analytics").create_policy_engine()


def _tools() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(FunctionTool(execute_sql, ToolCategory.SQL))
    tools.register(FunctionTool(describe_tables, ToolCategory.RECORD))
    tools.register(FunctionTool(significance_test, ToolCategory.STATISTICS))
    return tools


def _call(tool_name: str, **arguments) -> ActionDecision:
    return ActionDecision(thought=f"Call {tool_name}", action=ActionType.CALL_TOOL, tool_name=tool_name, tool_arguments=arguments)


def _rollout(answer: str, **parameters) -> ActionDecision:
    return ActionDecision(
        thought="Propose the rollout",
        action=ActionType.FINAL,
        answer=answer,
        proposed_decision=ProposedDecision(
            intent="Roll out the EXP-204 checkout redesign",
            recommended_action="rollout_variant_b",
            parameters=parameters,
            reasoning="Variant B outperformed control.",
            confidence=0.9,
        ),
    )


def _statuses(evaluation) -> dict[str, PolicyStatus]:
    return {result.rule_name: result.effective_status for result in evaluation.rule_results}


@pytest.mark.asyncio
async def test_full_investigation_escalates_a_full_traffic_rollout():
    runtime = AgentRuntime(
        llm=FakeLLMClient([
            _call("execute_sql", query="SELECT funnel_step, mobile_dropoff_rate FROM funnel_dropoff ORDER BY step_order"),
            _call("test_significance", experiment_id="EXP-204"),
            _rollout(
                "Mobile payment dropoff reached 48.0%. EXP-204 lifted conversion from 1.6% to 2.9% "
                "(p = 0.014). I recommend rolling out the redesign.",
                experiment_id="EXP-204",
                traffic_percentage=100,
            ),
        ]),
        tools=_tools(),
        policy_engine=_engine(),
        max_steps=5,
    )
    state = await runtime.run("Why did mobile conversion drop and what should we do?")

    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert state.answer_verified is True
    sql_item, stat_item = state.evidence_ledger.all()
    assert (sql_item.source, sql_item.kind) == (EvidenceSource.ANALYTICS_SQL, EvidenceKind.OBSERVATION)
    assert (stat_item.source, stat_item.kind) == (EvidenceSource.STATISTICS, EvidenceKind.CALCULATION)
    assert stat_item.data["treatment_better"] is True

    statuses = _statuses(state.policy_evaluation)
    assert statuses["statistical_significance_gate"] == PolicyStatus.ALLOWED
    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert "exceeds autonomous threshold (50%)" in state.policy_evaluation.audit_notes

    trace = state.audit_trace
    assert trace["steps_count"] == 3 and trace["evidence_count"] == 2
    assert trace["answer_verified"] is True
    assert trace["proposed_decision"]["parameters"]["traffic_percentage"] == 100
    assert trace["policy_evaluation"]["final_status"] == "requires_approval"


@pytest.mark.asyncio
async def test_significant_rollout_within_traffic_limit_is_allowed():
    runtime = AgentRuntime(
        llm=FakeLLMClient([
            _call("test_significance", experiment_id="EXP-204"),
            _rollout("EXP-204 is significant; roll out to 25% of traffic.", experiment_id="EXP-204", traffic_percentage=25),
        ]),
        tools=_tools(),
        policy_engine=_engine(),
        max_steps=4,
    )
    state = await runtime.run("Should we roll out EXP-204 to 25% of traffic?")

    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED


@pytest.mark.asyncio
async def test_fabricated_metric_is_rejected_then_corrected():
    runtime = AgentRuntime(
        llm=FakeLLMClient([
            _call("execute_sql", query="SELECT mobile_dropoff_rate FROM funnel_dropoff WHERE step_order = 3"),
            ActionDecision(thought="guess", action=ActionType.FINAL, answer="Mobile payment dropoff reached 92.5%."),
            ActionDecision(thought="fix", action=ActionType.FINAL, answer="Mobile payment dropoff reached 48.0%."),
        ]),
        tools=_tools(),
        max_steps=5,
    )
    state = await runtime.run("What was the mobile payment dropoff rate?")

    assert state.steps[1].status.value == "error"
    assert "Verification failed" in state.steps[1].observation
    assert state.final_answer == "Mobile payment dropoff reached 48.0%."
    assert state.answer_verified is True


@pytest.mark.asyncio
async def test_sql_budget_is_enforced_and_reported():
    runtime = AgentRuntime(
        llm=FakeLLMClient([
            _call("execute_sql", query="SELECT COUNT(*) AS n FROM daily_checkout_metrics"),
            _call("execute_sql", query="SELECT * FROM experiments"),
            ActionDecision(thought="answer", action=ActionType.FINAL, answer="Answered from the single allowed query."),
        ]),
        tools=_tools(),
        max_steps=5,
    )
    state = await runtime.run("Investigate metrics", budget=ActionBudget(max_steps=5, max_sql_queries=1))

    assert state.steps[1].status.value == "error"
    assert "SQL query budget exhausted" in state.steps[1].observation
    assert state.budget_usage["sql_queries_used"] == 1
    assert state.audit_trace["budget"]["max_sql_queries"] == 1


@pytest.mark.asyncio
async def test_budget_consumption_does_not_leak_into_the_next_run():
    # Regression: one BudgetController used to be shared by every run, so a
    # second question started with the first question's queries already spent.
    decisions = []
    for _ in range(2):
        decisions += [
            _call("execute_sql", query="SELECT COUNT(*) AS n FROM experiments"),
            ActionDecision(thought="answer", action=ActionType.FINAL, answer="Done."),
        ]
    runtime = AgentRuntime(llm=FakeLLMClient(decisions), tools=_tools(), budget=ActionBudget(max_steps=3, max_sql_queries=1))

    first = await runtime.run("q1")
    second = await runtime.run("q2")
    assert first.steps[0].status.value == "ok"
    assert second.steps[0].status.value == "ok"
    assert second.budget_usage["sql_queries_used"] == 1


# ---------------------------------------------------------------- policy regressions


def _ledger(*payloads: dict) -> EvidenceLedger:
    ledger = EvidenceLedger()
    for payload in payloads:
        ledger.add(source=EvidenceSource.STATISTICS, query_or_input="{}", data=payload, provenance="test", summary="")
    return ledger


def _evaluate(ledger: EvidenceLedger, **parameters):
    proposal = ProposedDecision(
        intent="rollout", recommended_action="rollout_variant_b", parameters=parameters, reasoning="r", confidence=0.8
    )
    return _engine().evaluate(proposal, DecisionContext(subject_or_entity="t", evidence_ledger=ledger))


def test_significance_of_a_different_experiment_does_not_count():
    evaluation = _evaluate(
        _ledger({"experiment_id": "EXP-100", "p_value": 0.001, "difference_pp": 2.0}),
        experiment_id="EXP-204",
        traffic_percentage=10,
    )
    assert _statuses(evaluation)["statistical_significance_gate"] == PolicyStatus.REQUIRES_APPROVAL


def test_significantly_worse_treatment_is_blocked():
    # p < alpha, but in the wrong direction: the old gate allowed this rollout.
    evaluation = _evaluate(
        _ledger({"experiment_id": "EXP-204", "p_value": 0.01, "difference_pp": -1.3, "significant": True}),
        experiment_id="EXP-204",
        traffic_percentage=10,
    )
    assert evaluation.final_status == PolicyStatus.BLOCKED


def test_policy_uses_its_own_alpha_not_the_tools_flag():
    # A model that calls test_significance with alpha=0.5 gets significant=True for p=0.2.
    evaluation = _evaluate(
        _ledger({"experiment_id": "EXP-204", "p_value": 0.2, "difference_pp": 1.0, "alpha": 0.5, "significant": True}),
        experiment_id="EXP-204",
        traffic_percentage=10,
    )
    assert evaluation.final_status == PolicyStatus.BLOCKED


def test_traffic_percentage_is_required_and_parsed():
    significant = {"experiment_id": "EXP-204", "p_value": 0.01, "difference_pp": 1.3}
    missing = _evaluate(_ledger(significant), experiment_id="EXP-204")
    assert _statuses(missing)["high_impact_rollout_approval"] == PolicyStatus.REQUIRES_APPROVAL

    as_text = _evaluate(_ledger(significant), experiment_id="EXP-204", traffic_percentage="100%")
    assert _statuses(as_text)["high_impact_rollout_approval"] == PolicyStatus.REQUIRES_APPROVAL
    assert not any("raised error" in result.message for result in as_text.rule_results)
