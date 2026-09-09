"""End-to-end integration tests for Product Analytics Decision & Workflow Runtime.

Exercises the full analytical pipeline:
Question -> SQL / MCP -> Statistical Significance -> Evidence Ledger ->
Deterministic Numerical Verification -> Policy Engine -> Audit Trace.
"""

import pytest
from pydantic import BaseModel

from agent_runtime.agent.actions import ActionDecision, ActionType
from agent_runtime.agent.budget import ActionBudget, BudgetController
from agent_runtime.agent.evidence import EvidenceKind, EvidenceSource
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.policy import (
    DecisionContext,
    PolicyEngine,
    PolicyRuleResult,
    PolicyStatus,
    ProposedDecision,
)
from agent_runtime.agent.state import TerminationReason
from agent_runtime.agent.tools.base import Tool, ToolResult
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.mcp_layer.servers.analytics_server import (
    describe_tables,
    execute_sql,
    test_significance as run_significance_test,
)


class _LocalSqlTool(Tool):
    name = "execute_sql"
    description = "Execute read-only SQL query."
    parameters_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    origin = "local"

    async def run(self, query: str) -> ToolResult:
        res = execute_sql(query)
        if not res.ok:
            return ToolResult(ok=False, content="", error=res.error)
        return ToolResult(
            ok=True,
            content=f"Returned {res.row_count} rows: {res.rows}",
            data={"columns": res.columns, "row_count": res.row_count, "rows": res.rows},
        )


class _LocalSignificanceTool(Tool):
    name = "test_significance"
    description = "Run 2-proportion hypothesis test on A/B experiment."
    parameters_schema = {
        "type": "object",
        "properties": {"experiment_id": {"type": "string"}, "alpha": {"type": "number"}},
        "required": ["experiment_id"],
    }
    origin = "local"

    async def run(self, experiment_id: str, alpha: float = 0.05) -> ToolResult:
        res = run_significance_test(experiment_id, alpha=alpha)
        if not res.ok:
            return ToolResult(ok=False, content="", error=res.error)
        data = {
            "experiment_id": res.experiment_id,
            "sample_size_a": res.sample_size_a,
            "sample_size_b": res.sample_size_b,
            "conversions_a": res.conversions_a,
            "conversions_b": res.conversions_b,
            "rate_a": res.rate_a,
            "rate_b": res.rate_b,
            "difference_pp": res.difference_pp,
            "p_value": res.p_value,
            "significant": res.significant,
        }
        return ToolResult(
            ok=True,
            content=(
                f"Experiment {res.experiment_id}: Rate A={res.rate_a:.3f}, Rate B={res.rate_b:.3f}, "
                f"Diff={res.difference_pp:.2f}pp, p-value={res.p_value:.4f}, Significant={res.significant}"
            ),
            data=data,
        )


class FakeLLM:
    def __init__(self, decisions: list[ActionDecision]):
        self._decisions = list(decisions)
        self.call_count = 0

    async def decide(self, messages, response_model, max_retries=2):
        if not self._decisions:
            raise RuntimeError("FakeLLM ran out of decisions")
        d = self._decisions.pop(0)
        self.call_count += 1
        return d


@pytest.mark.asyncio
async def test_full_product_analytics_investigation_and_governance_flow():
    """Verify complete analytical workflow:
    1. SQL inspection of funnel dropoff.
    2. Hypothesis significance testing of EXP-204.
    3. Final recommendation with verified numbers and policy constraint evaluation.
    """
    tools = ToolRegistry()
    tools.register(_LocalSqlTool())
    tools.register(_LocalSignificanceTool())

    # Deterministic rollout governance rules
    def require_significance_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        exp_id = proposal.parameters.get("experiment_id", "")
        facts = ctx.evidence_ledger.extract_numerical_facts() if ctx.evidence_ledger else {}
        # Check if p_value exists and is significant (< 0.05)
        p_vals = [v for k, v in facts.items() if "p_value" in k]
        if not p_vals or any(p >= 0.05 for p in p_vals):
            return PolicyRuleResult(
                rule_name="require_significance",
                passed=False,
                status=PolicyStatus.BLOCKED,
                message="Cannot rollout without statistically significant evidence (p < 0.05).",
            )
        return PolicyRuleResult(
            rule_name="require_significance",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message="Statistical significance verified.",
        )

    def traffic_allocation_rule(proposal: ProposedDecision, ctx: DecisionContext) -> PolicyRuleResult:
        allocation = proposal.parameters.get("traffic_percentage", 0)
        if allocation > 50:
            return PolicyRuleResult(
                rule_name="traffic_allocation_cap",
                passed=False,
                status=PolicyStatus.REQUIRES_APPROVAL,
                message=f"Rollout allocation of {allocation}% exceeds autonomous 50% cap. Human sign-off required.",
            )
        return PolicyRuleResult(
            rule_name="traffic_allocation_cap",
            passed=True,
            status=PolicyStatus.ALLOWED,
            message=f"Allocation of {allocation}% is within safe automated limits.",
        )

    policy_engine = PolicyEngine(rules=[require_significance_rule, traffic_allocation_rule])

    # LLM decisions simulating agentic decision process
    decisions = [
        # Step 0: inspect funnel
        ActionDecision(
            thought="Check funnel dropoff to identify bottleneck.",
            action=ActionType.CALL_TOOL,
            tool_name="execute_sql",
            tool_arguments={"query": "SELECT funnel_step, mobile_dropoff_rate FROM funnel_dropoff ORDER BY step_order"},
        ),
        # Step 1: run hypothesis test on EXP-204
        ActionDecision(
            thought="Run significance test on EXP-204 checkout experiment.",
            action=ActionType.CALL_TOOL,
            tool_name="test_significance",
            tool_arguments={"experiment_id": "EXP-204", "alpha": 0.05},
        ),
        # Step 2: finalize with grounded metrics and decision proposal
        ActionDecision(
            thought="Mobile dropoff was 48.0% at payment, and EXP-204 achieved 2.9% conversion with significance.",
            action=ActionType.FINAL,
            answer=(
                "Mobile payment dropoff reached 48.0%. Experiment EXP-204 recovered conversion "
                "from 1.6% to 2.9% with statistical significance. I recommend rolling out the redesign."
            ),
            proposed_decision=ProposedDecision(
                intent="Deploy EXP-204 mobile one-page checkout redesign",
                recommended_action="rollout_variant_b",
                parameters={"experiment_id": "EXP-204", "traffic_percentage": 100},
                reasoning="Variant B significantly outperforms control (2.9% vs 1.6%, p < 0.05).",
                confidence=0.95,
            ),
        ),
    ]

    llm = FakeLLM(decisions)
    runtime = AgentRuntime(
        llm=llm,
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Why did mobile conversion drop and what should we do?")

    # Verify agent completion and grounded facts
    assert state.done is True
    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert len(state.steps) == 3

    # Verify Evidence Ledger contents
    assert len(state.evidence_ledger) == 2
    sql_ev = state.evidence_ledger.all()[0]
    assert sql_ev.source == EvidenceSource.ANALYTICS_SQL
    assert sql_ev.kind == EvidenceKind.OBSERVATION

    stat_ev = state.evidence_ledger.all()[1]
    assert stat_ev.source == EvidenceSource.STATISTICS
    assert stat_ev.kind == EvidenceKind.CALCULATION
    assert stat_ev.data["significant"] is True

    # Verify Policy Evaluation & Precedence (REQUIRES_APPROVAL triggered due to 100% traffic)
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert state.policy_evaluation.requires_approval is True
    assert "Human sign-off required" in state.policy_evaluation.audit_notes

    # Verify Audit Trace completeness
    trace = state.audit_trace
    assert trace["query"] == "Why did mobile conversion drop and what should we do?"
    assert trace["steps_count"] == 3
    assert trace["evidence_count"] == 2
    assert trace["proposed_decision"]["parameters"]["traffic_percentage"] == 100
    assert trace["policy_evaluation"]["final_status"] == "requires_approval"


@pytest.mark.asyncio
async def test_numerical_verification_blocks_fabricated_analytics_metric():
    """Agent attempts to claim a fabricated conversion figure; verification blocks it."""
    tools = ToolRegistry()
    tools.register(_LocalSqlTool())

    decisions = [
        # Step 0: Real SQL query
        ActionDecision(
            thought="Query checkout metrics",
            action=ActionType.CALL_TOOL,
            tool_name="execute_sql",
            tool_arguments={"query": "SELECT mobile_dropoff_rate FROM funnel_dropoff WHERE step_order = 3"},
        ),
        # Step 1: Hallucinate 92.5% dropoff (actual is 48.0%)
        ActionDecision(
            thought="Propose ungrounded answer",
            action=ActionType.FINAL,
            answer="Mobile payment dropoff reached 92.5%.",
        ),
        # Step 2: Correct to real 48.0% dropoff
        ActionDecision(
            thought="Correct answer to observed 48.0%",
            action=ActionType.FINAL,
            answer="Mobile payment dropoff reached 48.0%.",
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        max_steps=5,
    )

    state = await runtime.run("What was the mobile payment dropoff rate?")

    assert len(state.steps) == 3
    # Step 1 failed verification
    assert state.steps[1].status.value == "error"
    assert "Verification failed" in state.steps[1].observation

    # Step 2 succeeded
    assert state.steps[2].status.value == "ok"
    assert state.final_answer == "Mobile payment dropoff reached 48.0%."
    assert state.termination_reason == TerminationReason.FINAL_ANSWER


@pytest.mark.asyncio
async def test_budget_controller_enforces_sql_query_cap():
    """Verify that ActionBudget limits excessive SQL explorations."""
    tools = ToolRegistry()
    tools.register(_LocalSqlTool())

    budget = ActionBudget(max_steps=5, max_sql_queries=1)
    controller = BudgetController(budget=budget)

    decisions = [
        # Query 1 (allowed)
        ActionDecision(
            thought="Run first query",
            action=ActionType.CALL_TOOL,
            tool_name="execute_sql",
            tool_arguments={"query": "SELECT COUNT(*) FROM daily_checkout_metrics"},
        ),
        # Query 2 (blocked by budget)
        ActionDecision(
            thought="Run second query",
            action=ActionType.CALL_TOOL,
            tool_name="execute_sql",
            tool_arguments={"query": "SELECT * FROM experiments"},
        ),
        # Finalize
        ActionDecision(
            thought="Answer with available data",
            action=ActionType.FINAL,
            answer="Answer concluded based on single allowed query.",
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        budget_controller=controller,
        max_steps=5,
    )

    state = await runtime.run("Investigate metrics")

    assert len(state.steps) == 3
    assert state.steps[0].status.value == "ok"
    assert state.steps[1].status.value == "error"
    assert "SQL query budget exhausted" in state.steps[1].observation
    assert controller.usage.sql_queries_used == 1
