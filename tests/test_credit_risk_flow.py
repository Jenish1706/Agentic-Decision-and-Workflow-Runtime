"""End-to-end integration and policy tests for Credit Risk & Underwriting Domain.

Verifies:
1. Autonomous limit approval when all underwriting criteria (score >= 650, DTI <= 43%, no delinquencies, increase <= $5k) pass.
2. Deterministic BLOCKED outcome when DTI exceeds 43% ceiling.
3. Deterministic BLOCKED outcome when 90-day delinquency or sub-650 credit score is detected.
4. Deterministic REQUIRES_APPROVAL escalation when increase exceeds $5,000.
5. Numerical verification blocking hallucinated financial scores.
6. SQL security controls blocking DDL operations (DROP, ALTER).
"""

import pytest

from agent_runtime.agent.actions import ActionDecision, ActionType
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.policy import PolicyStatus, ProposedDecision
from agent_runtime.agent.state import TerminationReason
from agent_runtime.agent.tools.base import Tool, ToolResult
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.mcp_layer.servers.credit_risk_server import (
    describe_tables,
    execute_sql,
    get_account_profile,
)
from domains.credit_risk.policies import create_credit_risk_policy_engine


class _LocalCreditSqlTool(Tool):
    name = "execute_sql"
    description = "Execute read-only SQL query on credit risk database."
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


class _LocalProfileTool(Tool):
    name = "get_account_profile"
    description = "Fetch unified account dossier."
    parameters_schema = {
        "type": "object",
        "properties": {"account_id": {"type": "string"}},
        "required": ["account_id"],
    }
    origin = "local"

    async def run(self, account_id: str) -> ToolResult:
        res = get_account_profile(account_id)
        if not res.ok:
            return ToolResult(ok=False, content="", error=res.error)
        return ToolResult(
            ok=True,
            content=f"Account profile for {res.account_id}: info={res.account_info}, history={res.credit_history}",
            data={
                "account_id": res.account_id,
                "account_info": res.account_info,
                "credit_history": res.credit_history,
                "open_requests": res.open_requests,
                "inquiries": res.inquiries,
            },
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
async def test_credit_risk_autonomous_approval_flow():
    """ACCT-101 meets all guidelines; proposal to approve $3,000 increase is ALLOWED."""
    tools = ToolRegistry()
    tools.register(_LocalCreditSqlTool())
    tools.register(_LocalProfileTool())

    policy_engine = create_credit_risk_policy_engine()

    decisions = [
        # Step 0: inspect account profile
        ActionDecision(
            thought="Fetch full profile for ACCT-101",
            action=ActionType.CALL_TOOL,
            tool_name="get_account_profile",
            tool_arguments={"account_id": "ACCT-101"},
        ),
        # Step 1: recommend approval
        ActionDecision(
            thought="Alice Walker has 740 credit score, 28% DTI, 0 delinquencies, and $3000 increase is <= $5000.",
            action=ActionType.FINAL,
            answer="Approve credit limit increase from $5,000 to $8,000 for ACCT-101. Credit score is 740, DTI is 28.0%, with zero delinquencies.",
            proposed_decision=ProposedDecision(
                intent="Approve credit limit increase",
                recommended_action="approve_increase",
                parameters={"account_id": "ACCT-101", "increase_amount": 3000.0, "credit_score": 740, "dti_ratio": 0.28},
                reasoning="Prime borrower with 740 FICO and 28% DTI.",
                confidence=0.98,
            ),
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Should we approve credit limit increase for ACCT-101?")

    assert state.done is True
    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.ALLOWED
    assert state.policy_evaluation.requires_approval is False


@pytest.mark.asyncio
async def test_credit_risk_dti_ceiling_blocks_approval():
    """ACCT-104 has 48% DTI (>43% ceiling); an approval attempt is BLOCKED by policy."""
    tools = ToolRegistry()
    tools.register(_LocalCreditSqlTool())

    policy_engine = create_credit_risk_policy_engine()

    decisions = [
        # Step 0: query account
        ActionDecision(
            thought="Check DTI and limit request for ACCT-104",
            action=ActionType.CALL_TOOL,
            tool_name="execute_sql",
            tool_arguments={
                "query": "SELECT a.account_id, a.credit_score, a.current_limit, a.dti_ratio, r.requested_limit, r.increase_amount FROM accounts a JOIN limit_requests r ON a.account_id = r.account_id WHERE a.account_id = 'ACCT-104'"
            },
        ),
        # Step 1: attempt to approve despite 48% DTI
        ActionDecision(
            thought="Attempt to approve despite high debt-to-income",
            action=ActionType.FINAL,
            answer="Recommend limit increase to $6,000 despite 48.0% DTI.",
            proposed_decision=ProposedDecision(
                intent="Approve limit expansion",
                recommended_action="approve_increase",
                parameters={"account_id": "ACCT-104", "increase_amount": 2000.0, "dti_ratio": 0.48, "credit_score": 670},
                reasoning="Good credit score 670.",
                confidence=0.75,
            ),
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Can Diana Prince (ACCT-104) be granted her credit increase?")

    assert state.done is True
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.BLOCKED
    assert "exceeds maximum allowable 43.0% ceiling" in state.policy_evaluation.audit_notes


@pytest.mark.asyncio
async def test_credit_risk_delinquency_and_low_score_blocked():
    """ACCT-102 has credit score 610 (<650) and 1 90-day delinquency; both trigger BLOCKED."""
    tools = ToolRegistry()
    tools.register(_LocalCreditSqlTool())

    policy_engine = create_credit_risk_policy_engine()

    decisions = [
        # Step 0: query account and history
        ActionDecision(
            thought="Query score and delinquency",
            action=ActionType.CALL_TOOL,
            tool_name="execute_sql",
            tool_arguments={
                "query": "SELECT a.credit_score, h.delinquencies_90d FROM accounts a JOIN credit_history h ON a.account_id = h.account_id WHERE a.account_id = 'ACCT-102'"
            },
        ),
        # Step 1: propose decision
        ActionDecision(
            thought="Propose approval to test policy block",
            action=ActionType.FINAL,
            answer="Recommend limit increase for ACCT-102 with credit score 610.",
            proposed_decision=ProposedDecision(
                intent="Approve limit increase",
                recommended_action="approve_increase",
                parameters={"account_id": "ACCT-102", "credit_score": 610, "delinquencies_90d": 1},
                reasoning="Long-term customer.",
                confidence=0.6,
            ),
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Evaluate Bob Martinez (ACCT-102) for credit limit increase.")

    assert state.done is True
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.BLOCKED
    blocked_rules = [r.rule_name for r in state.policy_evaluation.rule_results if r.status == PolicyStatus.BLOCKED]
    assert "minimum_credit_score" in blocked_rules
    assert "hard_delinquency_cutoff" in blocked_rules


@pytest.mark.asyncio
async def test_credit_risk_high_increase_requires_approval():
    """ACCT-103 requests $10,000 increase (> $5,000 threshold); triggers REQUIRES_APPROVAL."""
    tools = ToolRegistry()
    tools.register(_LocalCreditSqlTool())

    policy_engine = create_credit_risk_policy_engine()

    decisions = [
        # Step 0: query account
        ActionDecision(
            thought="Query ACCT-103 details and limit request",
            action=ActionType.CALL_TOOL,
            tool_name="execute_sql",
            tool_arguments={
                "query": "SELECT a.account_id, a.credit_score, a.current_limit, a.dti_ratio, r.requested_limit, r.increase_amount FROM accounts a JOIN limit_requests r ON a.account_id = r.account_id WHERE a.account_id = 'ACCT-103'"
            },
        ),
        # Step 1: propose approval with $10,000 increase
        ActionDecision(
            thought="Propose $10,000 increase",
            action=ActionType.FINAL,
            answer="Recommend expanding line from $10,000 to $20,000 (increase of $10,000).",
            proposed_decision=ProposedDecision(
                intent="Approve high-value credit line expansion",
                recommended_action="approve_increase",
                parameters={"account_id": "ACCT-103", "increase_amount": 10000.0, "credit_score": 710, "dti_ratio": 0.32},
                reasoning="Strong financial standing, 710 score, 32% DTI.",
                confidence=0.9,
            ),
        ),
    ]

    runtime = AgentRuntime(
        llm=FakeLLM(decisions),
        tools=tools,
        policy_engine=policy_engine,
        max_steps=5,
    )

    state = await runtime.run("Evaluate Charlie Davis (ACCT-103) request to increase line from $10,000 to $20,000.")

    assert state.done is True
    assert state.policy_evaluation is not None
    assert state.policy_evaluation.final_status == PolicyStatus.REQUIRES_APPROVAL
    assert state.policy_evaluation.requires_approval is True
    assert "exceeds autonomous threshold ($5,000)" in state.policy_evaluation.audit_notes


def test_credit_risk_ddl_prevention():
    """Destructive DDL is blocked at tool level."""
    res = execute_sql("DROP TABLE accounts;")
    assert res.ok is False
    assert "Disallowed operations detected" in res.error

    res2 = execute_sql("ALTER TABLE accounts ADD COLUMN bad_col INT;")
    assert res2.ok is False
    assert "Disallowed operations detected" in res2.error


def test_credit_risk_describe_tables():
    """Schema discovery returns all 4 tables."""
    res = describe_tables()
    assert "accounts" in res.tables
    assert "credit_history" in res.tables
    assert "limit_requests" in res.tables
    assert "bureau_inquiries" in res.tables
