import pytest

from agent_runtime.agent.actions import ActionDecision, ActionType, DecisionExtraction
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.policy import PolicyEngine, PolicyRuleResult, ProposedDecision
from agent_runtime.agent.state import TerminationReason
from agent_runtime.agent.tools.calculator import CalculatorTool
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.rag.documents import Chunk

from .fakes import FakeLLMClient, raises_llm_error


class _StubRag:
    """Minimal stand-in for the agent loop's Retriever protocol (async
    `retrieve(query)`) - no embeddings, no MCP, just a fixed chunk so
    retrieve-handling can be tested in isolation."""

    def __init__(self, chunk: Chunk | None = None):
        self._chunk = chunk

    async def retrieve(self, query: str):
        if self._chunk is None:
            return []
        return [_Result(self._chunk, 0.9)]


class _Result:
    def __init__(self, chunk, score):
        self.chunk = chunk
        self.score = score


def _chunk(doc_id="aurora-doc"):
    return Chunk(chunk_id=f"{doc_id}::0", doc_id=doc_id, doc_title="Aurora Doc", position=0, text="Aurora limit is 5000/s")


@pytest.mark.asyncio
async def test_immediate_grounded_final_answer_terminates():
    llm = FakeLLMClient(decisions=[ActionDecision(thought="I already know", action=ActionType.FINAL, answer="No evidence needed.")])
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=None, max_steps=5)
    state = await runtime.run("some question")
    assert state.done
    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert state.final_answer == "No evidence needed."
    assert len(state.steps) == 1


@pytest.mark.asyncio
async def test_retrieve_then_final_with_citation():
    chunk = _chunk()
    llm = FakeLLMClient(
        decisions=[
            ActionDecision(thought="need docs", action=ActionType.RETRIEVE, query="aurora rate limit"),
            ActionDecision(thought="got it", action=ActionType.FINAL, answer="The limit is 5000/s [1]."),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=_StubRag(chunk), max_steps=5)
    state = await runtime.run("what is the rate limit?")
    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert state.citations == ["[1]"]
    assert state.citation_sources["[1]"] == "aurora-doc"


@pytest.mark.asyncio
async def test_call_tool_then_final():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    llm = FakeLLMClient(
        decisions=[
            ActionDecision(thought="compute", action=ActionType.CALL_TOOL, tool_name="calculator", tool_arguments={"expression": "2+2"}),
            ActionDecision(thought="done", action=ActionType.FINAL, answer="The answer is 4."),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=registry, rag=None, max_steps=5)
    state = await runtime.run("what is 2+2?")
    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert len(state.tool_calls()) == 1
    assert "4" in state.steps[0].observation


@pytest.mark.asyncio
async def test_retrieve_with_empty_query_falls_back_to_original_question():
    class _RecordingRag:
        def __init__(self):
            self.queries_seen: list[str] = []

        async def retrieve(self, query: str):
            self.queries_seen.append(query)
            return []

    rag = _RecordingRag()
    llm = FakeLLMClient(
        decisions=[
            ActionDecision(thought="search, but forgot the query field", action=ActionType.RETRIEVE, query=None),
            ActionDecision(thought="done", action=ActionType.FINAL, answer="Nothing found."),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=rag, max_steps=5)
    await runtime.run("what is the aurora rate limit?")
    assert rag.queries_seen == ["what is the aurora rate limit?"]


def _allow_everything(proposal, ctx):
    return PolicyRuleResult.allowed("allow_everything", "ok")


@pytest.mark.asyncio
async def test_unattached_decision_is_recovered_and_judged_by_policy():
    # Measured with qwen2.5:7b: the model usually describes a decision in prose
    # and leaves `proposed_decision` null, so policy never ran.
    llm = FakeLLMClient(
        decisions=[
            ActionDecision(thought="answer", action=ActionType.FINAL, answer="Approve the request for ACCT-101."),
            DecisionExtraction(
                asks_for_decision=True,
                recommended_action="approve_increase",
                parameters={"account_id": "ACCT-101"},
                reasoning="The answer approves it.",
                confidence=90,
            ),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), policy_engine=PolicyEngine([_allow_everything]))
    state = await runtime.run("Should we approve ACCT-101's increase?")

    assert state.proposed_decision.recommended_action == "approve_increase"
    assert state.proposed_decision.confidence == pytest.approx(0.9)
    assert state.proposed_decision_source == "extracted"
    assert state.policy_evaluation is not None
    assert state.audit_trace["proposed_decision_source"] == "extracted"


@pytest.mark.asyncio
async def test_no_decision_is_invented_for_a_factual_question():
    llm = FakeLLMClient(
        decisions=[
            ActionDecision(thought="answer", action=ActionType.FINAL, answer="The service is healthy."),
            DecisionExtraction(asks_for_decision=False, recommended_action="", parameters={}, reasoning="", confidence=0.0),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), policy_engine=PolicyEngine([_allow_everything]))
    state = await runtime.run("Is the service healthy?")

    assert state.proposed_decision is None and state.policy_evaluation is None


@pytest.mark.asyncio
async def test_failed_extraction_keeps_the_answer():
    llm = FakeLLMClient(
        decisions=[ActionDecision(thought="answer", action=ActionType.FINAL, answer="Approve it."), raises_llm_error()]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), policy_engine=PolicyEngine([_allow_everything]))
    state = await runtime.run("Should we approve it?")

    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert state.final_answer == "Approve it."
    assert state.policy_evaluation is None


@pytest.mark.asyncio
async def test_attached_decision_needs_no_extraction_call():
    proposal = ProposedDecision(intent="i", recommended_action="approve", reasoning="r", confidence=0.8)
    llm = FakeLLMClient(
        decisions=[ActionDecision(thought="answer", action=ActionType.FINAL, answer="Approve it.", proposed_decision=proposal)]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), policy_engine=PolicyEngine([_allow_everything]))
    state = await runtime.run("Should we approve it?")

    assert len(llm.decide_calls) == 1  # the scripted LLM would raise if a second call were made
    assert state.proposed_decision_source == "final_step"


def test_citation_rule_is_only_prompted_when_retrieval_exists():
    # Found in a real run: a tool-only agent told to "cite retrieved documents
    # using [n] tags" appended invented tags to a SQL-based answer.
    tool_only = AgentRuntime(llm=FakeLLMClient(), tools=ToolRegistry(), rag=None)
    with_rag = AgentRuntime(llm=FakeLLMClient(), tools=ToolRegistry(), rag=_StubRag())
    assert "[n] tags" not in tool_only._system_prompt(max_steps=5)
    assert '"retrieve"' not in tool_only._system_prompt(max_steps=5)
    assert "[n] tags" in with_rag._system_prompt(max_steps=5)


@pytest.mark.asyncio
async def test_retrieve_unavailable_when_rag_is_none():
    llm = FakeLLMClient(
        decisions=[
            ActionDecision(thought="try to retrieve", action=ActionType.RETRIEVE, query="anything"),
            ActionDecision(thought="give up gracefully", action=ActionType.FINAL, answer="I can't search documents here."),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=None, max_steps=5)
    state = await runtime.run("some question")
    assert state.steps[0].status.value == "error"
    assert "not available" in state.steps[0].observation


@pytest.mark.asyncio
async def test_max_steps_reached_when_agent_never_finalizes():
    decisions = [ActionDecision(thought="keep looking", action=ActionType.RETRIEVE, query="x") for _ in range(3)]
    llm = FakeLLMClient(decisions=decisions)
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=_StubRag(None), max_steps=3)
    state = await runtime.run("unanswerable question")
    assert state.termination_reason == TerminationReason.MAX_STEPS_REACHED
    assert len(state.steps) == 3


@pytest.mark.asyncio
async def test_retrieval_is_instrumented_for_search_efficiency():
    chunk = _chunk()
    llm = FakeLLMClient(
        decisions=[
            ActionDecision(thought="search", action=ActionType.RETRIEVE, query="aurora limit"),
            ActionDecision(thought="search again, same thing", action=ActionType.RETRIEVE, query="aurora limit"),
            ActionDecision(thought="answer", action=ActionType.FINAL, answer="It's 5000/s [1]."),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=_StubRag(chunk), max_steps=5)
    state = await runtime.run("what is the rate limit?")
    assert state.retrieval_queries == ["aurora limit", "aurora limit"]
    # duplicates kept on purpose - the repeats are the signal
    assert state.retrieved_chunk_ids == [chunk.chunk_id, chunk.chunk_id]


@pytest.mark.asyncio
async def test_retrieval_budget_blocks_further_searches_and_explains_why():
    chunk = _chunk()
    llm = FakeLLMClient(
        decisions=[
            ActionDecision(thought="search", action=ActionType.RETRIEVE, query="first"),
            ActionDecision(thought="search again", action=ActionType.RETRIEVE, query="second"),
            ActionDecision(thought="answer", action=ActionType.FINAL, answer="It's 5000/s [1]."),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=_StubRag(chunk), max_steps=5, max_retrievals=1)
    state = await runtime.run("what is the rate limit?")

    assert state.retrieval_queries == ["first"]  # second search never executed
    assert state.steps[1].status.value == "error"
    assert "budget exhausted" in state.steps[1].observation
    assert state.termination_reason == TerminationReason.FINAL_ANSWER


@pytest.mark.asyncio
async def test_no_budget_by_default_allows_repeated_retrieval():
    chunk = _chunk()
    llm = FakeLLMClient(
        decisions=[
            ActionDecision(thought="a", action=ActionType.RETRIEVE, query="one"),
            ActionDecision(thought="b", action=ActionType.RETRIEVE, query="two"),
            ActionDecision(thought="c", action=ActionType.RETRIEVE, query="three"),
            ActionDecision(thought="answer", action=ActionType.FINAL, answer="The limit is 5000/s [1]."),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=_StubRag(chunk), max_steps=6)
    state = await runtime.run("q")
    assert len(state.retrieval_queries) == 3


@pytest.mark.asyncio
async def test_memory_write_failure_does_not_discard_a_good_answer():
    # Regression: the long-term-memory "is this worth remembering?" call runs
    # after the answer is finalized, so an LLMError there used to throw away
    # a perfectly good completed run.
    class _FailingLongTerm:
        def retrieve(self, query, top_k=2):
            return []

        async def maybe_store(self, llm, query, answer):
            raise raises_llm_error()

    # A non-numeric answer on purpose: this test is about memory-write
    # failures, and an ungrounded number would (correctly) be rejected by
    # numerical verification before memory is ever reached.
    llm = FakeLLMClient(
        decisions=[ActionDecision(thought="done", action=ActionType.FINAL, answer="The service is healthy.")]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=None, long_term=_FailingLongTerm(), max_steps=5)
    state = await runtime.run("some question")
    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert state.final_answer == "The service is healthy."
    assert state.answer_verified is True


@pytest.mark.asyncio
async def test_citations_reflect_only_tags_the_answer_actually_uses():
    # Two chunks get retrieved across the run, but the answer cites only [1].
    chunk = _chunk()
    llm = FakeLLMClient(
        decisions=[
            ActionDecision(thought="need docs", action=ActionType.RETRIEVE, query="aurora"),
            ActionDecision(thought="need more", action=ActionType.RETRIEVE, query="aurora again"),
            ActionDecision(thought="answer", action=ActionType.FINAL, answer="The limit is 5000/s [1]."),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=_StubRag(chunk), max_steps=5)
    state = await runtime.run("what is the rate limit?")
    assert state.citations == ["[1]"]
    assert set(state.citation_sources) == {"[1]"}


@pytest.mark.asyncio
async def test_llm_error_terminates_gracefully_without_crashing():
    llm = FakeLLMClient(decisions=[raises_llm_error()])
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=None, max_steps=5)
    state = await runtime.run("some question")
    assert state.termination_reason == TerminationReason.UNRECOVERABLE_ERROR
    assert state.final_answer is not None


@pytest.mark.asyncio
async def test_ungrounded_final_is_rejected_and_agent_gets_a_second_attempt():
    # first attempt cites a tag that was never retrieved -> rejected, fed
    # back as a corrective observation; second attempt drops the fabricated
    # citation and should be accepted.
    llm = FakeLLMClient(
        decisions=[
            ActionDecision(thought="guessing", action=ActionType.FINAL, answer="It's 5000 [1]."),
            ActionDecision(thought="fixing", action=ActionType.FINAL, answer="I don't have evidence for this."),
        ]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=None, max_steps=5)
    state = await runtime.run("what is the limit?")
    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert state.final_answer == "I don't have evidence for this."
    assert len(state.steps) == 2
    assert state.steps[0].status.value == "error"
    assert "Verification failed" in state.steps[0].observation


@pytest.mark.asyncio
async def test_ungrounded_final_on_last_step_is_accepted_anyway():
    # with no steps left to retry, the loop accepts the best-effort answer
    # rather than silently discarding it and returning nothing.
    llm = FakeLLMClient(
        decisions=[ActionDecision(thought="guessing", action=ActionType.FINAL, answer="It's 5000 [1].")]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=None, max_steps=1)
    state = await runtime.run("what is the limit?")
    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert state.final_answer == "It's 5000 [1]."
    # Accepted, but never mistaken for a verified answer.
    assert state.answer_verified is False
    assert state.verification_issues
