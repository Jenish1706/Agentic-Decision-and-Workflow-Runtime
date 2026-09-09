import pytest

from agent_runtime.agent.actions import ActionDecision, ActionType
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.state import TerminationReason
from agent_runtime.agent.tools.calculator import CalculatorTool
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.rag.context_builder import build_context
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
            ActionDecision(thought="answer", action=ActionType.FINAL, answer="Done [1]."),
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

    llm = FakeLLMClient(
        decisions=[ActionDecision(thought="done", action=ActionType.FINAL, answer="The answer is 42.")]
    )
    runtime = AgentRuntime(llm=llm, tools=ToolRegistry(), rag=None, long_term=_FailingLongTerm(), max_steps=5)
    state = await runtime.run("some question")
    assert state.termination_reason == TerminationReason.FINAL_ANSWER
    assert state.final_answer == "The answer is 42."


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
