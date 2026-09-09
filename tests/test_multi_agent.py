import pytest

from agent_runtime.agent.actions import ActionDecision, ActionType
from agent_runtime.agent.llm_judge import JudgeVerdict
from agent_runtime.agent.tools.calculator import CalculatorTool
from agent_runtime.agent.tools.registry import ToolRegistry
from agent_runtime.multi_agent.supervisor import DelegationPlan, Supervisor
from agent_runtime.multi_agent.synthesizer import Synthesizer
from agent_runtime.multi_agent.workers import ResearcherWorker, RetrieverWorker
from agent_runtime.rag.documents import Chunk

from .fakes import FakeLLMClient, raises_llm_error


class _StubRag:
    def retrieve(self, query: str, top_k_rerank: int | None = None):
        chunk = Chunk(chunk_id="doc::0", doc_id="doc", doc_title="Doc", position=0, text="Aurora limit is 5000/s")
        return [_Hit(chunk, 0.9)]


class _Hit:
    def __init__(self, chunk, score):
        self.chunk = chunk
        self.score = score


@pytest.mark.asyncio
async def test_retrieval_only_plan_skips_researcher_entirely():
    llm = FakeLLMClient(
        decisions=[
            DelegationPlan(thought="just needs docs", needs_retrieval=True, needs_tools=False),
            JudgeVerdict(grounded=True, issues=[], confidence=0.9),
        ],
        generations=["The limit is 5000/s [1]."],
    )
    supervisor = Supervisor(
        llm=llm,
        retriever=RetrieverWorker(rag=_StubRag()),
        researcher=ResearcherWorker(llm=llm, tools=ToolRegistry()),
        synthesizer=Synthesizer(llm=llm),
    )
    result = await supervisor.run("what is the aurora rate limit?")
    assert result.grounded
    assert result.citations == ["[1]"]
    assert result.tools_called == []
    assert [entry.worker for entry in result.worker_log] == ["retriever", "verifier"]


@pytest.mark.asyncio
async def test_tools_only_plan_delegates_to_researcher():
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    llm = FakeLLMClient(
        decisions=[
            DelegationPlan(thought="needs math", needs_retrieval=False, needs_tools=True),
            ActionDecision(thought="compute", action=ActionType.CALL_TOOL, tool_name="calculator", tool_arguments={"expression": "2+2"}),
            ActionDecision(thought="done researching", action=ActionType.FINAL, answer="(internal, discarded)"),
            JudgeVerdict(grounded=True, issues=[], confidence=0.9),
        ],
        generations=["The answer is 4."],
    )
    supervisor = Supervisor(
        llm=llm,
        retriever=RetrieverWorker(rag=_StubRag()),
        researcher=ResearcherWorker(llm=llm, tools=tools),
        synthesizer=Synthesizer(llm=llm),
    )
    result = await supervisor.run("what is 2+2?")
    assert result.tools_called == ["calculator"]
    assert result.citations == []
    assert result.final_answer == "The answer is 4."


@pytest.mark.asyncio
async def test_ungrounded_draft_triggers_one_repair_attempt():
    llm = FakeLLMClient(
        decisions=[
            DelegationPlan(thought="needs docs", needs_retrieval=True, needs_tools=False),
            JudgeVerdict(grounded=False, issues=["cites a source that wasn't retrieved"], confidence=0.3),
            JudgeVerdict(grounded=True, issues=[], confidence=0.9),
        ],
        generations=["It's 5000/s [9] (fabricated).", "It's 5000/s [1]."],
    )
    supervisor = Supervisor(
        llm=llm,
        retriever=RetrieverWorker(rag=_StubRag()),
        researcher=ResearcherWorker(llm=llm, tools=ToolRegistry()),
        synthesizer=Synthesizer(llm=llm),
    )
    result = await supervisor.run("what is the rate limit?")
    assert result.repair_attempts == 1
    assert result.grounded
    assert result.final_answer == "It's 5000/s [1]."


@pytest.mark.asyncio
async def test_multi_agent_retrieves_exactly_once_implicit_budget_of_one():
    # The premise of the retrieval-budget experiment: the multi-agent path
    # cannot search more than once, by construction.
    llm = FakeLLMClient(
        decisions=[
            DelegationPlan(thought="needs docs", needs_retrieval=True, needs_tools=False),
            JudgeVerdict(grounded=True, issues=[], confidence=0.9),
        ],
        generations=["It's 5000/s [1]."],
    )
    supervisor = Supervisor(
        llm=llm,
        retriever=RetrieverWorker(rag=_StubRag()),
        researcher=ResearcherWorker(llm=llm, tools=ToolRegistry()),
        synthesizer=Synthesizer(llm=llm),
    )
    result = await supervisor.run("what is the rate limit?")
    assert result.retrieval_queries == ["what is the rate limit?"]
    assert len(result.retrieved_chunk_ids) == 1


@pytest.mark.asyncio
async def test_multi_agent_records_no_retrieval_when_plan_skips_it():
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    llm = FakeLLMClient(
        decisions=[
            DelegationPlan(thought="needs math", needs_retrieval=False, needs_tools=True),
            ActionDecision(thought="compute", action=ActionType.CALL_TOOL, tool_name="calculator", tool_arguments={"expression": "2+2"}),
            ActionDecision(thought="done", action=ActionType.FINAL, answer="(internal)"),
            JudgeVerdict(grounded=True, issues=[], confidence=0.9),
        ],
        generations=["The answer is 4."],
    )
    supervisor = Supervisor(
        llm=llm,
        retriever=RetrieverWorker(rag=_StubRag()),
        researcher=ResearcherWorker(llm=llm, tools=tools),
        synthesizer=Synthesizer(llm=llm),
    )
    result = await supervisor.run("what is 2+2?")
    assert result.retrieval_queries == []
    assert result.retrieved_chunk_ids == []


@pytest.mark.asyncio
async def test_verifier_failure_does_not_crash_the_run_or_trigger_repairs():
    # Regression: an LLMError from the verifier used to propagate out of
    # supervisor.run() and abort the whole demo.
    llm = FakeLLMClient(
        decisions=[
            DelegationPlan(thought="needs docs", needs_retrieval=True, needs_tools=False),
            raises_llm_error(),
        ],
        generations=["It's 5000/s [1]."],
    )
    supervisor = Supervisor(
        llm=llm,
        retriever=RetrieverWorker(rag=_StubRag()),
        researcher=ResearcherWorker(llm=llm, tools=ToolRegistry()),
        synthesizer=Synthesizer(llm=llm),
    )
    result = await supervisor.run("what is the rate limit?")
    assert result.final_answer == "It's 5000/s [1]."
    assert result.grounded is False
    assert result.repair_attempts == 0  # no point repairing against a verdict we never got
    assert any("unavailable" in entry.summary for entry in result.worker_log)


@pytest.mark.asyncio
async def test_planner_failure_falls_back_to_consulting_every_specialist():
    tools = ToolRegistry()
    tools.register(CalculatorTool())
    llm = FakeLLMClient(
        decisions=[
            raises_llm_error(),  # the planner call
            ActionDecision(thought="compute", action=ActionType.CALL_TOOL, tool_name="calculator", tool_arguments={"expression": "2+2"}),
            ActionDecision(thought="done", action=ActionType.FINAL, answer="(internal)"),
            JudgeVerdict(grounded=True, issues=[], confidence=0.9),
        ],
        generations=["The answer is 4 [1]."],
    )
    supervisor = Supervisor(
        llm=llm,
        retriever=RetrieverWorker(rag=_StubRag()),
        researcher=ResearcherWorker(llm=llm, tools=tools),
        synthesizer=Synthesizer(llm=llm),
    )
    result = await supervisor.run("what is 2+2?")
    assert result.plan.needs_retrieval and result.plan.needs_tools
    assert result.final_answer == "The answer is 4 [1]."


@pytest.mark.asyncio
async def test_citations_reflect_only_tags_the_answer_actually_uses():
    # The retriever supplies one chunk as [1], but the draft cites nothing -
    # citations should be empty, not "everything that was retrieved".
    llm = FakeLLMClient(
        decisions=[
            DelegationPlan(thought="needs docs", needs_retrieval=True, needs_tools=False),
            JudgeVerdict(grounded=True, issues=[], confidence=0.9),
        ],
        generations=["The limit is 5000/s."],
    )
    supervisor = Supervisor(
        llm=llm,
        retriever=RetrieverWorker(rag=_StubRag()),
        researcher=ResearcherWorker(llm=llm, tools=ToolRegistry()),
        synthesizer=Synthesizer(llm=llm),
    )
    result = await supervisor.run("what is the rate limit?")
    assert result.citations == []
    assert result.citation_sources == {}


@pytest.mark.asyncio
async def test_neither_retrieval_nor_tools_still_produces_an_answer():
    llm = FakeLLMClient(
        decisions=[
            DelegationPlan(thought="conversational", needs_retrieval=False, needs_tools=False),
            JudgeVerdict(grounded=True, issues=[], confidence=0.5),
        ],
        generations=["I don't have evidence for that."],
    )
    supervisor = Supervisor(
        llm=llm,
        retriever=RetrieverWorker(rag=_StubRag()),
        researcher=ResearcherWorker(llm=llm, tools=ToolRegistry()),
        synthesizer=Synthesizer(llm=llm),
    )
    result = await supervisor.run("what's the capital of France?")
    assert [entry.worker for entry in result.worker_log] == ["verifier"]
    assert result.final_answer == "I don't have evidence for that."
