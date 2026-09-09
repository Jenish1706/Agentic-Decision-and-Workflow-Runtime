"""The explicit agent loop: observe state -> decide -> act -> observe ->
... -> final -> verify -> answer. This is the one piece of this project
that most directly is the thing being demonstrated, so it is written as a
plain, readable while-loop rather than hidden behind a graph/orchestration
framework.
"""

from typing import Protocol

from ..llm.base import LLMClient, LLMError, Message
from ..rag.context_builder import build_context
from ..rag.pipeline import RetrievedChunk
from .actions import ActionDecision, ActionType
from .budget import BudgetController
from .evidence import EvidenceKind, EvidenceLedger, EvidenceSource
from .memory.long_term import LongTermMemory
from .memory.short_term import ShortTermMemory
from .policy import DecisionContext, PolicyEngine, ProposedDecision
from .state import AgentState, Step, StepStatus, TerminationReason
from .tools.registry import ToolRegistry
from .verification import verify_citations, verify_numerical_grounding

_BASE_SYSTEM_PROMPT = """You are an assistant that answers questions step by step using \
{capabilities_summary} rather than your own memorized knowledge.

At every step you must choose exactly one action:
{action_menu}
- "final": give your final answer now.

Tools available to you:
{tool_schemas}

Rules:
- Prefer retrieving or calling a tool before answering from documents/records; do not \
guess at facts you have not actually observed in this conversation.
- When you give a final answer, cite evidence using the exact [n] tags shown in the \
observations below (e.g. [1], [2]). Only cite tags you were actually given.
- If, after gathering evidence, nothing available answers the question, say so plainly \
in your final answer (e.g. "the knowledge base does not contain this information") \
instead of guessing.
- You have at most {max_steps} steps total. Do not repeat the same action with the same \
arguments if it already gave you an answer.
"""

_RETRIEVE_ACTION_LINE = (
    '- "retrieve": search the knowledge base. Use this for any question that might be '
    "answered by internal documentation.\n"
)
_CALL_TOOL_ACTION_LINE = '- "call_tool": invoke one of the tools listed below by exact name.\n'


def _render_tool_schemas(tools: ToolRegistry) -> str:
    schemas = tools.list_schemas()
    if not schemas:
        return "(no tools registered)"
    lines = []
    for schema in schemas:
        lines.append(f"- {schema['name']} [{schema['origin']}]: {schema['description']}")
        lines.append(f"  parameters: {schema['parameters']}")
    return "\n".join(lines)


def _render_step(step: Step) -> str:
    decision = step.decision
    if decision.action is ActionType.RETRIEVE:
        return f"Step {step.index} - retrieve('{decision.query}') -> {step.observation}"
    if decision.action is ActionType.CALL_TOOL:
        return f"Step {step.index} - call_tool('{decision.tool_name}', {decision.tool_arguments}) -> {step.observation}"
    return f"Step {step.index} - final attempt -> {step.observation}"


class Retriever(Protocol):
    """Whatever `rag` retrieval actually is under the hood - see
    mcp_retriever.py's MCPRetriever, the one implementation used here. Kept
    as a Protocol (not RagPipeline directly) because 'retrieve' is
    MCP-mediated for the single agent, not a direct in-process call."""

    async def retrieve(self, query: str) -> list[RetrievedChunk]: ...


class AgentRuntime:
    """A single agentic loop over retrieval + tools. `rag=None` builds a
    tool-only agent (used as the multi-agent Researcher worker, which has no
    business doing document retrieval)."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        rag: Retriever | None = None,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
        max_steps: int = 8,
        max_decision_retries: int = 2,
        max_retrievals: int | None = None,
        policy_engine: PolicyEngine | None = None,
        budget_controller: BudgetController | None = None,
    ):
        self._llm = llm
        self._tools = tools
        self._rag = rag
        self._short_term = short_term or ShortTermMemory()
        self._long_term = long_term
        self._max_steps = max_steps
        self._max_decision_retries = max_decision_retries
        # None = no explicit retrieval budget; retrieval is bounded only by
        # max_steps, which is what this project shipped with. An integer
        # enforces a hard per-run cap, used by the Retrieval Budget &
        # Search Efficiency experiment (scripts/run_budget_experiment.py).
        self._max_retrievals = max_retrievals
        self._policy_engine = policy_engine
        self._budget_controller = budget_controller

    def _system_prompt(self) -> str:
        action_menu = _CALL_TOOL_ACTION_LINE
        capabilities = "tools"
        if self._rag is not None:
            action_menu = _RETRIEVE_ACTION_LINE + action_menu
            capabilities = "retrieval and tools"
        return _BASE_SYSTEM_PROMPT.format(
            capabilities_summary=capabilities,
            action_menu=action_menu,
            tool_schemas=_render_tool_schemas(self._tools),
            max_steps=self._max_steps,
        )

    def _build_messages(self, state: AgentState, memory_preamble: str) -> list[Message]:
        messages = [Message(role="system", content=self._system_prompt())]
        preamble = memory_preamble + f"Question: {state.query}"
        if not state.steps:
            messages.append(Message(role="user", content=preamble))
            return messages

        transcript = "\n".join(_render_step(s) for s in state.steps)
        messages.append(Message(role="user", content=preamble))
        messages.append(
            Message(
                role="user",
                content=f"Steps taken so far:\n{transcript}\n\nDecide the next action.",
            )
        )
        return messages

    def _memory_preamble(self, recent_turns, remembered_facts) -> str:
        parts = []
        if recent_turns:
            history = "\n".join(f"User: {t.query}\nAssistant: {t.answer}" for t in recent_turns)
            parts.append(f"Recent conversation history:\n{history}\n")
        if remembered_facts:
            facts = "\n".join(f"- {item.text}" for item in remembered_facts)
            parts.append(f"Remembered facts about this user:\n{facts}\n")
        return "\n".join(parts) + ("\n" if parts else "")

    async def run(self, query: str) -> AgentState:
        state = AgentState(query=query)
        citation_map: dict[str, object] = {}

        recent_turns = self._short_term.recent_turns()
        remembered_facts = self._long_term.retrieve(query) if self._long_term else []
        memory_preamble = self._memory_preamble(recent_turns, remembered_facts)

        effective_max_steps = (
            self._budget_controller.budget.max_steps
            if self._budget_controller is not None
            else self._max_steps
        )

        for step_index in range(effective_max_steps):
            if self._budget_controller is not None:
                if not self._budget_controller.can_step():
                    break
                self._budget_controller.record_step()

            messages = self._build_messages(state, memory_preamble)
            try:
                decision = await self._llm.decide(
                    messages, ActionDecision, max_retries=self._max_decision_retries
                )
            except LLMError as exc:
                state.done = True
                state.termination_reason = TerminationReason.UNRECOVERABLE_ERROR
                state.final_answer = (
                    "I couldn't produce a valid next step after multiple attempts, "
                    f"so I'm stopping rather than guessing. ({exc})"
                )
                break

            if decision.action is ActionType.RETRIEVE:
                observation, status = await self._handle_retrieve(decision, citation_map, state)
            elif decision.action is ActionType.CALL_TOOL:
                observation, status = await self._handle_tool_call(decision, state)
            else:  # FINAL
                is_last_step = step_index >= effective_max_steps - 1
                observation, status = self._handle_final(decision, citation_map, state, accept=is_last_step)

            # Appended before checking state.done (not before, as a prior version
            # did): the step that actually terminates the run still belongs in the
            # trace - skipping it made every successful run report zero steps.
            state.steps.append(Step(index=step_index, decision=decision, observation=observation, status=status))
            if state.done:
                break

        if not state.done:
            state.done = True
            state.termination_reason = TerminationReason.MAX_STEPS_REACHED
            state.final_answer = state.final_answer or (
                "I reached my step limit before finding a confident, grounded answer."
            )

        # Build comprehensive audit trace
        state.audit_trace = {
            "query": state.query,
            "steps_count": len(state.steps),
            "termination_reason": state.termination_reason.value if state.termination_reason else None,
            "final_answer": state.final_answer,
            "citations": state.citations,
            "citation_sources": state.citation_sources,
            "evidence_count": len(state.evidence_ledger),
            "evidence_ledger": state.evidence_ledger.to_list(),
            "budget_usage": (
                self._budget_controller.usage.to_dict()
                if self._budget_controller is not None
                else {
                    "steps_used": len(state.steps),
                    "retrievals_used": len(state.retrieval_queries),
                }
            ),
            "proposed_decision": (
                state.proposed_decision.model_dump()
                if state.proposed_decision is not None
                else None
            ),
            "policy_evaluation": (
                state.policy_evaluation.to_dict()
                if state.policy_evaluation is not None
                else None
            ),
        }

        self._short_term.record(query, state.final_answer or "")
        if self._long_term is not None:
            # Deciding what to remember is a side effect of answering, not part
            # of it: if that judgement call fails to parse, the answer we
            # already have is still good, and losing it to a memory-write
            # failure would be absurd.
            try:
                await self._long_term.maybe_store(self._llm, query, state.final_answer or "")
            except LLMError:
                pass

        return state

    async def _handle_retrieve(
        self, decision: ActionDecision, citation_map: dict, state: AgentState
    ) -> tuple[str, StepStatus]:
        if self._rag is None:
            return "Retrieval is not available to this agent.", StepStatus.ERROR

        # Check retrieval limits (budget_controller or max_retrievals)
        if self._budget_controller is not None:
            if not self._budget_controller.can_retrieve():
                return (
                    f"Retrieval budget exhausted: this run allows {self._budget_controller.budget.max_retrievals} "
                    "search(es) and you have used them all. Answer from the evidence you "
                    "already have, or use a tool.",
                    StepStatus.ERROR,
                )
            self._budget_controller.record_retrieval()
        elif self._max_retrievals is not None and len(state.retrieval_queries) >= self._max_retrievals:
            # Reported as an observation rather than a silent no-op, so the
            # model can react (answer from what it has, or use a tool) instead
            # of retrying a call that will never succeed.
            return (
                f"Retrieval budget exhausted: this run allows {self._max_retrievals} "
                "search(es) and you have used them all. Answer from the evidence you "
                "already have, or use a tool.",
                StepStatus.ERROR,
            )

        # Falls back to the original question when the model omits a search
        # query - a reasonable default, and one that avoids failing an entire
        # run over a single missing field (see actions.py's docstring).
        effective_query = decision.query or state.query
        results = await self._rag.retrieve(effective_query)
        state.retrieval_queries.append(effective_query)
        state.retrieved_chunk_ids.extend(r.chunk.chunk_id for r in results)

        if not results:
            return "No relevant documents found for that query.", StepStatus.OK
        context_text, new_citations = build_context(
            [(r.chunk, r.score) for r in results], start_index=len(citation_map) + 1
        )
        citation_map.update(new_citations)

        # Record in EvidenceLedger
        state.evidence_ledger.add(
            source=EvidenceSource.KNOWLEDGE_RAG,
            kind=EvidenceKind.RETRIEVAL,
            query_or_input=effective_query,
            data={
                "query": effective_query,
                "chunks": [
                    {
                        "chunk_id": r.chunk.chunk_id,
                        "doc_id": r.chunk.doc_id,
                        "text": r.chunk.text,
                        "score": r.score,
                    }
                    for r in results
                ],
            },
            provenance=f"rag:{len(results)}_chunks",
            summary=context_text[:200] if len(context_text) > 200 else context_text,
        )

        return context_text, StepStatus.OK

    async def _handle_tool_call(
        self, decision: ActionDecision, state: AgentState | None = None
    ) -> tuple[str, StepStatus]:
        tool_name = decision.tool_name or ""
        is_sql = "sql" in tool_name or tool_name in ("execute_sql", "describe_tables")

        if is_sql and self._budget_controller is not None:
            if not self._budget_controller.can_query_sql():
                return (
                    f"SQL query budget exhausted: this run allows {self._budget_controller.budget.max_sql_queries} "
                    "SQL queries and you have used them all. Answer from the evidence you already have.",
                    StepStatus.ERROR,
                )
            self._budget_controller.record_sql_query()

        result = await self._tools.execute(tool_name, decision.tool_arguments)
        if not result.ok:
            return f"Tool error: {result.error}", StepStatus.ERROR

        if state is not None:
            if is_sql:
                source = EvidenceSource.ANALYTICS_SQL
                kind = EvidenceKind.OBSERVATION
            elif "significance" in tool_name or tool_name in ("test_significance", "statistics"):
                source = EvidenceSource.STATISTICS
                kind = EvidenceKind.CALCULATION
            elif tool_name == "calculator":
                source = EvidenceSource.RECORD_LOOKUP
                kind = EvidenceKind.CALCULATION
            elif any(k in tool_name for k in ("vision", "ocr", "chart", "image", "vlm", "document")):
                source = EvidenceSource.VISION_VLM
                kind = EvidenceKind.OBSERVATION
            else:
                source = EvidenceSource.RECORD_LOOKUP
                kind = EvidenceKind.OBSERVATION

            raw_data = result.data
            if raw_data is None:
                if result.content.strip().startswith(("{", "[")):
                    try:
                        import json
                        parsed = json.loads(result.content)
                        raw_data = parsed if isinstance(parsed, dict) else {"items": parsed}
                    except Exception:
                        raw_data = {"output": result.content}
                else:
                    raw_data = {"output": result.content}

            state.evidence_ledger.add(
                source=source,
                kind=kind,
                query_or_input=str(decision.tool_arguments),
                data=raw_data,
                provenance=f"tool:{tool_name}",
                summary=result.content[:200] if len(result.content) > 200 else result.content,
            )

        return result.content, StepStatus.OK

    def _handle_final(
        self, decision: ActionDecision, citation_map: dict, state: AgentState, *, accept: bool
    ) -> tuple[str, StepStatus]:
        draft = decision.answer or ""
        verification = verify_citations(draft, citation_map)
        num_verification = verify_numerical_grounding(draft, state.evidence_ledger)

        citation_ok = verification.grounded or accept
        num_ok = num_verification.valid or accept

        if citation_ok and num_ok:
            state.final_answer = draft
            state.citations = verification.used_tags
            state.citation_sources = {
                tag: citation_map[tag].doc_id for tag in verification.used_tags if tag in citation_map
            }

            if decision.proposed_decision is not None:
                state.proposed_decision = decision.proposed_decision

            # Deterministic policy evaluation if policy engine is configured
            if self._policy_engine is not None and state.proposed_decision is not None:
                context = DecisionContext(
                    subject_or_entity=state.query,
                    evidence_ledger=state.evidence_ledger,
                )
                state.policy_evaluation = self._policy_engine.evaluate(state.proposed_decision, context)

            state.done = True
            state.termination_reason = TerminationReason.FINAL_ANSWER
            return "Accepted as final answer.", StepStatus.OK

        if not citation_ok:
            issues = "; ".join(verification.issues)
            return (
                f"Verification failed: {issues}. Your answer text must itself contain the [n] tags "
                "inline, next to the claims they support - e.g. 'The limit is 5,000 events/sec [1].' "
                "Restating the fact without the bracketed tag attached does not count as citing it. "
                "If the tags truly don't exist yet, retrieve or call a tool first.",
                StepStatus.ERROR,
            )

        issues = "; ".join(num_verification.issues)
        return (
            f"Verification failed: {issues}. Quantitative figures in your answer do not reconcile with "
            "the evidence ledger. Call analytical tools or provide grounded numbers.",
            StepStatus.ERROR,
        )
