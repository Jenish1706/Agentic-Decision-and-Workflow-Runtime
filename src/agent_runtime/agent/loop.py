"""The explicit agent loop.

    observe state -> decide -> act -> observe -> ... -> final -> verify -> answer

This is the piece of the project that most directly *is* the thing being
demonstrated, so it is a plain, readable loop rather than a graph framework.

RESPONSIBILITIES, IN THE ORDER THEY HAPPEN IN `run()`
    1. Allocate a fresh BudgetController for this run (if a budget applies).
    2. Each step: ask the LLM for one schema-constrained ActionDecision.
    3. Act: retrieve documents, call a tool, or attempt a final answer. Every
       successful retrieval / tool call is appended to the EvidenceLedger.
    4. A final answer is only accepted if it passes deterministic
       verification (citations + numbers). A rejected answer becomes an
       observation telling the model what failed, and the loop continues.
    5. If a PolicyEngine is configured, the answer's ProposedDecision - either
       attached by the model or, when the model described a decision without
       attaching it, recovered by one follow-up extraction call - is
       evaluated against the evidence, never against the model's own claims.
    6. Build the audit trace and update conversation memory.

THE CENTRAL INVARIANT
    LLM proposes; deterministic components verify and constrain. The model
    chooses actions and writes text; budgets, verification and policy
    decisions are plain Python operating on tool outputs.
"""

import contextlib
import json
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import ValidationError

from ..llm.base import LLMClient, LLMError, Message
from ..rag.context_builder import build_context
from ..rag.pipeline import RetrievedChunk
from .actions import ActionDecision, ActionType, DecisionExtraction
from .budget import ActionBudget, BudgetController
from .evidence import EvidenceKind, EvidenceSource
from .memory.long_term import LongTermMemory
from .memory.short_term import ShortTermMemory
from .policy import DecisionContext, PolicyEngine, ProposedDecision
from .state import AgentState, Step, StepStatus, TerminationReason
from .tools.base import ToolCategory, ToolResult
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
- Prefer {gather_verb} before answering; do not guess at facts you have not actually observed \
in this conversation.
{citation_rule}- Every number in your final answer must appear in an observation above or in the question. \
If you need a derived number (a sum, difference or percentage change), compute it with the \
calculator tool first.
- If, after gathering evidence, nothing available answers the question, say so plainly \
in your final answer (e.g. "the available data does not contain this information") \
instead of guessing.
- You have at most {max_steps} steps total. Do not repeat the same action with the same \
arguments if it already gave you an answer.
{decision_rules}"""

# Only shown when retrieval exists. A tool-only agent used to receive this rule
# too; observed with qwen2.5:7b on a credit-risk question, the model imitated
# the "[1], [2]" example and appended invented tags to an answer built purely
# from SQL results, which then failed citation verification.
_CITATION_RULE = (
    "- When you give a final answer, cite retrieved documents using the exact [n] tags shown in "
    "the observations below (e.g. [1], [2]). Only cite tags you were actually given.\n"
)

_DECISION_RULES = """
Decisions:
- If the question asks whether an action should be taken (approve, pay, roll out, remediate, \
deploy, grant...), you MUST attach a "proposed_decision" to your final answer.
- Set recommended_action to the action being evaluated, even if you expect it to be refused. \
A deterministic policy engine - not you - decides whether it is allowed, blocked or needs \
approval; give your own recommendation in the answer text.
- In proposed_decision.parameters, identify what the action targets and the quantities that \
define the action itself.
- Facts about the target are checked against the evidence you gathered, not against your \
parameters, so look the relevant records up before proposing a decision.
{domain_guidance}"""

# WHY the proposal names the action under evaluation rather than the model's
# verdict: if the model proposed "decline" for a borrower who fails policy, the
# engine would correctly allow the decline - and the audit trail would never
# record that the requested approval breaks a rule, or which one. Evaluating
# the requested action keeps the rule outcome (and its reasons) in the trace,
# and lets benchmarks check the engine's verdict against the expected one.
#
# WHY the action names and parameter keys come from the policy engine
# (`PolicyEngine.guidance`, written in each domain manifest) instead of this
# prompt: this generic runtime used to list examples from all four domains
# here. Observed with qwen2.5:7b on a product-analytics rollout question, the
# model copied the operations example and proposed "rolling_restart" with
# service_name="EXP-204". The core stays domain-agnostic; each domain states
# its own vocabulary and which evidence its rules read.

_DECISION_EXTRACTION_PROMPT = """You convert an assistant's final answer into a structured \
decision proposal for a deterministic policy engine. Do not re-decide anything and do not add \
facts that are not in the answer.

Set asks_for_decision to true only if the QUESTION asks whether an action should be taken \
(approve, pay, roll out, remediate, deploy, grant...). If it does, recommended_action is the \
action being evaluated - even if the answer recommends against it - and parameters identify its \
target and the action's own quantities. If it does not, set asks_for_decision to false and leave \
the other fields empty.
{domain_guidance}"""

_RETRIEVE_ACTION_LINE = (
    '- "retrieve": search the knowledge base. Use this for any question that might be '
    "answered by internal documentation.\n"
)
_CALL_TOOL_ACTION_LINE = '- "call_tool": invoke one of the tools listed below by exact name.\n'

# Which EvidenceSource / EvidenceKind a tool's result is recorded under. Driven
# by the tool's declared category, not by guessing from its name.
_EVIDENCE_FOR_CATEGORY: dict[ToolCategory, tuple[EvidenceSource, EvidenceKind]] = {
    ToolCategory.SQL: (EvidenceSource.ANALYTICS_SQL, EvidenceKind.OBSERVATION),
    ToolCategory.STATISTICS: (EvidenceSource.STATISTICS, EvidenceKind.CALCULATION),
    ToolCategory.VISION: (EvidenceSource.VISION_VLM, EvidenceKind.OBSERVATION),
    ToolCategory.RECORD: (EvidenceSource.RECORD_LOOKUP, EvidenceKind.OBSERVATION),
    ToolCategory.CALCULATION: (EvidenceSource.CALCULATOR, EvidenceKind.CALCULATION),
}


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


def _clip(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit]


def _evidence_payload(result: ToolResult) -> Mapping[str, Any]:
    """Structured data for the ledger: the tool's own `data` when it provides
    it, otherwise its content parsed as JSON, otherwise the raw text."""
    if result.data is not None:
        return result.data
    content = result.content.strip()
    if content.startswith(("{", "[")):
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"items": parsed}
    return {"output": result.content}


class Retriever(Protocol):
    """Whatever document retrieval is under the hood. For the single agent it
    is `MCPRetriever` (mcp_retriever.py), which calls the knowledge MCP server;
    it is a Protocol rather than RagPipeline because retrieval is
    protocol-mediated, not an in-process call."""

    async def retrieve(self, query: str) -> list[RetrievedChunk]: ...


class AgentRuntime:
    """A single agentic loop over retrieval and tools.

    `rag=None` builds a tool-only agent (used as the multi-agent Researcher
    worker, which has no business doing document retrieval).
    """

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
        budget: ActionBudget | None = None,
    ):
        self._llm = llm
        self._tools = tools
        self._rag = rag
        self._short_term = short_term or ShortTermMemory()
        self._long_term = long_term
        self._max_steps = max_steps
        self._max_decision_retries = max_decision_retries
        # A plain per-run retrieval cap, used by the retrieval-budget
        # experiment. `budget` (below) supersedes it when given.
        self._max_retrievals = max_retrievals
        self._policy_engine = policy_engine
        # The default allocation for runs that don't pass their own. Only the
        # immutable ActionBudget is stored here; the mutable counter is created
        # inside run(), so no consumption can leak from one run into the next.
        self._default_budget = budget

    # ------------------------------------------------------------------ session

    def reset_session(self) -> None:
        """Forget the recent conversation (short-term memory).

        Benchmarks must call this between independent questions. Without it,
        every question after the first saw earlier questions and answers in its
        prompt, so evaluation results were measured on contaminated inputs.
        Long-term memory is deliberately untouched: it is meant to persist.
        """
        self._short_term.clear()

    # ------------------------------------------------------------------ prompt

    def _system_prompt(self, max_steps: int) -> str:
        has_retrieval = self._rag is not None
        return _BASE_SYSTEM_PROMPT.format(
            capabilities_summary="retrieval and tools" if has_retrieval else "tools",
            action_menu=(_RETRIEVE_ACTION_LINE if has_retrieval else "") + _CALL_TOOL_ACTION_LINE,
            tool_schemas=_render_tool_schemas(self._tools),
            gather_verb="retrieving or calling a tool" if has_retrieval else "calling a tool",
            citation_rule=_CITATION_RULE if has_retrieval else "",
            max_steps=max_steps,
            decision_rules=self._decision_rules(),
        )

    def _decision_rules(self) -> str:
        if self._policy_engine is None:
            return ""
        return _DECISION_RULES.format(domain_guidance=self._domain_guidance())

    def _domain_guidance(self) -> str:
        """The policy engine's own vocabulary and limits, for prompts."""
        if self._policy_engine is None:
            return ""
        parts = []
        if guidance := self._policy_engine.guidance.strip():
            parts.append(f"For this domain:\n{guidance}")
        if self._policy_engine.thresholds:
            # The model is told the real limits so its answer text can explain
            # them instead of inventing its own. Observed before this was added:
            # qwen2.5:7b wrote unsupported figures (a "0.50" limit, a "30.49%"
            # ratio) into credit answers, which numerical verification then
            # rejected, and the run spent its steps failing verification. Stating a threshold
            # is always grounded (see verify_numerical_grounding's reference_values).
            limits = ", ".join(f"{name}={value:g}" for name, value in self._policy_engine.thresholds.items())
            parts.append(f"Policy thresholds (enforced by the engine, not by you): {limits}")
        return "".join(f"\n{part}\n" for part in parts)

    def _build_messages(self, state: AgentState, memory_preamble: str, max_steps: int) -> list[Message]:
        messages = [
            Message(role="system", content=self._system_prompt(max_steps)),
            Message(role="user", content=memory_preamble + f"Question: {state.query}"),
        ]
        if state.steps:
            transcript = "\n".join(_render_step(step) for step in state.steps)
            messages.append(Message(role="user", content=f"Steps taken so far:\n{transcript}\n\nDecide the next action."))
        return messages

    def _memory_preamble(self, recent_turns, remembered_facts) -> str:
        parts = []
        if recent_turns:
            history = "\n".join(f"User: {turn.query}\nAssistant: {turn.answer}" for turn in recent_turns)
            parts.append(f"Recent conversation history:\n{history}\n")
        if remembered_facts:
            facts = "\n".join(f"- {item.text}" for item in remembered_facts)
            parts.append(f"Remembered facts about this user:\n{facts}\n")
        return "\n".join(parts) + ("\n" if parts else "")

    # ------------------------------------------------------------------ run

    async def run(self, query: str, *, budget: ActionBudget | None = None) -> AgentState:
        """Answer one question. `budget` overrides the runtime's default
        allocation for this run only (used by adaptive budgeting, which picks
        a budget per question)."""
        state = AgentState(query=query)
        citation_map: dict[str, Any] = {}

        allocation = budget or self._default_budget
        controller = BudgetController(allocation) if allocation is not None else None
        max_steps = allocation.max_steps if allocation is not None else self._max_steps

        recent_turns = self._short_term.recent_turns()
        remembered_facts = self._long_term.retrieve(query) if self._long_term else []
        memory_preamble = self._memory_preamble(recent_turns, remembered_facts)

        for step_index in range(max_steps):
            if controller is not None:
                controller.record_step()

            messages = self._build_messages(state, memory_preamble, max_steps)
            try:
                decision = await self._llm.decide(messages, ActionDecision, max_retries=self._max_decision_retries)
            except LLMError as exc:
                state.done = True
                state.termination_reason = TerminationReason.UNRECOVERABLE_ERROR
                state.final_answer = (
                    "I couldn't produce a valid next step after multiple attempts, "
                    f"so I'm stopping rather than guessing. ({exc})"
                )
                break

            if decision.action is ActionType.RETRIEVE:
                observation, status = await self._handle_retrieve(decision, citation_map, state, controller)
            elif decision.action is ActionType.CALL_TOOL:
                observation, status = await self._handle_tool_call(decision, state, controller)
            else:  # FINAL
                is_last_step = step_index >= max_steps - 1
                observation, status = await self._handle_final(decision, citation_map, state, accept=is_last_step)

            # Append before checking `done`: the step that terminates the run
            # still belongs in the trace (a past version skipped it, so every
            # successful run reported zero steps).
            state.steps.append(Step(index=step_index, decision=decision, observation=observation, status=status))
            if state.done:
                break

        if not state.done:
            state.done = True
            state.termination_reason = TerminationReason.MAX_STEPS_REACHED
            state.final_answer = state.final_answer or (
                "I reached my step limit before finding a confident, grounded answer."
            )

        if controller is not None:
            state.budget_usage = controller.usage.to_dict()
        else:
            state.budget_usage = {"steps_used": len(state.steps), "retrievals_used": len(state.retrieval_queries)}
        state.audit_trace = self._build_audit_trace(state, allocation)

        self._short_term.record(query, state.final_answer or "")
        if self._long_term is not None:
            # Deciding what to remember is a side effect of answering, not part
            # of it: if that judgement fails to parse, the answer is still good.
            with contextlib.suppress(LLMError):
                await self._long_term.maybe_store(self._llm, query, state.final_answer or "")

        return state

    def _build_audit_trace(self, state: AgentState, allocation: ActionBudget | None) -> dict[str, Any]:
        return {
            "query": state.query,
            "steps_count": len(state.steps),
            "termination_reason": state.termination_reason.value if state.termination_reason else None,
            "final_answer": state.final_answer,
            "answer_verified": state.answer_verified,
            "verification_issues": list(state.verification_issues),
            "citations": state.citations,
            "citation_sources": state.citation_sources,
            "evidence_count": len(state.evidence_ledger),
            "evidence_ledger": state.evidence_ledger.to_list(),
            "budget": allocation.to_dict() if allocation is not None else None,
            "budget_usage": state.budget_usage,
            "proposed_decision": state.proposed_decision.model_dump() if state.proposed_decision else None,
            "proposed_decision_source": state.proposed_decision_source,
            "policy_evaluation": state.policy_evaluation.to_dict() if state.policy_evaluation else None,
        }

    # ------------------------------------------------------------------ actions

    async def _handle_retrieve(
        self,
        decision: ActionDecision,
        citation_map: dict[str, Any],
        state: AgentState,
        controller: BudgetController | None,
    ) -> tuple[str, StepStatus]:
        if self._rag is None:
            return "Retrieval is not available to this agent.", StepStatus.ERROR

        # Budget checks are reported as observations, not silent no-ops, so the
        # model can answer from what it has instead of retrying forever.
        if controller is not None:
            if controller.budget.max_retrievals == 0:
                return (
                    "Document retrieval is not permitted for this task. Use a tool or answer "
                    "from the evidence you already have.",
                    StepStatus.ERROR,
                )
            if not controller.can_retrieve():
                return (
                    f"Retrieval budget exhausted: this run allows {controller.budget.max_retrievals} "
                    "search(es) and you have used them all. Answer from the evidence you already "
                    "have, or use a tool.",
                    StepStatus.ERROR,
                )
            controller.record_retrieval()
        elif self._max_retrievals is not None and len(state.retrieval_queries) >= self._max_retrievals:
            return (
                f"Retrieval budget exhausted: this run allows {self._max_retrievals} "
                "search(es) and you have used them all. Answer from the evidence you already "
                "have, or use a tool.",
                StepStatus.ERROR,
            )

        # Fall back to the original question when the model omits a query.
        effective_query = decision.query or state.query
        results = await self._rag.retrieve(effective_query)
        state.retrieval_queries.append(effective_query)
        state.retrieved_chunk_ids.extend(result.chunk.chunk_id for result in results)

        if not results:
            return "No relevant documents found for that query.", StepStatus.OK
        context_text, new_citations = build_context(
            [(result.chunk, result.score) for result in results], start_index=len(citation_map) + 1
        )
        citation_map.update(new_citations)

        state.evidence_ledger.add(
            source=EvidenceSource.KNOWLEDGE_RAG,
            kind=EvidenceKind.RETRIEVAL,
            query_or_input=effective_query,
            data={
                "query": effective_query,
                "chunks": [
                    {
                        "chunk_id": result.chunk.chunk_id,
                        "doc_id": result.chunk.doc_id,
                        "text": result.chunk.text,
                        "score": result.score,
                    }
                    for result in results
                ],
            },
            provenance=f"rag:{len(results)}_chunks",
            summary=_clip(context_text),
        )
        return context_text, StepStatus.OK

    async def _handle_tool_call(
        self,
        decision: ActionDecision,
        state: AgentState,
        controller: BudgetController | None,
    ) -> tuple[str, StepStatus]:
        tool_name = decision.tool_name or ""
        tool = self._tools.get(tool_name)
        category = tool.category if tool is not None else None

        # Budgeted categories consume their allowance when attempted, even if
        # the call then fails: a failed query is still exploration spent.
        if controller is not None and category is ToolCategory.SQL:
            if not controller.can_query_sql():
                return (
                    f"SQL query budget exhausted: this run allows {controller.budget.max_sql_queries} "
                    "SQL queries and you have used them all. Answer from the evidence you already have.",
                    StepStatus.ERROR,
                )
            controller.record_sql_query()
        elif controller is not None and category is ToolCategory.VISION:
            if not controller.can_call_vision():
                return (
                    f"Document/chart extraction budget exhausted: this run allows "
                    f"{controller.budget.max_vision_calls} extraction call(s). Answer from the evidence "
                    "you already have.",
                    StepStatus.ERROR,
                )
            controller.record_vision_call()

        result = await self._tools.execute(tool_name, decision.tool_arguments)
        if not result.ok:
            return f"Tool error: {result.error}", StepStatus.ERROR

        source, kind = _EVIDENCE_FOR_CATEGORY[category or ToolCategory.RECORD]
        state.evidence_ledger.add(
            source=source,
            kind=kind,
            query_or_input=json.dumps(decision.tool_arguments, sort_keys=True, default=str),
            data=_evidence_payload(result),
            provenance=f"tool:{tool_name}",
            summary=_clip(result.content),
        )
        return result.content, StepStatus.OK

    async def _handle_final(
        self,
        decision: ActionDecision,
        citation_map: dict[str, Any],
        state: AgentState,
        *,
        accept: bool,
    ) -> tuple[str, StepStatus]:
        draft = decision.answer or ""
        has_tool_evidence = any(item.kind is not EvidenceKind.RETRIEVAL for item in state.evidence_ledger.all())
        citation_check = verify_citations(draft, citation_map, has_non_retrieval_evidence=has_tool_evidence)
        numbers_check = verify_numerical_grounding(
            draft,
            state.evidence_ledger,
            query=state.query,
            reference_values=self._policy_engine.thresholds if self._policy_engine is not None else None,
        )
        verified = citation_check.grounded and numbers_check.valid

        if not verified and not accept:
            if not citation_check.grounded:
                issues = "; ".join(citation_check.issues)
                return (
                    f"Verification failed: {issues}. Your answer text must itself contain the [n] tags "
                    "inline, next to the claims they support - e.g. 'The limit is 5,000 events/sec [1].' "
                    "Restating the fact without the bracketed tag attached does not count as citing it. "
                    "If the tags truly don't exist yet, retrieve or call a tool first.",
                    StepStatus.ERROR,
                )
            issues = "; ".join(numbers_check.issues)
            return (
                f"Verification failed: {issues}. Every number must match an observation above or the "
                "question. Remove numbers you cannot support, or call a tool (the calculator for "
                "arithmetic) to obtain them.",
                StepStatus.ERROR,
            )

        # Accepted: either verified, or the last step's best-effort answer. The
        # difference is recorded, never erased.
        state.final_answer = draft
        state.answer_verified = verified
        state.verification_issues = [*citation_check.issues, *numbers_check.issues]
        state.citations = citation_check.used_tags
        state.citation_sources = {
            tag: citation_map[tag].doc_id for tag in citation_check.used_tags if tag in citation_map
        }
        if decision.proposed_decision is not None:
            state.proposed_decision = decision.proposed_decision
            state.proposed_decision_source = "final_step"
        elif self._policy_engine is not None:
            state.proposed_decision = await self._extract_decision(state.query, draft)
            state.proposed_decision_source = "extracted" if state.proposed_decision is not None else None

        if self._policy_engine is not None and state.proposed_decision is not None:
            context = DecisionContext(
                subject_or_entity=state.query,
                evidence_ledger=state.evidence_ledger,
                # An unverified answer must not auto-approve a decision; the
                # engine escalates when this is False.
                metadata={"answer_verified": verified},
            )
            state.policy_evaluation = self._policy_engine.evaluate(state.proposed_decision, context)

        state.done = True
        state.termination_reason = TerminationReason.FINAL_ANSWER
        if verified:
            return "Accepted as final answer.", StepStatus.OK
        return (
            "Accepted as a best-effort final answer on the last step; it did NOT pass verification: "
            + "; ".join(state.verification_issues),
            StepStatus.ERROR,
        )

    async def _extract_decision(self, query: str, answer: str) -> ProposedDecision | None:
        """Recover a decision the model described but did not attach.

        Returns None when the question does not ask for a decision, when the
        extraction is unusable, or when the call fails: a missing proposal
        leaves the answer intact and simply means no policy verdict - the same
        outcome as before this fallback existed, never a crash.
        """
        messages = [
            Message(role="system", content=_DECISION_EXTRACTION_PROMPT.format(domain_guidance=self._domain_guidance())),
            Message(role="user", content=f"Question: {query}\n\nFinal answer: {answer}"),
        ]
        try:
            extraction = await self._llm.decide(messages, DecisionExtraction, max_retries=self._max_decision_retries)
        except LLMError:
            return None
        if not extraction.asks_for_decision or not extraction.recommended_action.strip():
            return None
        try:
            return ProposedDecision(
                intent=f"Decision requested by: {query}"[:300],
                recommended_action=extraction.recommended_action.strip(),
                parameters=extraction.parameters,
                reasoning=extraction.reasoning,
                confidence=extraction.confidence,  # a percentage-style value is rescaled by the model's validator
            )
        except ValidationError:
            return None
