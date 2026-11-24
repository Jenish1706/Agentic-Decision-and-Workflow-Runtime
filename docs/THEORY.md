# Theory & Interview Reference

Every answer here points at real code, not just a concept - the point of
this document is to be able to say "here's where that's actually decided"
during an interview, not to recite definitions.

## RAG

**What is RAG?** Instead of asking an LLM to answer from what it memorized
during training, you retrieve relevant text from an external corpus at
query time and put it in the prompt, so the model generates from evidence
it can be checked against. See [rag/pipeline.py](src/agent_runtime/rag/pipeline.py).

**Why does retrieval help hallucination?** It doesn't eliminate it - it
gives the model something concrete to ground claims in, and gives *you* a
mechanism (citations) to check whether it did. [agent/verification.py](src/agent_runtime/agent/verification.py)
`verify_citations` is exactly this check: does every `[n]` tag in the
answer trace back to a chunk that was actually retrieved?

**How does vector similarity work? How do embeddings work conceptually?**
An embedding model maps text to a point in a high-dimensional space such
that semantically similar text ends up at nearby points. "Nearby" is
measured here as cosine similarity - the cosine of the angle between two
vectors, which (once vectors are L2-normalized) is just their dot product.
See [rag/embeddings.py](src/agent_runtime/rag/embeddings.py) (`normalize_embeddings=True`)
and [rag/vector_index.py](src/agent_runtime/rag/vector_index.py) (`self._vectors @ query_vector`).

**Why can retrieval return irrelevant documents?** A bi-encoder embeds the
query and each chunk *independently* - the query vector never actually
attends to a specific chunk's content, so two texts that share vocabulary
or topic can score as similar even when one doesn't actually answer the
other's question. This is precisely what reranking (below) is for.

**Why rerank?** A cross-encoder scores a `(query, chunk)` pair *jointly* -
the model sees both texts at once and can attend across them - which is
far more accurate at judging true relevance, but too slow to run over an
entire corpus. So: bi-encoder for a fast shortlist, cross-encoder to
re-order just that shortlist. See [rag/reranker.py](src/agent_runtime/rag/reranker.py)
and the naive-vs-reranked comparison in [EVALUATION.md](EVALUATION.md).

**What is chunking? How does chunk size affect retrieval?** Splitting
documents into smaller, topically coherent spans before embedding them,
because a whole multi-topic document embeds into one vector that
represents none of its parts well. Too small and a chunk loses context
needed to make sense of it (or to answer a question that spans two
sentences); too large and it dilutes toward a vague topic average, hurting
similarity to any specific question. See the docstring in
[rag/chunking.py](src/agent_runtime/rag/chunking.py) for the size/overlap
tradeoff specifically, and [test_chunking.py](tests/test_chunking.py) for
the boundary cases (empty text, one long paragraph, overlap behavior).

**Why top-k retrieval?** Retrieval is a ranking, not a classifier - there's
no natural threshold that separates "relevant" from "not," so you take a
fixed number of the best-scoring candidates and let reranking (and the
LLM's own judgment) narrow further.

## Agents

**What makes an LLM system an agent?** Per [Anthropic's framing](https://www.anthropic.com/engineering/building-effective-agents),
the line is whether the LLM dynamically directs its own process and tool
use, versus following a predetermined code path. [agent/loop.py](src/agent_runtime/agent/loop.py)'s
`AgentRuntime.run()` is the former: the model decides, at every step,
whether to retrieve, call a tool, or answer - the code doesn't hard-code
that sequence.

**What is an agent loop? What is state? What is an observation?** The loop
is `decide -> act -> observe -> update state`, repeated until a final
answer is accepted or the step bound is hit. State
(`agent/state.py`'s `AgentState`) is everything accumulated *this run*: the
query, every step taken, the evidence ledger, whether it's finished. An
observation is the result of an action - retrieved text, a tool's return
value, a tool's error, or a verification failure - appended to state so the
*next* decision can see it.

**What is tool calling? How does an agent decide which tool to use?** The
LLM is shown every registered tool's name, description, and parameter
schema (rendered into the system prompt by `_render_tool_schemas` in
loop.py) and asked to choose. This project does not use any provider's
native tool-calling API for that choice - see ARCHITECTURE.md's "Why a
hand-rolled agent loop" - the model's decision is schema-constrained JSON
(`ActionDecision`, [agent/actions.py](src/agent_runtime/agent/actions.py))
that this code parses and dispatches itself.

**How do you prevent infinite loops and runaway cost?** A hard step bound -
the loop is a `for` over `range(max_steps)`, not a `while True` - plus
per-run budgets for retrievals, SQL queries and document extraction
([agent/budget.py](src/agent_runtime/agent/budget.py)). A budget is created
fresh for every run; an exhausted one becomes an observation ("SQL query
budget exhausted") so the model answers with what it has. If no grounded
final answer arrives, the run ends with `TerminationReason.MAX_STEPS_REACHED`.

**What happens when a tool fails?** [agent/tools/registry.py](src/agent_runtime/agent/tools/registry.py)'s
`execute()` first checks the arguments against the tool's signature, then
runs it; bad arguments and the tool's own exceptions both come back as
`ToolResult(ok=False, error=...)` - never an exception that would crash the
loop. (Checking the signature *before* the call matters: catching
`TypeError` around the whole call misreported a bug inside a tool as "bad
arguments".) The error text becomes the next observation. If the LLM itself
never produces schema-valid JSON after retries, the loop catches `LLMError`
and terminates with `UNRECOVERABLE_ERROR`. A dead Ollama server is
deliberately *not* converted into `LLMError` - see `llm/ollama_client.py` -
so an outage crashes visibly instead of producing a benchmark full of
plausible-looking failures.

## Multi-agent

**Why multiple agents? Why not one?** Per Anthropic's guidance (quoted in
ARCHITECTURE.md): start with a single agent, and add multi-agent
orchestration only where it demonstrably helps - specialization
(routing a sub-task to a worker built for it) and separated
generation/verification (evaluator-optimizer). This project's Supervisor
([multi_agent/supervisor.py](src/agent_runtime/multi_agent/supervisor.py))
exists specifically to demonstrate and measure that trade-off against the
single-agent path - see [EVALUATION.md](EVALUATION.md) for the actual
comparison and its caveats.

**How do agents communicate? What does a supervisor do?** Here,
communication is a Python function call / return value, not a
message-passing protocol - the Supervisor calls `RetrieverWorker.run()`
and `ResearcherWorker.run()` directly and collects their outputs into
`evidence_text` / `tool_observations`, then hands both to the Synthesizer.
The Supervisor's own job is exactly one decision (`DelegationPlan`: does
this need retrieval, does it need tools) - it does not answer the question
itself.

**What are the disadvantages of multi-agent systems?** More LLM calls per
query (higher latency and cost - the Researcher worker alone can add
several), more places for something to go subtly wrong (a worker's
output can be wrong in a way that isn't caught until verification), and
more state to keep straight. A concrete example from this codebase: the
Researcher worker reuses one `AgentRuntime`, which keeps conversation
memory, so every delegated task used to see the previous tasks' questions
and answers until it started resetting its session per task.

## Memory

**Context vs. state vs. memory vs. retrieval?** Four different lifetimes
and mechanisms, easy to conflate:
- **State** ([agent/state.py](src/agent_runtime/agent/state.py)) - this
  run's scratchpad. Exists only inside one `AgentRuntime.run()` call, gone
  after.
- **Context** - not a class, a *concept*: whatever text actually gets
  assembled into the next LLM prompt (`_build_messages` in loop.py). It's
  built fresh from state + memory every step.
- **Memory** ([agent/memory/](src/agent_runtime/agent/memory/)) - survives
  *across* runs. Short-term = recent turns in this session; long-term =
  persisted to disk, retrieved by similarity.
- **Retrieval** - the general mechanism (embed a query, search an index).
  RAG and long-term memory both use it, over different collections.

**Short-term vs. long-term memory?** [short_term.py](src/agent_runtime/agent/memory/short_term.py)
is the last N `(query, answer)` pairs in a bounded deque, injected verbatim -
no retrieval, no persistence. [long_term.py](src/agent_runtime/agent/memory/long_term.py)
is embedded and persisted to `data/long_term_memory.json`, and survives a
process restart.

**When is memory a bug rather than a feature?** When the next question is
unrelated. Recent turns are injected into every prompt, so benchmark
questions, API requests from different users, and delegated sub-tasks must
each start from an empty session (`AgentRuntime.reset_session()`). The
evaluation harnesses originally did not reset, so each benchmark question
saw the previous ones; see EVALUATION.md for what that means for the
published numbers.

**How is memory retrieved?** By the exact same mechanism as document RAG -
`LongTermMemory` holds its own `Embedder` + `VectorIndex` instance, over a
much smaller collection of remembered facts rather than document chunks -
with a similarity floor, so a memory is only injected when it is related to
the question. Without the floor, top-k retrieval over two stored memories
returned both of them for every question.

**When should information be stored? Why isn't the whole conversation
automatically memory?** Because most of a conversation is not durable -
routine questions and their answers aren't useful to recall in an unrelated
future conversation. `LongTermMemory.maybe_store()` puts this decision to
an explicit LLM call (`_MemoryWriteDecision`) after every turn, asking
specifically whether *this* exchange contains a preference or constraint
worth keeping - see [test_memory.py](tests/test_memory.py) for both outcomes.

## MCP

See [MCP.md](MCP.md) for the full write-up. Short version: MCP solves tool
*discovery and invocation across a process boundary* - what problem it
solves, client vs. server, why schemas, tool metadata, and local-vs-protocol
tool contrast are all covered there with direct code references.

## Evaluation

**How do you know your RAG system works? How do you evaluate retrieval
separately from generation?** By scoring the retriever against known
ground truth (`expected_doc_ids` per benchmark query) *independent* of
whatever the LLM does with the result - Recall@k and MRR
([evaluation/rag_metrics.py](src/agent_runtime/evaluation/rag_metrics.py))
only look at which chunks came back, never at the final answer text.
Groundedness and citation correctness are the separate, generation-side
check: given what *was* retrieved, did the answer actually stick to it.

**How do you evaluate agents?** Task success (does the final answer match
the expected answer, judged by a second LLM call rather than exact string
match - see `llm_judge_task_success` in [agent/llm_judge.py](src/agent_runtime/agent/llm_judge.py)),
tool selection accuracy, unnecessary tool calls, step count, expected-number
recall (are the benchmark's expected figures actually in the answer), policy
alignment, and failure rate - a run fails if it never reached a final answer
*or* only reached an unverified one. See
[evaluation/agent_metrics.py](src/agent_runtime/evaluation/agent_metrics.py).

**What makes an evaluation number trustworthy?** Mostly things this project
got wrong first and then fixed: independent questions must not share
conversation memory; a judge that fails to answer must be excluded, not
silently counted as a pass or a fail; "numerical correctness" must compare
against expected values rather than merely check that numbers exist
somewhere in evidence; and an average must only include rows the metric
applies to (a question that expects no decision is not "policy aligned").

**What failure modes did you observe?** See [EVALUATION.md](EVALUATION.md)'s
"Observed failure modes" section for what actually happened running this
project against a real model - not a hypothetical list.

**Which of your results were negative or surprising?** Reranking produced no
measurable improvement (corpus too small for the bi-encoder's top-k to be
polluted); the multi-agent path was *faster* than the single agent; and
capping the single agent at one retrieval reproduced most of the multi-agent
advantage. Those numbers were measured before the session-isolation fix
(earlier questions leaked into later prompts), so they are hypotheses to
re-measure rather than settled results - EVALUATION.md says exactly what is
affected.

**What is a retrieval budget, and does your system have one?** Two explicit
per-run caps exist: `AgentRuntime(max_retrievals=...)`, the plain cap the
retrieval-budget experiment varies, and `ActionBudget.max_retrievals`,
part of the steps / retrievals / SQL / extraction allocation used by adaptive
budgeting. There is also an implicit per-call bound (`top_k_retrieve`), and
no context-token bound at all. Search *efficiency* - redundancy, repeated
queries, chunks consumed - is measured in
[evaluation/retrieval_metrics.py](src/agent_runtime/evaluation/retrieval_metrics.py).

## Decision & Workflow Runtime

**Why separate the LLM proposal from deterministic policy enforcement?**
LLMs are probabilistic and susceptible to prompt injection, drift, and
hallucinated justifications. If an agent evaluates its own actions against
policy, it can talk itself into an exception. The runtime's invariant is
*LLM proposes; deterministic code verifies and enforces*: a
`ProposedDecision` is evaluated by `PolicyEngine` (`agent/policy.py`) using
pure Python functions with no LLM in the loop.

**If the model proposes, what stops it from proposing its own facts?**
This is the most important question about the design, because the original
rules failed it. They read facts from the proposal's parameters first, so a
model could write `{"override_authorized": true}` and lift a production
freeze, or `{"credit_score": 800}` and approve a 610 borrower. Rules now use
[agent/facts.py](src/agent_runtime/agent/facts.py): the proposal may only
name the *target* and the action's *own quantities*; every fact about the
target is resolved from the evidence ledger, scoped to that target's id.
Evidence may *restrict* a decision on weak grounds (any violating value, even
unattributed) but *permit* one only on strong grounds (one agreed value
attributed to the target), and a fact nobody looked up means
`REQUIRES_APPROVAL`, not "no violation".

**What if the model never attaches a decision at all?** Measured against
qwen2.5:7b, it usually doesn't: in the first two smoke runs it attached a
`proposed_decision` in 1 of 10 runs, describing decisions in prose instead
and leaving the optional field null. When a
policy engine is configured and no decision is attached, the loop makes one
follow-up call with an all-required schema (`DecisionExtraction`) to recover
it from the answer. The extraction is still only a proposal - the engine
judges it against evidence - and the audit trace records
`proposed_decision_source` as `final_step` or `extracted`.

**Why is an append-only Evidence Ledger different from agent state or memory?**
Agent state is an ephemeral scratchpad; long-term memory is semantic and
heuristic. The `EvidenceLedger` (`agent/evidence.py`) is an append-only
record of exact tool observations with unique ids and timestamps, and its
payloads are deep-frozen copies - so neither the model's later steps nor a
tool mutating its own result afterwards can rewrite an observation. (A
frozen dataclass alone does not do this: it still held a mutable dict, which
the original version exposed.)

**Why deterministic numerical verification instead of an LLM judge?**
Small LLMs are weak at arithmetic and scale (0.0293 is 2.93%, not 29.3%;
percentage points are not relative percent change). `verify_numerical_grounding`
(`agent/verification.py`) extracts claims with regular expressions, parses
units (%, pp, currency, counts), and requires each to match evidence within
the precision it was written at ("2.9%" accepts 2.85-2.95), at a compatible
unit, in a consistent direction. Measured on 2,000 random fabricated claims
against a realistic ledger, the previous version (flat ±1.5 pp tolerance,
any-pair derivations) accepted 50.9%; the current one accepts 6.7% - mostly
values that coincide with real ones - and none of 2,000 values chosen to lie
outside every real value and gap. What it cannot check is meaning: swapping
two real numbers between concepts passes, which is why decisions never read
the answer text.

**How does policy precedence work?**
$$\text{BLOCKED} > \text{REQUIRES\_APPROVAL} > \text{ALLOWED}$$
Any `BLOCKED` result blocks; otherwise any `REQUIRES_APPROVAL` escalates;
only when every rule allows is the proposal `ALLOWED`. So adding a rule can
only make the system stricter, and an "allow" rule can never override
anything. The engine also fails closed: a crashing rule counts as `BLOCKED`,
an internally inconsistent rule result is read restrictively, and a decision
attached to an answer that failed verification cannot be auto-approved.

**How does adaptive budgeting work?**
Static caps fail on mixed workloads (too loose for lookups, too tight for
root-cause analysis). `classify_task_complexity` (`agent/budget.py`) is a
transparent keyword classifier - matched on word boundaries, since substring
matching read "count" inside "account" - that picks an `ActionBudget`. A
misrouted question gets a tighter or looser budget, never a policy bypass.

**How does the runtime generalize across four domains without domain code?**
Each domain folder (`domains/<name>/`) contributes an MCP server, policy
rules, a manifest (thresholds the rules enforce, plus a short guide to the
domain's action names) and a benchmark; `agent_runtime.domains` loads and
cross-checks them. The loop has no logic for "credit risk" or "invoices": it
gathers evidence through MCP, verifies numbers, and hands proposals to
whatever `PolicyEngine` it was given.

**What does the multimodal domain actually demonstrate?**
Honestly: not vision. Invoices and charts are pre-extracted JSON fixtures -
what a document-extraction model would output - behind tools named for the
role they play. What the domain demonstrates is the part that matters for
safety once extraction exists: arithmetic is recomputed from the extracted
fields (an invoice billing $2,120 for $1,500 + $120 is caught by
`invoice_math_reconciliation`, not by hoping the LLM notices), missing fields
are null rather than invented, and payment rules fail closed on anything the
extraction did not establish.

**What is the difference between an AI showcase and a decision runtime?**
A showcase connects an LLM to tools and trusts what comes back. A decision
runtime treats the LLM as an untrusted proposer: quantitative claims must be
grounded in an append-only ledger, business actions are judged by
deterministic rules over that ledger, and every step lands in an audit trace
- including when verification failed and where the proposal came from.
