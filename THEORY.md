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
is `while not done: decide -> act -> observe -> update state`. State
(`agent/state.py`'s `AgentState`) is everything accumulated *this run*: the
query, every step taken, whether it's finished. An observation is the
result of an action - retrieved text, a tool's return value, or a tool's
error message - appended to state so the *next* decision can see it.

**What is tool calling? How does an agent decide which tool to use?** The
LLM is shown every registered tool's name, description, and parameter
schema (rendered into the system prompt by `_render_tool_schemas` in
loop.py) and asked to choose. This project does not use any provider's
native tool-calling API for that choice - see ARCHITECTURE.md's "Why a
hand-rolled agent loop" - the model's decision is schema-constrained JSON
(`ActionDecision`, [agent/actions.py](src/agent_runtime/agent/actions.py))
that this code parses and dispatches itself.

**How do you prevent infinite loops?** A hard `max_steps` bound
(`config.py`'s `max_agent_steps`) - the loop is a bounded `for` over
`range(max_steps)`, not a `while True`. If the model never reaches a
grounded final answer, the run terminates with `TerminationReason.MAX_STEPS_REACHED`
rather than continuing indefinitely.

**What happens when a tool fails?** [agent/tools/registry.py](src/agent_runtime/agent/tools/registry.py)'s
`execute()` catches both bad arguments (`TypeError`) and the tool's own
exceptions, turning either into a `ToolResult(ok=False, error=...)` - never
an exception that would crash the loop. The error text becomes the next
observation, so the model can see *what* went wrong and try something else.
The same applies to the LLM call itself: if `decide()` never produces
schema-valid JSON after retries, `agent/loop.py` catches `LLMError` and
terminates with `UNRECOVERABLE_ERROR` rather than propagating a crash.

## Multi-agent

**Why multiple agents? Why not one?** Per Anthropic's guidance (quoted in
ARCHITECTURE.md): start with a single agent, and add multi-agent
orchestration only where it demonstrably helps - specialization
(routing a sub-task to a worker built for it) and separated
generation/verification (evaluator-optimizer). This project's Supervisor
([multi_agent/supervisor.py](src/agent_runtime/multi_agent/supervisor.py))
exists specifically to demonstrate and measure that trade-off against the
single-agent path - see [EVALUATION.md](EVALUATION.md) for the actual
comparison, not just the architecture.

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
more code to reason about. [EVALUATION.md](EVALUATION.md) reports the
actual step-count and latency delta measured on this project's benchmark,
not just the theoretical concern.

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
  *across* runs. Short-term = recent turns in this process; long-term =
  persisted to disk, retrieved by similarity.
- **Retrieval** - the general mechanism (embed a query, search an index).
  RAG and long-term memory both use it, over different collections.

**Short-term vs. long-term memory?** [short_term.py](src/agent_runtime/agent/memory/short_term.py)
is just the last N `(query, answer)` pairs, injected verbatim - no
retrieval, no persistence, cleared when the process ends.
[long_term.py](src/agent_runtime/agent/memory/long_term.py) is embedded and
persisted to `data/long_term_memory.json`, and survives a process restart.

**How is memory retrieved?** By the exact same mechanism as document RAG -
`LongTermMemory` holds its own `Embedder` + `VectorIndex` instance, over a
much smaller collection of remembered facts rather than document chunks.
The retrieval *code path* is identical; only the collection differs.

**When should information be stored? Why isn't the whole conversation
automatically memory?** Because most of a conversation is not durable -
routine questions and their answers aren't useful to recall in an unrelated
future conversation. `LongTermMemory.maybe_store()` puts this decision to
an explicit LLM call (`_MemoryWriteDecision`) after every turn, asking
specifically whether *this* exchange contains a preference or constraint
worth keeping. Most turns come back `worth_remembering=False` and nothing
is written - see the test cases in
[test_memory.py](tests/test_memory.py) for both outcomes.

## MCP

See [MCP.md](MCP.md) for the full write-up. Short version: MCP solves tool
*discovery and invocation across a process boundary* - what problem it
solves, client vs. server, why schemas, and local-vs-protocol tool contrast
are all covered there with direct code references.

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
tool selection accuracy (did it call the tool the query actually needed),
unnecessary tool calls, step count, and failure rate (didn't reach a final
answer at all). See [evaluation/agent_metrics.py](src/agent_runtime/evaluation/agent_metrics.py).

**What failure modes did you observe?** See [EVALUATION.md](EVALUATION.md)'s
"Observed failure modes" section for what actually happened running this
project's own benchmark - not a hypothetical list. The six worth being able
to recount from memory: a citation-avoidance loop that nearly exhausted the
step budget; a verification bypass caused by two overlapping document-search
paths; three separate structured-output violations that aborted whole runs;
tool thrashing that turned inefficiency into outright task failure; a vague
worker description that silently disabled the entire multi-agent path; and a
wrong `expected_tool` value in the benchmark that made a *measurement* bug
look like an agent bug.

**Which of your results were negative or surprising?** Worth rehearsing,
because it's the most likely follow-up: reranking produced no measurable
improvement (corpus too small for the bi-encoder's top-k to be polluted);
the multi-agent path was *faster* than the single agent, not slower; and -
the one that most deflates my own architecture - **capping the single
agent at one retrieval reproduced almost all of the multi-agent
advantage** (0.70 → 0.90 vs multi-agent's 0.95). So the measured benefit
is mostly a *constraint* effect, not an orchestration effect.

**What is a retrieval budget, and does your system have one?** Careful
answer: it has *implicit* budgets (`top_k_retrieve=8` per call,
`max_agent_steps=8` per run as a blunt shared counter) and no context-token
bound at all. The only explicit per-run retrieval cap,
`AgentRuntime(max_retrievals=...)`, was added to run the experiment and
defaults to `None`. Search *efficiency* - redundancy, repeated queries,
chunks consumed - is measured in
[evaluation/retrieval_metrics.py](src/agent_runtime/evaluation/retrieval_metrics.py).
The interesting finding is that loosening the budget degraded every
dimension measured, and that a tight cap acted as a **hallucination
control**: out-of-corpus refusals went 1/3 → 3/3, because an agent that
can keep searching for a nonexistent fact eventually confabulates one from
tangential context.

## Decision & Workflow Runtime Theory

**Why separate the LLM proposal from deterministic policy enforcement?**
LLMs are probabilistic and susceptible to prompt injection, semantic drifting, and
hallucinated justifications. If an agent evaluates its own actions against policy, it
can hallucinate that an exception applies. The runtime enforces an architectural
invariant: *LLM proposes; deterministic code verifies and enforces*. A `ProposedDecision`
is evaluated by `PolicyEngine` (`agent/policy.py`) using pure Python functions without
any LLM in the loop.

**Why is an append-only Evidence Ledger different from agent state or memory?**
Agent state is an ephemeral scratchpad that is discarded after a run. Long-term memory
is semantic and heuristic. The `EvidenceLedger` (`agent/evidence.py`) is an append-only,
immutable audit record of exact tool observations, SQL rows, and statistical tests with
unique IDs (`ev-sql-001`) and execution timestamps. Because it is immutable, the model
cannot retroactively alter historical observations to fit a desired conclusion.

**Why does regex + scale reconciliation beat LLM-as-a-judge for numerical verification?**
LLMs are notoriously weak at arithmetic and scale reconciliation (e.g. confusing 0.0293
with 29.3% instead of 2.93%, or conflating percentage points with relative percentage change).
`verify_numerical_grounding` (`agent/verification.py`) uses deterministic regex to extract
all quantitative claims, parses units (%, pp, currency, counts), aligns scale ($x \times 100$),
and verifies that every reported number mathematically reconciles with ledger facts within a
strict tolerance ($\pm 0.015$).

**How does policy precedence work?**
Enterprise governance requires predictable escalation:
$$\text{BLOCKED} > \text{REQUIRES\_APPROVAL} > \text{ALLOWED}$$
If a proposal violates a safety or compliance threshold, it is immediately `BLOCKED`.
If it requires operational sign-off (e.g., rolling out to >50% of users), it transitions to
`REQUIRES_APPROVAL`. Only proposals passing all registered checks are `ALLOWED`.

**How does adaptive budgeting work?**
Static resource caps fail on mixed workloads (too loose for factual questions, too tight
for multi-step root-cause analysis). `classify_task_complexity` (`agent/budget.py`) inspects
query intent deterministically and assigns an `ActionBudget` bounding steps, searches, and
SQL queries, preventing runaway loops while allowing deep investigation.

**How does the runtime generalize across 4 diverse domains without code duplication?**
By strictly separating domain assets (`domains/<domain>/` with manifests, schemas, and policy rules)
from the core execution engine (`src/agent_runtime/`). The agent loop does not have any hardcoded
logic for "credit risk" or "invoices" or "servers". Instead, it speaks the Model Context Protocol (MCP),
extracts evidence into an immutable ledger, enforces deterministic Python policy functions via
`PolicyEngine`, and verifies quantitative assertions against ledger numbers. This allows the SAME
runtime to underwrite loans, audit scanned expense invoices, diagnose SRE outages, and analyze A/B tests.

**Why use deterministic sidecars/OCR extraction instead of black-box multimodal APIs?**
Proprietary multimodal LLMs hallucinate numbers and alter arithmetic totals when reading invoices or charts.
By combining layout/OCR parsing with the `EvidenceLedger`, line items, subtotals, and taxes are converted into
deterministic data payloads *before* reaching the LLM. If an invoice has a \$500 phantom surcharge, the
`invoice_math_reconciliation` policy catches it mathematically, rather than hoping the LLM notices a subtle discrepancy.

**What is the difference between an AI showcase and an Enterprise Decision Runtime?**
An AI showcase connects an LLM to tools and trusts whatever the LLM outputs. An enterprise decision runtime
treats the LLM as an un-trusted reasoning engine: the LLM proposes actions and drafts recommendations, but
all quantitative claims are strictly grounded in an append-only ledger, all business actions are evaluated by
deterministic governance rules, and all steps are recorded in an auditable decision trace.
