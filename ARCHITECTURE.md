# Architecture

## System diagram

```
                     QUESTION (+ optional ActionBudget chosen by the caller)
                                       |
                                       v
             AgentRuntime.run()  - fresh AgentState, fresh BudgetController
                    (decide -> act -> observe, bounded by max_steps)
                                       |
        +------------------------------+------------------------------+
        |                              |                              |
        v                              v                              v
   "retrieve"                     local tools                    "call_tool"
  (MCPRetriever)                 (calculator)                (MCP tool client)
        |                              |                              |
        v                              |              +---------------+---------------+
  knowledge server                     |              |                               |
 (search_documents)                    |       records server               a domain server
        |                              |       (lookup_record,            (domains/<name>/server.py:
        |                              |        search_records)            SQL, statistics, documents,
        |                              |                                   incidents, ...)
        +------------------------------+-------------------------------------------+
                                       |
                                       v
                       +-------------------------------+
                       |   Append-only EvidenceLedger  |
                       |  deep-frozen tool outputs,    |
                       |  provenance per item          |
                       +---------------+---------------+
                                       |
                           final answer draft (+ ProposedDecision)
                                       |
                                       v
                       +-------------------------------+
                       |  Deterministic verification   |
                       |  - verify_citations           |
                       |  - verify_numerical_grounding |
                       |  failed -> corrective         |
                       |  observation, loop continues  |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |  PolicyEngine (domain rules)  |
                       |  facts read from the LEDGER,  |
                       |  never from the proposal      |
                       |  BLOCKED > APPROVAL > ALLOWED |
                       +---------------+---------------+
                                       |
                                       v
                  answer + answer_verified + policy verdict + audit trace
```

The loop is split into proposal and deterministic control:

- **The LLM proposes**: which action to take next, the answer text, and an
  optional structured `ProposedDecision`.
- **Deterministic code verifies and constrains**: budgets, citation checks,
  numerical checks and policy rules are plain Python over tool outputs, with
  zero LLM calls.

Multi-agent mode (`Supervisor`) sits beside `AgentRuntime`, reusing the same
RAG pipeline, tool registry, and LLM client rather than duplicating them:

```
             Supervisor (orchestrator)
             decides: needs_retrieval? needs_tools?
             /                                    \
            v                                      v
   RetrieverWorker                          ResearcherWorker
   (direct RAG pipeline call,                (an AgentRuntime instance,
    no LLM call of its own)                   rag=None, tools-only,
            \                                  max_steps=3, fresh session per task)
             \                                    /
              v                                  v
                    Synthesizer (generate)
                              |
                              v
                    Verifier (llm_judge_faithfulness)
                              |
                    grounded? --no--> Synthesizer retry (x1) -> Verifier
                              |
                             yes
                              v
                        final answer
```

## Design choices that shaped the structure

The first design was larger. Three simplifications were made on purpose,
each for a concrete reason:

1. **No web-search MCP server.** A real web-search tool means an external
   API, a cost, and non-determinism in a project meant to run entirely
   offline. The structured-records server still gives a genuinely different
   tool *shape* (exact field lookup vs. semantic search) without that
   dependency.
2. **Query rewriting is not a pipeline stage.** The agent loop can already
   retrieve again with a reformulated query when results are weak; a
   dedicated "query rewriter" would duplicate a capability the loop has by
   construction.
3. **Only the Researcher worker runs a full agent loop.** Retriever is a
   direct pipeline call and Verifier/Synthesizer are single LLM calls -
   giving every worker its own autonomous loop would be four ReAct loops
   instead of one, for no additional capability. See
   [Anthropic's building-effective-agents guidance](https://www.anthropic.com/engineering/building-effective-agents):
   "consider adding complexity only when it demonstrably improves outcomes."

Deliberately out of scope: model training or fine-tuning, approximate
nearest-neighbour index algorithms, distributed infrastructure (queues,
containers, databases beyond embedded DuckDB), authentication, production
observability stacks, and re-implementing the MCP wire protocol.

## Component responsibilities

| Module | Responsibility |
|---|---|
| `llm/` | One interface (`generate`, `decide`) over the LLM. `decide` is schema-constrained (Ollama's `format` JSON-schema parameter) and retries a bounded number of times on validation failure. |
| `rag/` | `documents.py` (load + frontmatter) -> `chunking.py` (paragraph-aware, size + overlap) -> `embeddings.py` (sentence-transformers) -> `vector_index.py` (brute-force cosine, numpy) -> `reranker.py` (cross-encoder) -> `pipeline.py` -> `context_builder.py` (citation tagging). |
| `agent/` | `state.py` / `actions.py` (the data), `loop.py` (the control flow), `tools/` (registry, local calculator, MCP adapter, `ToolCategory`), `memory/` (short + long term), `evidence.py` (ledger), `verification.py` (citations + numbers), `facts.py` (evidence-scoped fact resolution for rules), `policy.py` (engine + precedence), `budget.py` (allocations + per-run controller), `llm_judge.py` (LLM-judged groundedness / task success, used by evaluation and the multi-agent verifier). |
| `mcp_layer/` | `client.py` (spawns servers as stdio subprocesses, discovery, invocation), `duckdb_tools.py` (the shared read-only SQL toolkit), `servers/` (knowledge + records). Named `mcp_layer`, not `mcp`, so it never shadows the `mcp` SDK. |
| `domains.py` | Loads and validates a business domain folder: manifest thresholds and rule names checked against `policies.py`, server path, benchmark path. |
| `multi_agent/` | `supervisor.py` (delegation + evaluator-optimizer cycle), `workers.py` (Retriever, Researcher), `synthesizer.py`. |
| `evaluation/` | `dataset.py` (benchmark loader), `rag_metrics.py`, `retrieval_metrics.py` (search efficiency), `agent_metrics.py` (task success, tools, expected-number recall, failures), `run_eval.py`, `budget_experiment.py`, `product_analytics_experiment.py`. |
| `runtime_factory.py` | Builds single- and multi-agent runtimes identically for scripts, API and evaluations. |
| `tracing.py` | Human-readable trace printer (`rich`) - a debugging/interview tool, not an observability stack. |

## Why brute-force vector search, not FAISS/HNSW

The corpus is ~30 documents, ~30 chunks (most documents are short enough to
stay a single chunk at `chunk_size=800`). Search is one `(n, dim) @ (dim,)`
matrix-vector product - microseconds on a CPU. An approximate-nearest-neighbor
index (FAISS/HNSW/etc.) trades a small amount of recall for large speedups at
scale, and that trade only pays for itself past roughly 10^5-10^6 vectors.
Below that, it's an opaque dependency standing in for four lines of numpy
that are more useful to be able to explain than to have imported. See
[src/agent_runtime/rag/vector_index.py](src/agent_runtime/rag/vector_index.py).

## Why a hand-rolled agent loop instead of native tool-calling

Ollama supports a native `tools` parameter for capable models, and so does
every hosted LLM API. This project deliberately does not use it for the
agent's own decision-making: the point of the project is the loop itself
(state -> decision -> action -> observation -> termination), and routing
that through a provider's built-in tool-calling machinery would hide
exactly the mechanism being demonstrated. Instead, `llm/ollama_client.py`
uses Ollama's `format` parameter (JSON-schema-constrained decoding) purely
as a "produce text matching this shape" primitive, and `agent/loop.py`
owns every bit of the decide -> act -> observe control flow itself.

MCP tool *invocation* is a different concern from LLM decision-making - see
[MCP.md](MCP.md) for why that boundary is not reimplemented from scratch.

## Why `retrieve` is MCP-mediated (a bug found by actually running this)

The first working version of this project gave the single agent *two*
separate ways to search documents: the dedicated `retrieve` action (calling
`RagPipeline` directly, in-process) and the knowledge server's
`search_documents` tool (reachable through the ordinary `call_tool` action,
since it was discovered and registered like any other MCP tool). Both
worked. The bug only showed up running the real demo against Ollama: an
agent that reached for `search_documents` as a tool call got back real
retrieved text, but that text never touched `citation_map` - only the
dedicated `retrieve` action populated it. So an answer built from a
`search_documents` tool call sailed through `verify_citations` with zero
citations and zero complaints, including, in one real run, a fabricated
revenue figure for a question the corpus has no answer to.

The fix ([agent/mcp_retriever.py](src/agent_runtime/agent/mcp_retriever.py))
makes `retrieve` itself call the knowledge server's `search_documents` tool
through the MCP client, and `search_documents` is no longer separately
registered as a callable tool anywhere an agent could reach it directly
(see `_TOOLS_EXCLUDED_FROM_AGENT_REGISTRY` in
[runtime_factory.py](src/agent_runtime/runtime_factory.py)). There is now
exactly one way to search documents, it is protocol-accessed, and it is the
one path that populates `citation_map` - so nothing can bypass
verification by choosing a different action type.

## Evidence ledger and provenance

[agent/evidence.py](src/agent_runtime/agent/evidence.py): every successful
retrieval and tool call appends one `EvidenceItem` to the run's
`EvidenceLedger`:

- `evidence_id` - prefixed and unique within the run (`ev-sql-001`, `ev-stat-002`, `ev-rag-003`)
- `source` - `knowledge_rag`, `analytics_sql`, `statistics`, `vision_vlm`, `record_lookup`, `calculator`
- `kind` - `observation`, `calculation` or `retrieval`
- `query_or_input` - the SQL, search query or tool arguments
- `data` - the tool's structured output
- `summary`, `provenance`, `timestamp`

`source` comes from the tool's declared `ToolCategory`, not from its name.

The ledger is the trust boundary of the runtime, so its immutability is
enforced, not assumed: each item's `data` is deep-frozen on construction
(dicts become read-only mappings, lists become tuples) from a *copy* of the
tool's output, and exports return fresh copies. An earlier version was a
frozen dataclass holding a plain dict, which `frozen` does not protect -
`item.data["p_value"] = 0.9` silently succeeded.

## Budgets are per run

[agent/budget.py](src/agent_runtime/agent/budget.py): an `ActionBudget` is an
immutable allocation (steps, retrievals, SQL queries, extraction calls). The
caller picks one - fixed, or by `classify_task_complexity` for adaptive
budgeting - and `AgentRuntime.run(query, budget=...)` creates a fresh
`BudgetController` for that run. An earlier version stored one controller on
the runtime and never reset it, so every later question started with an
already-spent budget. Exhausted budgets are reported to the model as
observations ("SQL query budget exhausted...") rather than silently skipped.

## Deterministic numerical verification

[agent/verification.py](src/agent_runtime/agent/verification.py):
`verify_numerical_grounding()` extracts quantitative claims (percentages,
percentage points, currency, counts) from the draft and requires each one to
be supported by a number in evidence, a number in the user's question, or a
declared policy threshold. Identifiers, versions, dates and ordinals are not
claims.

- **Rounding-aware tolerance**: "2.9%" accepts 2.85-2.95, nothing looser.
- **Units**: a percentage claim matches a stored ratio (0.029) or a
  percent-unit field, never a plain count - "43%" is not grounded by 43
  purchases. Calculator results are unitless and may ground either.
- **Derived values**: differences and relative changes are searched only
  between comparable values (same field, or the same tool output).
- **Direction**: "fell 1.3 pp" is rejected when evidence holds a signed change
  of +1.3. A gap derived from two unordered values has no recoverable
  direction, so only its magnitude is verified - a stated limitation.

It verifies that numbers exist in evidence at a compatible scale, not that
each is attached to the right concept; decisions therefore never rely on the
answer text (next section).

A rejected answer becomes an observation telling the model what failed, and
the loop continues. On the very last step an unverified answer is accepted as
best effort, but marked `answer_verified=False` - and the policy engine then
refuses to auto-approve any decision attached to it.

## Policy engine: facts from evidence, fail closed

[agent/policy.py](src/agent_runtime/agent/policy.py) evaluates a
`ProposedDecision` with registered rule functions and strict precedence:

$$\text{BLOCKED} > \text{REQUIRES\_APPROVAL} > \text{ALLOWED}$$

One `BLOCKED` result blocks the decision however many rules allow it, so
adding a rule can only make the system more restrictive; an ALLOWED rule is a
no-op. The engine fails closed: a rule that raises counts as `BLOCKED`, a rule
result whose `passed` flag contradicts its status is read restrictively, a
proposal citing evidence ids the ledger does not contain is `BLOCKED`, and a
decision attached to an unverified answer needs approval.

The domain rules are where the most serious bugs were. They used to read
facts from the proposal first (so `{"credit_score": 700}` in the model's own
parameters overrode a 600 in the database, and `{"override_authorized": true}`
lifted a production freeze), otherwise took the first matching number in the
ledger regardless of which account or invoice it belonged to, and treated a
missing fact as "no violation". They now use
[agent/facts.py](src/agent_runtime/agent/facts.py):

- The proposal only names the **target** (account, invoice, service,
  experiment) and the action's **own quantities** (increase amount, traffic
  percentage). Where evidence also speaks to a quantity, the stricter value
  wins.
- Facts are resolved **per entity**: a value counts for ACCT-104 only if it
  sits inside a record carrying `account_id = ACCT-104`.
- **Evidence may restrict on weak grounds but permit only on strong ones**:
  any violating candidate value blocks, even one that cannot be attributed to
  the entity; allowing requires one agreed value attributed to the target.
- A fact the evidence does not establish means `REQUIRES_APPROVAL`.
- Rules apply to the actions they govern, and an unrecognised action name is
  governed (the old exact-string list let "grant_limit_raise" skip every
  credit rule).

## Business domains

Each domain is a self-contained folder, loaded by
[agent_runtime/domains.py](src/agent_runtime/domains.py):

```
domains/<name>/
  domain_manifest.yaml   title, thresholds, documented rules, decision guide for the model
  server.py              the domain's MCP server
  policies.py            REQUIRED_THRESHOLDS + create_policy_engine(thresholds)
  benchmark.json         evaluation questions (expected numbers, expected policy status)
  data/                  DuckDB databases, JSON fixtures, generators
```

The manifest is no longer documentation that merely *looks* like
configuration: its thresholds are the values the code enforces, and the
loader rejects missing or unknown thresholds and any mismatch between
documented and implemented rule names. (The old manifests had drifted: the
product analytics one said rollouts above 20% need approval while the code
enforced 50%, and the operations one described a rule no code implemented.)
The core runtime has no domain-specific code; each domain contributes a
server, rules and a short decision guide.

1. **Product analytics** - DuckDB checkout metrics, funnel drop-off and A/B
   results; a two-proportion z-test. Rules: a rollout needs the named
   experiment's p-value below alpha *and* the treatment ahead of control;
   more than 50% of traffic needs approval.
2. **Credit risk** - DuckDB accounts, credit history, limit requests,
   inquiries. Rules: score below 650, any 90-day delinquency, or DTI above
   43% blocks; an increase above $5,000 needs approval.
3. **Multimodal investigation** - invoices and charts as pre-extracted JSON
   fixtures (no image model runs; the tools compute invoice arithmetic and
   SLA status from the extracted fields). Rules for payments: line items and
   subtotal + tax must reconcile to the cent, duplicates and settled invoices
   are blocked, totals above $1,000 need approval. For SLA penalties: a
   monthly average below the contractual threshold needs commercial review.
4. **Enterprise operations** - incident registry and service configurations.
   Rules: changes to a service with an active, un-overridden P0 are blocked
   ("no active P0" must come from a complete incident listing); blast radius
   above 25% needs approval unless an emergency override covers it;
   deployments of services with TLS below 1.3 or an open debug port are
   blocked.

## Concurrency model

The whole runtime is `async` end-to-end (LLM calls, tool execution, MCP
client calls) rather than sync-with-a-thread-bridge. Embedding and
reranking (sentence-transformers on CPU) are the exception - they are called
synchronously inside async functions and block the event loop briefly. That
is acceptable for a local project; a concurrent server would move them to a
thread pool. The FastAPI wrapper shares one runtime between requests, so it
resets the conversation at the start of every request and disables long-term
memory (see `api.py`).

## Known simplifications

See the README's "Known simplifications" section - repeated here only
where it's an architectural decision rather than a setup note: no
persisted vector index (rebuilt in-memory per run), and each MCP server
subprocess builds its own state independently (no shared memory between
processes - see MCP.md).
