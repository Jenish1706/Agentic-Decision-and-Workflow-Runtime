# Architecture

## System diagram: Agentic Decision & Workflow Runtime

```
                         BUSINESS QUESTION / USER QUERY
                                       |
                                       v
               +------------------------------------------------+
               |              BudgetController                  |
               |  (classifies complexity & bounds resources)    |
               +-----------------------+------------------------+
                                       |
                                       v
                             AgentRuntime Loop
                     (observe -> decide -> act -> verify)
                                       |
        +------------------------------+------------------------------+
        |                              |                              |
        v                              v                              v
   "retrieve"                     local tools                    "call_tool"
  (MCPRetriever)                 (calculator)                (MCP Tool Client)
        |                              |                              |
        v                              |                  +-----------+-----------+
  knowledge server                     |                  |                       |
(search_documents)                     |                  v                       v
        |                              |            records server         analytics server
        |                              |           (lookup_records)       (execute_sql,
        |                              |                                  describe_tables,
        |                              |                                  test_significance)
        +------------------------------+------------------------------------------+
                                       |
                                       v
                       +-------------------------------+
                       |   Append-Only EvidenceLedger  |
                       |  (immutable facts & provenance)|
                       +---------------+---------------+
                                       |
                                       v
                             LLM ActionDecision
                         (proposed_decision & draft)
                                       |
                                       v
                       +-------------------------------+
                       | Deterministic Verifications   |
                       | - verify_citations            |
                       | - verify_numerical_grounding  |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |    PolicyEngine (Governance)  |
                       |  BLOCKED > APPROVAL > ALLOWED |
                       +---------------+---------------+
                                       |
                                       v
                     Grounded Recommendation + AuditTrace
```

Note that the core loop is strictly decoupled into proposal and deterministic verification:
- **LLM proposes**: The LLM suggests tools, interprets questions, and forms drafts.
- **Deterministic verification & constraints**: Citations, numbers, resource limits, and policies
  are verified by deterministic Python logic with zero LLM dependence.

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
            \                                  max_steps=3)
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

## Why this differs from a naive "diagram-first" design

This project started from a proposed architecture (see
[docs/SPEC.md](docs/SPEC.md) section 7) that already looked roughly like
the above. Three deliberate simplifications were made to it, each for a
concrete reason rather than aesthetics:

1. **No separate "Search" (web) MCP server.** A real web-search tool means
   an external API, a cost, and non-determinism in a project meant to run
   entirely offline. A structured-records server still gives a genuinely
   different tool *shape* (exact field lookup vs. semantic search) without
   that dependency.
2. **Query rewriting is not a pipeline stage.** It's something the agent
   loop can already do: if retrieval comes back weak, the loop can decide
   to retrieve again with a reformulated query. Bolting on a dedicated
   "query rewriter" module would duplicate a capability the loop already
   has by construction.
3. **Only the Researcher worker runs a full agent loop.** Retriever is a
   direct pipeline call and Verifier/Synthesizer are single LLM calls -
   giving every worker its own autonomous loop would be four ReAct loops
   instead of one, for no additional capability. See
   [Anthropic's building-effective-agents guidance](https://www.anthropic.com/engineering/building-effective-agents):
   "consider adding complexity only when it demonstrably improves outcomes."

## Component responsibilities

| Module | Responsibility |
|---|---|
| `llm/` | One interface (`generate`, `decide`) over the LLM. `decide` is schema-constrained (Ollama's `format` JSON-schema parameter) and bounded-retries on validation failure. |
| `rag/` | `documents.py` (load+frontmatter) -> `chunking.py` (paragraph-aware, size+overlap) -> `embeddings.py` (sentence-transformers) -> `vector_index.py` (brute-force cosine, numpy) -> `reranker.py` (cross-encoder) -> `pipeline.py` (ties it together) -> `context_builder.py` (citation tagging). |
| `agent/` | `state.py`/`actions.py` (the data), `loop.py` (the control flow), `tools/` (registry + local + MCP adapter), `memory/` (short + long term), `verification.py` (rule-based groundedness), `llm_judge.py` (LLM-based groundedness/task-success, shared with evaluation). |
| `mcp_layer/` | `client.py` (spawns servers as stdio subprocesses, discovery, invocation), `servers/` (two `FastMCP`-style servers). Named `mcp_layer`, not `mcp`, so it never shadows the third-party `mcp` SDK package. |
| `multi_agent/` | `supervisor.py` (delegation + evaluator-optimizer cycle), `workers.py` (Retriever, Researcher), `synthesizer.py`. |
| `evaluation/` | `dataset.py` (loads `data/benchmark.json`), `rag_metrics.py`, `agent_metrics.py`, `run_eval.py` (orchestrates both modes, prints + persists a report). |
| `tracing.py` | Human-readable trace printer (`rich`) - debugging/interview tool, not a production observability stack. |

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
verification by choosing a different action type. This is also a stronger
MCP story than the original design: retrieval, the agent's core
capability, is now genuinely MCP-mediated, not just an auxiliary tool
alongside it.

## Evidence Ledger & Provenance

The runtime treats evidence as a first-class citizen. In `agent/evidence.py`, an append-only
`EvidenceLedger` records every tool execution, statistical test, document excerpt, and calculation
as an immutable `EvidenceItem`. Each item contains:
- `evidence_id`: unique prefixed identifier (e.g. `ev-sql-001`, `ev-stat-002`, `ev-rag-003`).
- `source`: provenance origin (`ANALYTICS_SQL`, `STATISTICS`, `KNOWLEDGE_RAG`, `RECORD_LOOKUP`, `POLICY_RULE`).
- `kind`: nature of fact (`OBSERVATION`, `CALCULATION`, `RETRIEVAL`).
- `query_or_input`: exact SQL text, search query, or expression.
- `data`: structured payload (SQL rows, test metrics, chunk metadata).
- `summary`: human-readable observation snippet.

Because the ledger is strictly append-only and internal state is never exposed as mutable, the LLM
cannot rewrite or fabricate observations retroactively.

## Deterministic Policy Engine & Precedence

Enterprise automation requires that business decisions (e.g., traffic rollouts, budget allocations,
credit limits) be gated by hard, deterministic rules rather than probabilistic LLM self-evaluation.
In `agent/policy.py`, the `PolicyEngine` evaluates a `ProposedDecision` using registered Python
functions (`PolicyRuleFn`).

The engine enforces strict deterministic precedence:
$$\text{BLOCKED} > \text{REQUIRES\_APPROVAL} > \text{ALLOWED}$$

If any rule produces `BLOCKED`, the final decision is blocked regardless of other approvals. If any rule
produces `REQUIRES_APPROVAL` and none block, human sign-off is mandated. Only when all rules pass is the
proposal `ALLOWED`. Crucially, this evaluation contains zero LLM calls.

## Deterministic Numerical Verification

In business and analytical workflows, hallucinations often manifest as distorted numbers, inverted
percentage trends, or invented metrics. In `agent/verification.py`, `verify_numerical_grounding()`
extracts all quantitative claims (percentages, percentage points, currencies, counts) from the draft
answer and reconciles them deterministically against facts in the `EvidenceLedger`.
It supports:
- Scale alignment ($0.0293$ matches $2.9\%$)
- Percentage point difference validation
- Direction validation (preventing "fell" from matching an increase)
- Rejection of ungrounded figures before the agent can terminate.

## Multi-Domain Extensibility: The 4 Enterprise Domains

To ensure the runtime remains fundamentally domain-agnostic, domain-specific assets are segregated under `domains/<domain_name>/`:
- `domain_manifest.yaml`: declared tools, parameters, and deterministic policy rule thresholds.
- Data stores & schemas: DuckDB databases, JSON document models, visual charts, and telemetry.
- `policies.py`: deterministic policy rule definitions enforcing domain compliance.
- `benchmark.json`: comprehensive evaluation questions with expected tools, numbers, and policy outcomes.

The core runtime in `src/agent_runtime/` remains completely generic, interacting with domains strictly through MCP protocol interfaces and standard data contracts:

1. **Product Analytics & Experimentation** (`domains/product_analytics/`):
   - *MCP Server*: `analytics_server.py` exposing `execute_sql`, `describe_tables`, `test_significance`.
   - *Data Warehouse*: DuckDB database with `daily_checkout_metrics`, `funnel_dropoff`, `experiments`, and `experiment_results`.
   - *Statistical Engine*: SciPy pooled two-proportion hypothesis z-test calculating z-statistics, p-values, and 95% confidence intervals.
   - *Governance*: `statistical_significance_gate` (blocks rollout if $p \ge 0.05$) and `high_impact_rollout_approval` (requires human approval if traffic $> 50\%$).

2. **Credit Risk & Lending Underwriting** (`domains/credit_risk/`):
   - *MCP Server*: `credit_risk_server.py` exposing read-only DuckDB operations and `get_account_profile` dossier lookups.
   - *Data Warehouse*: DuckDB database with `accounts`, `credit_history`, `limit_requests`, and `bureau_inquiries`.
   - *Governance*: `minimum_credit_score` (blocks if FICO $< 650$), `hard_delinquency_cutoff` (blocks if 90d delinquencies $> 0$), `dti_ceiling` (blocks if DTI $> 43\%$), and `high_limit_increase_approval` (requires underwriter approval if increase $> \$5,000$).

3. **Multimodal Business Investigation** (`domains/multimodal_investigation/`):
   - *MCP Server*: `vision_server.py` exposing `extract_document_metrics`, `analyze_chart_data`, and `list_visual_artifacts`.
   - *Data Assets*: Scanned invoice JSON documents with OCR layout sidecars and time-series telemetry / funnel charts.
   - *Governance*: `invoice_math_reconciliation` (blocks on arithmetic mismatch $> \$0.01$), `duplicate_invoice_prevention` (blocks re-submitted/settled claims), `executive_expense_approval` (escalates if total $> \$1,000$), and `sla_breach_detection` (flags commercial penalty reviews if uptime $< 99.90\%$).

4. **Enterprise Systems Operations** (`domains/enterprise_operations/`):
   - *MCP Server*: `ops_server.py` exposing `get_incident_telemetry`, `inspect_service_config`, and `simulate_remediation_action`.
   - *Data Assets*: Production incident registries (`incidents.json`) and deployment configurations (`service_configs.json`).
   - *Governance*: `production_blast_radius_cap` (escalates if blast radius $> 25\%$), `active_p0_incident_freeze` (blocks routine mutations during active P0 outages), `security_baseline_compliance` (blocks TLS $< 1.3$ or exposed debug ports), and `safe_automated_restart` (allows low-impact restarts $\le 10\%$).

## Concurrency model

The whole runtime is `async` end-to-end (LLM calls, tool execution, MCP
client calls) rather than sync-with-a-thread-bridge. Embedding and
reranking (sentence-transformers/CPU) are the one exception - they're
called synchronously from within async functions, which blocks the event
loop for a few milliseconds per call. For a single-user local project with
no concurrent requests, that's a non-issue; a concurrent-request server
would run them in a thread pool executor instead.

## Known simplifications

See the README's "Known simplifications" section - repeated here only
where it's an architectural decision rather than a setup note: no
persisted vector index (rebuilt in-memory per run), and each MCP server
subprocess builds its own RAG pipeline independently (no shared memory
between processes - see MCP.md).
