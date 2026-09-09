# Agentic RAG Runtime — Project Specification

Status: finalized design, pre-implementation reference. This is the internal
design record; polished user-facing docs (README, ARCHITECTURE, THEORY,
EVALUATION, MCP) are written after implementation so they describe what was
actually built, not what was intended.

## 1. Title
**Agentic RAG Runtime**

## 2. Thesis
A from-scratch runtime that fuses retrieval-augmented generation, an explicit
agent reasoning loop, protocol-based tool access via MCP, short/long-term
memory, and a small multi-agent architecture — built to demonstrate
first-principles understanding of how modern agentic AI systems are actually
assembled under the frameworks, not to ship a product or explore novel
research.

## 3. Problem the system demonstrates
Given a natural-language query that may require unstructured document
evidence, structured record lookup, arithmetic, multiple reasoning steps, or
prior conversation context, the system must: decide what to do step by step,
ground its claims in retrieved evidence, invoke the correct tool (local or
MCP) with valid arguments, recover from malformed model output, optionally
delegate to specialist sub-agents, verify its own answer before returning it,
and be independently measurable via an evaluation harness rather than "looks
right in a demo."

## 4. Learning objectives
RAG mechanics; the agent loop (state → decision → action → observation →
termination); tool calling and failure recovery; MCP client/server
architecture; memory vs. context vs. state; multi-agent delegation and
verification; practical LLM/agent evaluation.

## 5. Core concepts covered
Chunking, embeddings, vector search, reranking, grounded generation,
citations · agent state/loop/termination/retries · tool registry (local +
MCP) · MCP client & server, tool discovery, schemas, resources · short-term
and long-term memory · supervisor/worker multi-agent coordination ·
retrieval, groundedness, tool-selection, and task-success evaluation.

## 6. Deliberately excluded
Model training/fine-tuning/RLHF/DPO; building embeddings or transformers
from scratch; ANN index algorithms (HNSW/IVF — brute-force numpy is the
right call at this corpus size); distributed infra, k8s, queues, Redis,
Postgres; auth systems; production observability stacks; real web-search
APIs (cost/flakiness, replaced by a local structured-records tool);
programmatic/code-executing tool orchestration (Anthropic's newer
"programmatic tool calling" — real but adds a layer this project doesn't
need to make its point); recreating the MCP wire protocol (use the official
SDK).

## 7. Final architecture
Single-agent core:
```
query -> AgentRuntime (state, decision loop, tool router, memory)
           |-- decide: retrieve | call_tool | delegate | final
           |-- retrieve -> RAG pipeline (embed -> vector search -> rerank -> context+citations)
           |-- call_tool -> ToolRegistry -> {local tool | MCP client -> MCP server}
           |-- final -> Verification (groundedness check) -> answer
```
Multi-agent (opt-in mode over the same primitives):
```
Supervisor (orchestrator: decomposes query, delegates, collects results)
  |-- Retriever  worker  (RAG pipeline + knowledge MCP tool)
  |-- Researcher worker  (records/calculator tools, local reasoning)
  |-- Verifier   worker  (evaluator-optimizer: checks draft against evidence)
  -> Synthesizer (single LLM call combining worker outputs into final answer)
```

Changes from the originally proposed diagram, and why:
- **Search server dropped** to a **Records server** (structured JSON lookup).
  A real web-search tool means an external API, cost, and flakiness in a
  project meant to run offline/free; a structured-record tool still gives a
  genuinely different tool shape (exact/field lookup vs. semantic search)
  without the dependency.
- **3 MCP servers → 2** (Knowledge, Records) **+ 1 local tool** (calculator).
  Two servers already prove multi-server discovery/schema/invocation; a
  third would be head-count padding, not a new concept. The local calculator
  is what makes the local-vs-MCP contrast concrete.
- **Query rewriting is not a separate pipeline stage.** It's an emergent
  agent capability: if reranked evidence is weak, the loop can decide to
  retrieve again with a reformulated query. This gives the "fixed vs.
  adaptive retrieval" comparison the spec asks for, for free, instead of as
  a bolted-on module.
- **Verification is one function, two call sites**, not two separate
  concepts: single-agent mode calls it once on the draft answer; multi-agent
  mode runs it as the dedicated Verifier worker (Anthropic's
  evaluator-optimizer pattern). Same grounding-check logic either way.
- **Synthesizer is a single generation call, not an autonomous agent.**
  Giving it its own decide/act/observe loop would be a fourth independent
  ReAct loop for no added capability — it only ever combines already-gathered
  worker outputs.

## 8. Component responsibilities
- `llm/` — provider interface + Ollama implementation; structured decision
  decoding via JSON-schema-constrained output.
- `rag/` — ingestion, chunking, embeddings, brute-force vector index,
  cross-encoder reranker, context/citation builder.
- `agent/` — state, action schema, the explicit loop, tool registry, local
  tools, verification, memory.
- `mcp/` — two FastMCP servers (knowledge, records) + a thin MCP client
  wrapper the agent's tool registry can call through.
- `multi_agent/` — supervisor, workers, synthesizer built on the same
  agent/tool/memory primitives.
- `evaluation/` — benchmark loader, RAG metrics, agent metrics, report CLI.
- `tracing.py` — human-readable step trace for debugging/interview demos.

## 9. Data flow (single-agent, happy path)
`query → AgentState.init → loop: LLM decision → {retrieve|tool|final} →
observation appended to state → … → final answer draft → verification →
(retry once if ungrounded) → answer + citations + trace`

## 10. Agent execution flow
```
while step < max_steps and not done:
    prompt = build_prompt(state)                 # system + tool schemas + history
    decision = llm.decide(prompt, ActionDecision) # schema-constrained JSON
    validate decision (tool exists? args match schema?) -> retry on failure (bounded)
    execute action -> Observation
    state.append(step, decision, observation)
    if decision.action == FINAL: verify(); done = True
termination reasons: FINAL_ANSWER | MAX_STEPS_REACHED | UNRECOVERABLE_ERROR
```

## 11. RAG flow
`documents (data/documents/*.md) → chunk (paragraph-aware, size+overlap) →
embed (all-MiniLM-L6-v2) → brute-force cosine top-k → cross-encoder rerank
(ms-marco-MiniLM-L-6-v2) → top-n context assembly with per-chunk citation
tags → LLM generation constrained to cite only provided chunk ids`

## 12. MCP flow
`agent tool registry startup → spawn each MCP server as a stdio subprocess →
ClientSession.list_tools() (discovery) → schemas merged into the same
registry shape as local tools → decision names a tool → registry routes by
origin → MCP path calls session.call_tool(name, args) → structured content
converted to an Observation identically to a local tool's return value`

## 13. Multi-agent flow
`query → Supervisor decomposes into sub-tasks → Retriever + Researcher run
(can run independently) → Verifier checks the combined evidence/draft →
Synthesizer produces final cited answer → trace records which worker
contributed which evidence`

## 14. Memory flow
`STATE (this run's scratchpad: steps/observations, cleared per query) ≠
CONTEXT (what's actually assembled into the next prompt: relevant state +
retrieved chunks + memory excerpts) ≠ MEMORY (persists across queries:
short-term = recent turn history in this session, truncated/summarized;
long-term = a small separate embedded store written only on explicit
"worth remembering" signals, not every message) ≠ RETRIEVAL (the general
mechanism — long-term memory is retrieved using the *same* vector-search
code path as document RAG, just a different collection).`

## 15. Evaluation flow
`benchmark.json (~18 hand-written queries) → run each through single-agent
and multi-agent → log full trace → compute RAG metrics (Recall@k, MRR,
groundedness, citation correctness) + agent metrics (task success, tool
selection accuracy, unnecessary calls, step count, failure rate) + system
metrics (latency, tool-call count) → report.json + printed summary table`

## 16. Technology choices
| Choice | Reason |
|---|---|
| Ollama (local) | User-selected; free, offline, no key management. |
| `all-MiniLM-L6-v2` embeddings | Standard, CPU-fast (384d, ~90MB), well documented. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker | Standard MS MARCO cross-encoder, small, CPU-fast. |
| Brute-force numpy vector search | Corpus is dozens of docs/few hundred chunks — O(n·d) is milliseconds; writing the actual similarity math is more instructive than an opaque ANN library. FAISS/HNSW are the right call only past ~10⁵–10⁶ vectors. |
| Official `mcp` Python SDK (FastMCP) | Reinventing the wire protocol teaches nothing; using the SDK correctly is the actual skill being demonstrated. |
| `pydantic` | Schema validation for the agent's structured decisions and tool I/O — used as infrastructure, not to hide agent logic. |
| `uv` | Fast, modern Python env/dependency management. |
| `pytest` | Standard; unit tests use a fake deterministic LLM, integration tests are marked and need Ollama running. |
| Thin optional FastAPI wrapper | Ties into coursework, ~2-3 endpoints over the same runtime, not the primary interface. |

## 17. Repository structure
```
agent-runtime/
  README.md ARCHITECTURE.md THEORY.md EVALUATION.md MCP.md
  pyproject.toml  .env.example  .gitignore
  src/agent_runtime/
    config.py
    llm/{base.py, ollama_client.py}
    rag/{documents.py, chunking.py, embeddings.py, vector_index.py,
         reranker.py, context_builder.py, pipeline.py}
    agent/{state.py, actions.py, loop.py, verification.py,
           tools/{base.py, registry.py, calculator.py, mcp_tool.py},
           memory/{short_term.py, long_term.py}}
    mcp/{servers/knowledge_server.py, servers/records_server.py, client.py}
    multi_agent/{supervisor.py, workers.py, synthesizer.py}
    evaluation/{dataset.py, rag_metrics.py, agent_metrics.py, run_eval.py}
    tracing.py
  data/{documents/*.md, records.json, benchmark.json}
  scripts/{ingest.py, demo_single_agent.py, demo_multi_agent.py, demo_mcp.py, run_evaluation.py}
  api.py
  tests/ (unit + `@pytest.mark.integration`)
```

## 18. Important interfaces
`LLMClient.decide(prompt, schema) -> BaseModel`; `Embedder.embed(texts) ->
np.ndarray`; `VectorIndex.add(ids, vectors)/search(query_vec, k)`;
`Reranker.rerank(query, candidates, top_n)`; `Tool.name/description/schema/
run(**kwargs)`; `ToolRegistry.register/list_schemas/execute(name, args)`;
`AgentRuntime.run(query, state) -> AgentResult`.

## 19. Major data structures
`Document, Chunk, RetrievalResult(chunk, score, rank)`; `AgentState(query,
steps: list[Step], status)`, `Step(action, input, observation)`;
`ActionDecision(thought, action_type, tool_name?, arguments?, answer?)`
(pydantic, schema-constrained LLM output); `ToolSchema(name, description,
parameters, origin: local|mcp)`; `ToolResult(ok, content, error?)`;
`VerificationResult(grounded, issues, confidence)`; `MemoryItem(text,
source, ts)`; `EvalQuery(id, question, expected_doc_ids, expected_answer,
category, difficulty)`.

## 20. Configuration
Single `.env` + `Settings(BaseSettings)`: `OLLAMA_HOST, OLLAMA_MODEL,
EMBEDDING_MODEL, RERANKER_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVE,
TOP_K_RERANK, MAX_AGENT_STEPS, MAX_DECISION_RETRIES`. No secondary config
system.

## 21. Test strategy
Unit tests (fast, no Ollama needed — fake deterministic `LLMClient`):
chunking behavior/overlap, vector index ranking correctness, empty-retrieval
handling, tool registry registration/lookup, agent state transitions,
termination on max steps, malformed-decision retry/repair, memory
persistence/isolation. Integration tests (`@pytest.mark.integration`,
require Ollama + MCP servers running): full agent loop end-to-end, MCP
discovery→invocation round trip, multi-agent delegation.

## 22. Evaluation dataset strategy
~18 hand-authored queries over a synthetic engineering-knowledge corpus I
write myself (so ground truth is exact): categories = pure document lookup,
structured record lookup, arithmetic/tool use, multi-hop (needs 2 tools),
and out-of-corpus (no valid evidence exists — must refuse, not hallucinate).
Each row: `question, expected_doc_ids, expected_answer, category,
difficulty`. Small by design — illustrative, not statistically rigorous,
and documented as such.

## 23. Metrics
RAG: Recall@k, MRR, groundedness (fraction of answer claims traceable to a
cited chunk), citation correctness. Agent: task success (rubric match vs.
expected_answer), tool selection accuracy, unnecessary tool calls, step
count, failure rate. System: latency, tool-call count. Single-agent vs.
multi-agent compared on the same benchmark.

## 24. Minimal demo scenarios
1. Pure document RAG question. 2. Local calculator tool question. 3. MCP
records-lookup question. 4. Multi-hop needing both knowledge + records. 5.
Out-of-corpus question (must refuse/ground-fail gracefully). 6. Multi-turn
conversation exercising short- and long-term memory. 7. Same query run
single-agent vs. multi-agent, traces compared side by side.

## 25. CV-worthy contributions (measured, see EVALUATION.md)
Built a from-scratch agentic RAG runtime with explicit state/tool
execution/memory/verification loops; implemented MCP tool discovery and
invocation against two custom MCP servers, with document retrieval itself
routed through the protocol; built a supervisor multi-agent architecture
and measured it against the single-agent path on a hand-built 20-query
benchmark; built an evaluation harness covering retrieval (Recall@k, MRR,
naive vs reranked), groundedness, citation correctness, tool selection,
task success, search efficiency, and latency; then ran a Retrieval Budget
& Search Efficiency experiment showing that most of the multi-agent
advantage is reproduced by capping the single agent at one retrieval
(0.70 -> 0.90 task success), i.e. a constraint effect rather than an
orchestration effect.

The strongest CV line here is the last one, because it is a result about
one's own design being partly wrong - that reads as engineering judgment
rather than salesmanship. Claim all of it with the caveats attached:
20 queries, one 7B model, run-to-run variance up to 0.15, and reranking
showed no measurable benefit at this corpus size. Being able to explain
*why* each number came out as it did (EVALUATION.md's "Reading these
numbers honestly") is worth more in an interview than the numbers.

## 26. Interview questions this must make answerable
Reused verbatim from the request's Part 15 list; THEORY.md will map each
question to the specific file/function that answers it.

## 27. Complexity/risk assessment
Small local model (chosen: to be pulled — see below) may be inconsistent
even under schema-constrained decoding → mitigated with bounded retry +
graceful degradation, and this is itself a documented, demonstrable finding
about agentic systems on small models, not just a risk. MCP stdio subprocess
management on Windows needs early smoke-testing. Scope is large relative to
"small project" — mitigated by the exclusions in §6 and by building the
smallest version of each concept rather than the most complete one.
