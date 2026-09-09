# Agentic Decision & Workflow Runtime

A reusable LLM runtime for business investigation and decision workflows, combining RAG, SQL/statistical analysis and multimodal tools with deterministic evidence verification, resource budgets, policy enforcement and audit traces.

> **Central Engineering Principle**: *LLM proposes; deterministic components verify and constrain.*

See [docs/SPEC.md](docs/SPEC.md) for the design rationale, and
[ARCHITECTURE.md](ARCHITECTURE.md) / [THEORY.md](THEORY.md) /
[MCP.md](MCP.md) / [EVALUATION.md](EVALUATION.md) for deeper technical write-ups.

## What this project actually demonstrates

This project is not intended to claim that multi-agent systems, RAG, or action-space constraints are universally superior. Instead, it demonstrates how to engineer an LLM-based business workflow with deterministic controls and evaluate those controls empirically:

- **Evidence-grounded investigation**: structured evidence gathering across documents and databases.
- **Structured analytics**: sandboxed DuckDB SQL querying and statistical hypothesis testing (SciPy z-tests).
- **Deterministic numerical verification**: regex claim extraction and scale reconciliation; detects and rejects unsupported quantitative claims.
- **Policy-constrained decisions**: deterministic rule engine evaluating proposals with strict precedence (`BLOCKED > REQUIRES_APPROVAL > ALLOWED`).
- **Resource budgeting**: complexity-driven action bounds curbing speculative exploration and latency variance.
- **MCP-based tool isolation**: protocol-level process separation over stdio JSON-RPC.
- **Multimodal evidence extraction**: document and invoice line-item reconciliation.
- **Cross-domain reuse**: a shared domain-agnostic runtime core evaluated across four heterogeneous enterprise-style business domains.
- **Empirical evaluation and failure analysis**: systematic comparisons against unconstrained baselines and honest ablation analysis.

## What's actually built here (not hidden behind a framework)

- **Agent loop & Evidence Ledger**: explicit state -> decide -> act -> observe -> terminate.
  Every tool execution (SQL, statistical test, document retrieval, calculation) appends an
  immutable, structured `EvidenceItem` into an `EvidenceLedger` tracking provenance.
  See [src/agent_runtime/agent/loop.py](src/agent_runtime/agent/loop.py) and
  [src/agent_runtime/agent/evidence.py](src/agent_runtime/agent/evidence.py).
- **Deterministic Policy Engine**: evaluates proposed business decisions against
  deterministic rule sets with strict precedence: `BLOCKED > REQUIRES_APPROVAL > ALLOWED`.
  Zero LLM calls in policy enforcement.
  See [src/agent_runtime/agent/policy.py](src/agent_runtime/agent/policy.py).
- **Deterministic Numerical Verification**: mathematical and quantitative claim reconciliation
  (percentages, percentage points, currencies, counts) against ledger facts. Detects and rejects
  unsupported quantitative claims and policy-inconsistent decisions within evaluated workflows.
  See [src/agent_runtime/agent/verification.py](src/agent_runtime/agent/verification.py).
- **Adaptive Budget Controller**: complexity-classified action budgets regulating RAG searches,
  SQL queries, and step limits based on task difficulty (`FACTUAL_LOOKUP`, `STRUCTURED_ANALYTIC`,
  `MULTI_HOP_ANALYSIS`, `INVESTIGATION`).
  See [src/agent_runtime/agent/budget.py](src/agent_runtime/agent/budget.py).
- **Analytics MCP Server & DuckDB**: read-only, isolated DuckDB connection (`enable_external_access=False`)
  exposing table schemas, parameterized SQL queries, and SciPy-powered hypothesis tests
  (two-proportion z-tests for A/B experiment significance).
  See [src/agent_runtime/mcp_layer/servers/analytics_server.py](src/agent_runtime/mcp_layer/servers/analytics_server.py).
- **Four Enterprise-Style Business Domains**:
  1. *Product Analytics & Experimentation*: checkout drop-off funnels, A/B hypothesis tests, statistical significance rollout gates. See [domains/product_analytics/](domains/product_analytics/).
  2. *Credit Risk & Lending Underwriting*: borrower bureau scores, DTI ratio ceilings, delinquency cutoffs, and limit approval caps. See [domains/credit_risk/](domains/credit_risk/).
  3. *Multimodal Business Investigation*: scanned invoice extraction, arithmetic reconciliation, duplicate detection, and visual telemetry charts. See [domains/multimodal_investigation/](domains/multimodal_investigation/).
  4. *Enterprise Systems Operations*: SRE incident telemetry, configuration drift audit, and blast-radius remediation limits. See [domains/enterprise_operations/](domains/enterprise_operations/).
- **RAG**: chunking, embeddings, a brute-force numpy vector index, a
  cross-encoder reranker, citation-tagged context construction - all
  hand-written. See [src/agent_runtime/rag/](src/agent_runtime/rag/).
- **Tools & MCP**: shared registry where local Python tools and MCP servers (`knowledge`,
  `records`, `analytics`, `credit_risk`, `vision`, `ops`) sit behind the exact same interface over stdio.
  See [src/agent_runtime/agent/tools/](src/agent_runtime/agent/tools/) and [MCP.md](MCP.md).
- **Memory**: short-term (recent turns) and long-term (persisted,
  retrieval-backed, written only on an explicit "worth remembering" decision).
  See [src/agent_runtime/agent/memory/](src/agent_runtime/agent/memory/).
- **Multi-agent**: a Supervisor that delegates to Retriever/Researcher
  workers and runs a Synthesizer -> Verifier evaluator-optimizer cycle.
  See [src/agent_runtime/multi_agent/](src/agent_runtime/multi_agent/).
- **Evaluation**: 123 comprehensive tests covering unit, protocol, numerical verification,
  policy governance, budget control, and 4 complete domain integration suites with 100% pass rate.
  See [src/agent_runtime/evaluation/](src/agent_runtime/evaluation/) and [EVALUATION.md](EVALUATION.md).

## Measured results

From `scripts/run_evaluation.py` over a 20-query hand-built benchmark
(`qwen2.5:7b-instruct`, single run per mode). Full numbers, caveats, and
negative results in [EVALUATION.md](EVALUATION.md):

| | Single-agent | Multi-agent |
|---|---|---|
| Task success | 0.80 | 1.00 |
| Groundedness | 0.95 | 1.00 |
| Unnecessary tool calls (avg) | 0.80 | 0.00 |
| Retrieval redundancy | 0.30 | 0.00 |
| Repeated searches (total) | 12 | 0 |
| Latency (avg) | 11.1s | 7.9s |

Retrieval: Recall@k 0.98, MRR 0.92 (bi-encoder) -> 0.93 (reranked).

Across three identical runs single-agent task success spanned 0.65-0.80
and multi-agent 0.95-1.00, so treat sub-0.15 differences as noise. The
search-efficiency numbers were far more stable than the LLM-judged
quality ones.

### Retrieval Budget & Search Efficiency experiment

`scripts/run_budget_experiment.py` caps the single agent's retrievals per
run and measures what extra searching actually buys:

| | budget=1 | budget=2 | budget=3 | unlimited |
|---|---|---|---|---|
| Task success | **0.90** | 0.70 | 0.70 | 0.70 |
| Citation correctness | 1.00 | 1.00 | 0.86 | 0.71 |
| Retrieval redundancy | 0.00 | 0.14 | 0.38 | 0.32 |
| Latency | 10.4s | 11.7s | 11.0s | 13.4s |

More retrieval freedom made **every** measured dimension worse. And since
the multi-agent `RetrieverWorker` searches exactly once by construction,
this reframes the headline result: single-agent at `budget=1` (0.90) is
statistically indistinguishable from multi-agent (0.95), so most of the
multi-agent advantage is a **constraint** effect, not an orchestration
effect.

Three results worth stating plainly because they cut against expectations:
**reranking did not measurably help on this corpus** (too small for the
bi-encoder's top-k to be polluted); **the multi-agent path was faster, not
slower**; and **a retrieval cap acted as a hallucination control** -
out-of-corpus refusals went 1/3 → 3/3 when the agent could no longer keep
searching for facts that don't exist. EVALUATION.md explains all three and
is explicit about what this benchmark does *not* support.

## Setup

Requires [Ollama](https://ollama.com) running locally (this project's LLM
backend is local-only, by design - no API key needed) and
[uv](https://docs.astral.sh/uv/) for Python dependency management.

```bash
ollama pull qwen2.5:7b-instruct
uv sync
cp .env.example .env   # optional - defaults already work
```

Ollama must be *running* (not just installed) for anything except the fast
test suite. On Windows the desktop app starts it; otherwise run `ollama
serve` in its own terminal and leave it open. If a demo dies with
`ConnectionError: Failed to connect to Ollama`, that's what happened.

## Running it

```bash
# Raw MCP protocol demo: discovery + invocation, no agent involved
uv run python scripts/demo_mcp.py

# Single-agent runtime over 5 representative questions, full trace printed
uv run python scripts/demo_single_agent.py

# Same questions through the multi-agent supervisor path
uv run python scripts/demo_multi_agent.py

# Full evaluation harness (~20 queries x single-agent + multi-agent; ~11 min)
uv run python scripts/run_evaluation.py

# Retrieval Budget & Search Efficiency experiment (4 budgets x 20 queries; ~25 min)
uv run python scripts/run_budget_experiment.py

# Product Analytics Action-Space & Budgeting Experiment (Unconstrained vs Fixed vs Adaptive)
uv run python scripts/run_product_analytics_experiment.py

# Cross-Domain Generalization Evaluation across all 4 enterprise domains
uv run python scripts/run_cross_domain_eval.py

# Optional thin HTTP wrapper
uv run uvicorn api:app --reload
```

## Tests

```bash
uv run pytest                           # 123 tests: unit, protocols, verification, policies, all 4 domains
```

## Repository layout

```
domains/
  product_analytics/         DuckDB database, e-commerce checkout schema, 12-task benchmark
  credit_risk/               Credit underwriting database, FICO/DTI/delinquency policies, benchmark
  multimodal_investigation/  Scanned invoices, visual charts, arithmetic fraud policies, benchmark
  enterprise_operations/     SRE incident telemetry, configuration drift audit, blast-radius policies, benchmark

src/agent_runtime/
  config.py                  settings (Ollama model, chunk size, step limits, ...)
  llm/                       provider-agnostic LLM interface + Ollama implementation
  rag/                       documents -> chunking -> embeddings -> vector index -> reranker -> pipeline
  agent/                     state, actions, loop, evidence ledger, policy engine, budget, verification
  mcp_layer/                 MCP client wrapper + MCP servers (knowledge, records, analytics, credit_risk, vision, ops)
  multi_agent/               supervisor, workers (retriever, researcher), synthesizer, verifier
  evaluation/                groundedness, citations, tool metrics, search efficiency, benchmarks
  tracing.py                 human-readable execution traces
data/                        synthetic corpus, structured records, experiment reports
scripts/                     demos, experiment runners, cross-domain eval suite
tests/                       123 unit & integration tests across all 4 domains (100% pass)
```

## Known simplifications

Documented here rather than left implicit:

- **No persisted vector index.** The ~30-document corpus is small enough
  that embedding it from scratch takes a couple of seconds, so the index is
  rebuilt in memory on every run instead of serialized to disk. At a larger
  corpus size you'd persist and incrementally update it instead.
- **Brute-force vector search**, not FAISS/HNSW - the right call at this
  corpus size and the wrong one well past it (see
  [ARCHITECTURE.md](ARCHITECTURE.md) for where that line falls).
- **MCP servers rebuild their own RAG pipeline at subprocess startup** (a
  few seconds), since they share no process memory with the client. A
  production deployment would run them as long-lived services instead of
  spawning fresh per session.
- **The evaluation benchmark is ~20 hand-written queries** - enough to be
  directionally meaningful and to compare single- vs. multi-agent on equal
  footing, not enough to be statistically rigorous. See
  [EVALUATION.md](EVALUATION.md).
