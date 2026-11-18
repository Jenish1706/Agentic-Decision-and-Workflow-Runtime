# Agentic Decision & Workflow Runtime

A from-scratch LLM runtime for investigation and decision workflows: RAG,
SQL and statistical tools over MCP, deterministic verification of citations
and numbers, per-run action budgets, policy rules that judge proposed
decisions against evidence, and an audit trace for every run.

> **Central engineering principle**: *LLM proposes; deterministic components verify and constrain.*

Deeper write-ups: [ARCHITECTURE.md](ARCHITECTURE.md) (how it fits together
and why), [THEORY.md](THEORY.md) (interview-style Q&A pointing at code),
[MCP.md](MCP.md) (the protocol layer), [EVALUATION.md](EVALUATION.md)
(benchmarks, measurements, failure modes).

## What this project demonstrates

It does not claim that multi-agent systems, RAG, or budgets are universally
better. It shows how to engineer an LLM workflow with deterministic controls,
and how to measure - and find the bugs in - those controls:

- **Evidence-grounded investigation**: every tool result lands in an
  append-only, deep-frozen evidence ledger with provenance.
- **Structured analytics**: read-only DuckDB SQL and a two-proportion z-test,
  served over MCP.
- **Deterministic numerical verification**: claims (%, pp, currency, counts)
  must match evidence at the precision and unit they were written in, in a
  consistent direction.
- **Policy-constrained decisions**: rules read facts from evidence scoped to
  the proposal's target - never from the proposal - with strict precedence
  `BLOCKED > REQUIRES_APPROVAL > ALLOWED`, failing closed on missing facts.
- **Resource budgeting**: fixed or complexity-based per-run caps on steps,
  searches, SQL queries and document extractions.
- **MCP tool isolation**: every tool server is a separate process; tools
  declare their category in MCP metadata.
- **Cross-domain reuse**: one runtime, four business domains (product
  analytics, credit risk, invoice/SLA audit, SRE operations), each a
  self-contained folder of server, rules, manifest and benchmark.
- **Honest evaluation**: negative results, run-to-run variance, and the
  measurement bugs found along the way are all documented.

## What's actually built here (not hidden behind a framework)

- **Agent loop & evidence ledger** - explicit decide -> act -> observe ->
  verify. See [agent/loop.py](src/agent_runtime/agent/loop.py) and
  [agent/evidence.py](src/agent_runtime/agent/evidence.py).
- **Policy engine & fact resolution** - zero LLM calls in enforcement. See
  [agent/policy.py](src/agent_runtime/agent/policy.py) and
  [agent/facts.py](src/agent_runtime/agent/facts.py).
- **Numerical and citation verification** - see
  [agent/verification.py](src/agent_runtime/agent/verification.py).
- **Budgets** - see [agent/budget.py](src/agent_runtime/agent/budget.py).
- **RAG** - chunking, embeddings, a brute-force numpy vector index, a
  cross-encoder reranker and citation-tagged context, all hand-written. See
  [rag/](src/agent_runtime/rag/).
- **Tools & MCP** - one registry for local tools and MCP-discovered ones; a
  shared hardened read-only SQL toolkit. See
  [agent/tools/](src/agent_runtime/agent/tools/),
  [mcp_layer/](src/agent_runtime/mcp_layer/) and [MCP.md](MCP.md).
- **Memory** - short-term (bounded recent turns, reset between independent
  questions) and long-term (persisted, similarity-retrieved above a relevance
  floor, written only on an explicit "worth remembering" decision). See
  [agent/memory/](src/agent_runtime/agent/memory/).
- **Multi-agent** - a Supervisor delegating to Retriever/Researcher workers
  with a Synthesizer -> Verifier evaluator-optimizer cycle. See
  [multi_agent/](src/agent_runtime/multi_agent/).
- **Four business domains** - [product_analytics](domains/product_analytics/),
  [credit_risk](domains/credit_risk/),
  [multimodal_investigation](domains/multimodal_investigation/) (invoices and
  charts as pre-extracted JSON fixtures - no image model runs), and
  [enterprise_operations](domains/enterprise_operations/), loaded and
  validated by [domains.py](src/agent_runtime/domains.py).
- **Evaluation** - retrieval, groundedness, citations, task success, tool
  use, search efficiency, expected-number recall and policy alignment. See
  [evaluation/](src/agent_runtime/evaluation/) and [EVALUATION.md](EVALUATION.md).

## Measured results - read the caveat first

The previous revision reported, over a 20-question knowledge-base benchmark
with `qwen2.5:7b-instruct`, that the multi-agent path beat the single agent on
task success (1.00 vs 0.80) and that capping the single agent at one
retrieval closed most of that gap (0.90), suggesting a *constraint* effect
rather than an orchestration effect.

**Those LLM-measured numbers are not reliable.** A later audit found that the
evaluation harnesses never reset conversation memory between benchmark
questions, so each question's prompts contained earlier questions and
answers - for the single agent at every step, for the multi-agent path only
inside its Researcher worker. The numerical verifier used at the time also
accepted about half of randomly fabricated numbers (50.9% of 2,000 in a later
measurement; the current verifier accepts 4.0%). The findings are kept in
[EVALUATION.md](EVALUATION.md) as hypotheses to re-measure, not as results;
re-run them with the commands below before quoting any of them.

### Deterministic Verifier Measurements

<!-- BEGIN_VERIFIER_TABLE -->
| Verifier Check | Baseline (b13a25d) | Current (Deterministic) |
|---|---|---|
| Random fabricated claims accepted (2,000 trials) | 1,018 (50.9%) | 79 (4.0%) |
| Out-of-range claims accepted (2,000 trials) | 67 (3.4%) | 0 (0.0%) |
| Entity swaps accepted (3 trials) | 3 (100.0%) | 0 (0.0%) |
| Direction reversal ("fell" for +1.33 pp) | accepted | rejected |
| Version number in factual sentence ("iOS 17") | accepted | accepted |
<!-- END_VERIFIER_TABLE -->

<!-- BEGIN_CITATION_TABLE -->
| Citation Verification Metric | Lexical Overlap Verifier |
|---|---|
| Hand-labelled benchmark pairs | 60 (30 supported, 30 unsupported) |
| Recall (supported citations preserved) | 100.0% (30/30) |
| Out-of-domain hallucinations rejected | 100.0% (15/15) |
| In-domain topical overlap false positives | 50.0% (15/30) |
| Precision | 66.7% (30/45) |
| Overall Accuracy | 75.0% (45/60) |
<!-- END_CITATION_TABLE -->

Unaffected by those bugs: retrieval quality (Recall@k 0.98; MRR 0.92 naive ->
0.93 reranked - deterministic, no LLM involved) and the failure modes
documented in EVALUATION.md. Measured on this revision: real-model smoke
runs of five domain decision questions, whose failures (invented citations,
copied cross-domain examples, decisions described but never attached,
calculator thrashing) each led to a fix; in the last run 4 of 5 policy
verdicts matched the benchmark's expectation - one run, not a benchmark
result.

## Setup

Requires [Ollama](https://ollama.com) running locally (this project's LLM
backend is local-only, by design - no API key needed) and
[uv](https://docs.astral.sh/uv/) for Python dependency management.

```bash
ollama pull qwen2.5:7b-instruct
uv sync
cp .env.example .env   # optional - defaults already work
```

Ollama must be *running* (not just installed) for anything except the test
suite. On Windows the desktop app starts it; otherwise run `ollama serve` in
its own terminal and leave it open. A `ConnectionError: Failed to connect to
Ollama` means it is not running.

## Running it

```bash
# Raw MCP protocol demo: discovery + invocation, no agent involved
uv run python scripts/demo_mcp.py

# Single-agent runtime over 5 representative questions, full trace printed
uv run python scripts/demo_single_agent.py

# Same questions through the multi-agent supervisor path
uv run python scripts/demo_multi_agent.py

# Knowledge-base evaluation: single-agent vs multi-agent (~20 questions)
uv run python scripts/run_evaluation.py

# Retrieval Budget & Search Efficiency experiment (4 budgets x ~20 questions)
uv run python scripts/run_budget_experiment.py

# Product analytics budgeting experiment (unconstrained vs fixed vs adaptive)
uv run python scripts/run_product_analytics_experiment.py

# Cross-domain evaluation over every domain benchmark
uv run python scripts/run_cross_domain_eval.py            # --domain credit_risk --sample 3

# Numerical verifier vs the original implementation (deterministic, no Ollama needed)
uv run python scripts/measure_verifier.py

# Optional thin HTTP wrapper (stateless per request)
uv run uvicorn api:app --reload
```

## Tests and lint

```bash
uv run pytest        # fast, no Ollama needed: a scripted fake LLM drives the real loop, servers and policies
uvx ruff check .     # correctness, bugbear and import-order rules (see pyproject.toml)
```

## Repository layout

```
domains/
  <name>/                    domain_manifest.yaml, server.py (MCP), policies.py, benchmark.json, data/
  product_analytics/         checkout funnel + A/B experiment warehouse (DuckDB)
  credit_risk/               credit underwriting database (DuckDB)
  multimodal_investigation/  invoice and chart fixtures
  enterprise_operations/     incident registry and service configurations

src/agent_runtime/
  config.py                  settings (Ollama model, chunk sizes, step limits, ...)
  llm/                       LLM interface + Ollama implementation
  rag/                       documents -> chunking -> embeddings -> vector index -> reranker -> pipeline
  agent/                     state, actions, loop, evidence, facts, policy, budget, verification, tools, memory
  mcp_layer/                 MCP client, shared read-only SQL toolkit, knowledge + records servers
  domains.py                 domain loader and manifest validation
  multi_agent/               supervisor, workers (retriever, researcher), synthesizer
  evaluation/                metrics, benchmark loader, evaluation and experiment harnesses
  runtime_factory.py         builds runtimes identically for scripts, API and evaluations
  tracing.py                 human-readable execution traces
data/                        Nimbus knowledge-base corpus, records, benchmark, experiment reports
scripts/                     demos, experiment runners, corpus generator
tests/                       unit and integration tests (fake LLM, real servers and policies)
api.py                       optional FastAPI wrapper
```

## Known simplifications

Documented here rather than left implicit:

- **No persisted vector index.** The ~30-document corpus embeds in a couple
  of seconds, so the index is rebuilt in memory on every run. At a larger
  corpus size you'd persist and incrementally update it instead.
- **Brute-force vector search**, not FAISS/HNSW - the right call at this
  corpus size and the wrong one well past it (see
  [ARCHITECTURE.md](ARCHITECTURE.md) for where that line falls).
- **MCP servers rebuild their own state at subprocess startup** (a few
  seconds for the knowledge server), since they share no process memory
  with the client. A production deployment would run them as long-lived
  services.
- **"Multimodal" extraction is simulated** over JSON fixtures; the domain
  exercises everything downstream of extraction.
- **Benchmarks are small and hand-written** (~20 knowledge-base questions,
  8-12 per domain) - enough to find failure modes and compare configurations
  directionally, not enough for statistical claims. See
  [EVALUATION.md](EVALUATION.md).
