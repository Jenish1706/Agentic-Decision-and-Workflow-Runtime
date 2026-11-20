# Start Here: Agent Runtime Guide

## 1. Threat Model: What Goes Wrong When an LLM Acts

When an autonomous LLM interacts with external tools and executes real-world business actions, five distinct failure modes emerge:
1. **Hallucinated Authority:** The model manufactures permissions or executes arbitrary SQL expressions (`SELECT 780 AS credit_score`) to authorize its own sensitive proposals.
2. **The Confused Deputy:** An attacker requests an action on entity A, but the model proposes actions on entity B (or manipulates file paths), bypassing security checks.
3. **Fabricated Numbers:** The model invents plausible metrics, swaps numbers across segments (e.g. attributing desktop conversion to mobile), or inverts change directions.
4. **Misattributed Citations:** The model cites chunks that contradict its claims, match only superficial stopwords, or invert negation.
5. **Runaway Compute:** The model enters infinite observation loops on ambiguous queries, exhausting token, retrieval, or analytical SQL budgets.

---

## 2. The Invariant

> **The LLM proposes actions, parameters, and textual reasoning; deterministic code verifies facts and decides authority.**

The LLM is strictly confined to intent drafting. Tool execution, policy evaluation, and claim verification run through deterministic code with zero LLM self-evaluation.

---

## 3. Evidence: Immutable Ledger with Provenance

Observations from external tool calls (SQL databases, record APIs, document retrieval) are stored in an append-only `EvidenceLedger`:
- **Deep-Frozen Immutability:** Observations are recursively deep-frozen using `MappingProxyType`, preventing downstream tampering.
- **Asymmetric Permission Provenance (`can_permit`):** Evidence carries an explicit `can_permit` boolean. Unverified or model-shaped queries (such as analytical SQL) have `can_permit = False`. They can restrict (block or escalate) decisions, but only authoritative sources can autonomously permit an action.

---

## 4. Authority: Policy Lattice, Fail-Closed Target Binding, Client-Pinned Permit Tools

- **The Policy Lattice:** Decisions evaluate strictly via deterministic domain rules across a join-semilattice with unambiguous precedence:
  $$\text{BLOCKED} > \text{REQUIRES\_APPROVAL} > \text{ALLOWED}$$
- **Fail-Closed Target Binding:** Proposed targets must match the entity requested in the user prompt, resolved from manifest `target_params`. If parameter entities differ or no entity is found, the engine fails closed to `REQUIRES_APPROVAL`.
- **Client-Pinned Permit Tools:** Authority comes exclusively from `permit_tools` declared in the client's domain manifest (`domain_manifest.yaml`). Server-declared categories (e.g. an untrusted MCP server claiming `category: "record"`) are ignored.

---

## 5. Answer Checks: Numbers and Citations

Before an answer is returned, deterministic verification gates check every claim against the evidence ledger:
- **Numerical Verification:** Clauses are deconstructed and checked against evidence bounds, domain synonyms, and opposing metric definitions. Plausible near-miss fabrications remain the known challenge (especially relative changes), while random claims and category swaps are reliably rejected.
- **Citation Verification:** Guarded against negation swaps, number contradictions, and semantic opposition using local NLI cross-encoders with offline lexical fallbacks.

<!-- BEGIN_GUIDE00_RESULTS -->
All measurements are verified and rendered directly by `scripts/render_readme.py`:
- **Numerical Claim Verification:** Correct answers are accepted at **100.0%** across baseline test suites, while rejecting 100% of entity swaps, metric swaps, and synonym swaps. Random fabricated claims are accepted at 2.9% (59/2,000 trials). Near-miss fabricated values in the plausible range are accepted at higher rates: 6.8% for plausible percentages, 23.9% for percentage-point gaps, and 50.4% for relative changes (2,000 trials each). Development probe sets (Sets 1 & 2, used while building fixes) show 0/20 wrong claims accepted; on fresh held-out probes (Set 3), 2/15 wrong claims are accepted (period swaps) and 2/15 correct claims are falsely rejected prior to AR-15. See full breakdown in [README.md#numerical-verification-benchmark](../../README.md#numerical-verification-benchmark) and [EVALUATION.md](../../EVALUATION.md).
- **Citation Entailment Verification:** In offline lexical fallback mode, precision is 83.3% (benchmark, 6/30 contradictions accepted) and 83.3% (held-out); with local NLI enabled, precision reaches 90.3% (benchmark, 3/30 contradictions accepted) and 100.0% (held-out). See [README.md#citation-verification-benchmark](../../README.md#citation-verification-benchmark).
- **Deterministic Policy Enforcement:** Tested across four business domains (Credit Underwriting, Enterprise Systems Operations, Multimodal Business Operations, and Product Growth Analytics), the policy engine achieves 0 false autonomous approvals on spoofed records and malicious SQL projections.
<!-- END_GUIDE00_RESULTS -->

---

## 6. Evaluation Integrity: Contamination Bugs, Retractions, and LLM Measurement Status

- **Memory Contamination Retraction:** An audit revealed that earlier benchmark runs failed to reset short-term memory between evaluation questions, leaking context and inflating multi-turn scores. The harness now mandates deterministic `runtime.reset_session()` between runs, and contaminated scores were retracted.
- **LLM Measurement Status:** There is currently **no end-to-end LLM measurement** beyond the 5-question smoke run (`data/eval_report.json`). Large-scale live evaluations (`scripts/run_cross_domain_eval.py` and `scripts/run_budget_experiment.py`) require a configured Ollama instance and remain deferred. All reported benchmark figures for verifiers and policies reflect deterministic code measurements over benchmark datasets.

---

## 7. Known Limits

- **Derivation Coincidences:** Pairwise combinations across multi-row evidence tables can derive values that match fabricated relative changes (currently accepted at significant rates in the plausible range).
- **Entity Vocabulary:** Subject entity extraction relies on declared domain manifest synonyms and evidence vocabulary; terms outside this vocabulary fail closed.
- **Benchmark Size:** Hand-crafted benchmarks isolate specific architectural failure modes (60 citation pairs, 20 held-out pairs, 6,000 synthetic trials, and domain suites) rather than forming broad web-scale leaderboards.

---

## 8. Reading Order

1. [00_START_HERE.md](00_START_HERE.md) (This file) — Overview, headline results, and 10-minute quickstart.
2. [01_CONCEPTS.md](01_CONCEPTS.md) — Intuition, mathematical formulations, worked numeric examples, and code references for every core concept.
3. [02_ARCHITECTURE.md](02_ARCHITECTURE.md) — Mermaid architecture diagram, component matrix, and end-to-end request lifecycle.
4. [03_DATA.md](03_DATA.md) — Data schemas, row counts, licenses, and quality handling across domains.
5. [04_METHODOLOGY.md](04_METHODOLOGY.md) — Experimental protocols, baselines, statistical tests, and falsification conditions.
6. [05_CODE_WALKTHROUGH.md](05_CODE_WALKTHROUGH.md) — Detailed pipeline breakdown by stage, function signatures, and tricky implementation lines.
7. [06_RUNBOOK.md](06_RUNBOOK.md) — Step-by-step setup, test execution, benchmark regeneration, and troubleshooting.
8. [07_TESTING.md](07_TESTING.md) — Test suite map, coverage map, offline skip mechanisms, and how to write tests.
9. [08_DESIGN_DECISIONS.md](08_DESIGN_DECISIONS.md) — Architectural Decision Records (ADRs) for the 10 major architectural choices.
10. [09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md) — Complete experimental results, confidence intervals, negative results, and threat model boundaries.
11. [GLOSSARY.md](GLOSSARY.md) — Definitions for all abbreviations, acronyms, and specialized terminology.

---

## 9. 10-Minute Quickstart

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or standard `pip` / `venv`
- Git

### Installation
Clone the repository and sync the virtual environment:
```bash
# Clone the repository
git clone <repository-url> agent-runtime
cd agent-runtime
uv sync
```

### Run the Test Suite (Fast, Offline-Safe)
The full test suite uses scripted deterministic LLM drivers and offline-safe fixtures, completing in under 2 minutes with no external API calls:
```bash
uv run pytest
```

### Verify Documentation Tables
Confirm that all documentation tables match live verified code runs:
```bash
uv run python scripts/render_readme.py --check
```

### Run an Agent Run Interactively
Run a sample domain query through the agent runtime:
```bash
uv run python -c "
import asyncio
from agent_runtime.domains import load_domain
from agent_runtime.agent.loop import AgentRuntime
from agent_runtime.agent.tools.registry import ToolRegistry
from tests.fakes import FakeLLMClient
from agent_runtime.agent.actions import ActionDecision, ActionType

async def main():
    domain = load_domain('credit_risk')
    engine = domain.create_policy_engine()
    print(f'Loaded domain: {domain.title}')
    print(f'Permit tools pinned: {domain.permit_tools}')

asyncio.run(main())
"
```
