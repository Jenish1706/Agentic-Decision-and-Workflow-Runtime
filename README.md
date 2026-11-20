# Agentic Decision & Workflow Runtime

A deterministic execution runtime for high-stakes LLM workflows, document retrieval (RAG), and policy-governed decision pipelines.

For complete conceptual documentation, mathematical formulations, architecture diagrams, and runbooks, see the [Public Documentation Guide](docs/guide/00_START_HERE.md).

---

## 1. Threat Model: What Goes Wrong When an LLM Acts

When an autonomous LLM interacts with external tools and executes real-world business actions, five distinct failure modes emerge:
1. **Hallucinated Authority:** The model manufactures permissions or executes constant queries (`SELECT 780 AS credit_score`) to authorize its own sensitive proposals.
2. **The Confused Deputy:** An attacker requests an action on entity A, but the model submits a proposal targeting entity B (or manipulates file paths), bypassing authorization.
3. **Fabricated Numbers:** The model invents plausible metrics, swaps numbers across segments (e.g. attributing desktop conversion to mobile), or inverts change directions.
4. **Misattributed Citations:** The model cites chunks that contradict its claims, match only superficial stopwords, or invert negation.
5. **Runaway Compute:** The model enters infinite observation loops on ambiguous queries, exhausting token, retrieval, or analytical SQL budgets.

---

## 2. The Invariant

> **The LLM proposes actions, parameters, and textual reasoning; deterministic code verifies facts and decides authority.**

The LLM is never granted execution authority. It drafts a `ProposedDecision`. All subsequent fact extraction, policy evaluation, and claim verification run through deterministic code with zero LLM self-evaluation.

---

## 3. Evidence: Immutable Ledger with Provenance

Observations from tool invocations, database queries, and retrieval chunks are collected in an append-only `EvidenceLedger`:
- **Deep-Frozen Immutability:** Data payloads are recursively frozen via `MappingProxyType`, preventing downstream tampering.
- **Asymmetric Permission Provenance (`can_permit`):** Every piece of evidence carries a boolean `can_permit` flag. Exploratory or unverified queries (such as analytical SQL) have `can_permit = False`. They can restrict (block or escalate) decisions, but only authoritative sources can autonomously permit an action.

---

## 4. Authority: Policy Lattice, Fail-Closed Target Binding, Client-Pinned Permit Tools

- **The Policy Lattice:** Decisions evaluate strictly via deterministic domain rules across a join-semilattice with unambiguous precedence:
  $$\text{BLOCKED} > \text{REQUIRES\_APPROVAL} > \text{ALLOWED}$$
- **Fail-Closed Target Binding:** Proposed targets must match the entity requested in the user prompt, resolved from manifest `target_params`. If parameter entities differ or no entity is found, the engine fails closed to `REQUIRES_APPROVAL`.
- **Client-Pinned Permit Tools:** Authority comes exclusively from `permit_tools` declared in the client's domain manifest (`domain_manifest.yaml`). Server-declared categories (e.g. an untrusted MCP server claiming `category: "record"`) are ignored.

---

## 5. Answer Checks: Numbers and Citations

Before an answer is returned, deterministic verification gates check every claim against the evidence ledger:

### Numerical Claim Verification
Near-miss fabrication acceptance is the verifier's primary open challenge: while random claims and category swaps are reliably caught, plausible-range numbers that fall near real ledger values can pass by derivation coincidence:

<!-- BEGIN_VERIFIER_TABLE -->
| Verifier Check | Baseline (b13a25d) | Current (Deterministic) |
|---|---|---|
| Random fabricated claims accepted (2,000 trials) | 1,018 (50.9%) | 59 (2.9%) |
| Out-of-range claims accepted (2,000 trials) | 67 (3.4%) | 0 (0.0%) |
| Plausible-range percentages (1.4-4.3%, 2,000 trials) | 2,000 (100.0%) | 136 (6.8%) |
| Percentage-point gaps (0.1-3.0 pp, 2,000 trials) | 2,000 (100.0%) | 479 (23.9%) |
| Relative changes (1-60%, 2,000 trials) | 2,000 (100.0%) | 1,009 (50.4%) |
| Entity swaps accepted (6 trials) | 6 (100.0%) | 0 (0.0%) |
| Synonym swaps accepted (6 trials) | 6 (100.0%) | 0 (0.0%) |
| Metric swaps accepted (6 trials) | 5 (83.3%) | 0 (0.0%) |
| Development probes: adversarial Set 1 (used while building fix, 20 trials) | 20/20 (100.0%) | 0/20 (0.0%) |
| Development probes: adversarial Set 2 (used while building fix, 20 trials) | 20/20 (100.0%) | 0/20 (0.0%) |
| Development correct paraphrases (16 trials) | 16/16 (100.0%) | 16/16 (100.0%) |
| Held-out adversarial probes (Set 3, pre-fix, 15 trials) | 15/15 (100.0%) | 2/15 (13.3%) |
| Held-out correct claims (Set 3, pre-fix, 15 trials) | 15/15 (100.0%) | 13/15 (86.7%) |
| Direction reversal ("fell" for +1.33 pp) | accepted | rejected |
| Version number in factual sentence ("iOS 17") | accepted | accepted |
| Correct answers accepted (3 benchmark cases) | 3/3 (100.0%) | 3/3 (100.0%) |
<!-- END_VERIFIER_TABLE -->

### Citation Entailment Verification
Evaluated across 60 hand-labelled benchmark pairs and 20 held-out pairs (written before tuning thresholds), comparing offline lexical matching against local neural NLI cross-encoders:

<!-- BEGIN_CITATION_TABLE -->
| Citation Verification Metric | Lexical Fallback (Offline) | NLI Entailment (cross-encoder) |
|---|---|---|
| Hand-labelled benchmark pairs | 60 (30 supported, 30 unsupported) | 60 (30 supported, 30 unsupported) |
| Recall (supported citations preserved) | 100.0% (30/30) | 93.3% (28/30) |
| Out-of-domain hallucinations rejected | 100.0% (15/15) | 100.0% (15/15) |
| Contradictory citations accepted (false pos.) | 6/30 (20.0%) | 3/30 (10.0%) |
| Precision | 83.3% | 90.3% |
| Overall Accuracy | 90.0% | 91.7% |
| Held-out test pairs (20 pairs) | Precision 83.3%, Recall 100.0% | Precision 100.0%, Recall 100.0% |
<!-- END_CITATION_TABLE -->

---

## 6. Evaluation Integrity: Retractions & Measurement Status

- **Memory Contamination Retraction:** An audit revealed that earlier benchmark runs failed to reset short-term memory between evaluation questions, leaking context and inflating multi-turn scores. The harness now mandates deterministic `runtime.reset_session()` between runs, and contaminated scores were retracted.
- **LLM Measurement Status:** There is currently **no end-to-end LLM measurement** beyond the 5-question smoke run (`data/eval_report.json`). Large-scale live evaluations (`scripts/run_cross_domain_eval.py` and `scripts/run_budget_experiment.py`) require a configured Ollama instance and remain deferred. All reported benchmark figures for verifiers and policies reflect deterministic code measurements over benchmark datasets.

---

## 7. Known Limits

- **Derivation Coincidences:** Pairwise combinations across multi-row evidence tables can derive values that match fabricated relative changes (currently accepted at significant rates in the plausible range).
- **Entity Vocabulary:** Subject entity extraction relies on declared domain manifest synonyms and evidence vocabulary; terms outside this vocabulary fail closed.
- **Benchmark Size:** Hand-crafted benchmarks isolate specific architectural failure modes (60 citation pairs, 20 held-out pairs, 6,000 synthetic trials, and domain suites) rather than forming broad web-scale leaderboards.

---

## 8. Quickstart

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or standard `pip`

### Setup & Test Execution
```bash
# Clone the repository
git clone <repository-url> agent-runtime
cd agent-runtime

# Install dependencies in virtual environment
uv sync

# Run full hermetic test suite (283 tests, ~90s, no GPU or Ollama required)
uv run pytest

# Verify documentation consistency
uv run python scripts/render_readme.py --check
```

---

## 9. Documentation Guide

Read the complete documentation in `docs/guide/`:
- [00_START_HERE.md](docs/guide/00_START_HERE.md): Reading order, core questions, and 10-minute walkthrough.
- [01_CONCEPTS.md](docs/guide/01_CONCEPTS.md): Formal formulas, intuitions, worked examples, and code links for 11 core concepts.
- [02_ARCHITECTURE.md](docs/guide/02_ARCHITECTURE.md): Component matrix, Mermaid topology diagram, and request lifecycle.
- [03_DATA.md](docs/guide/03_DATA.md): DuckDB schemas, JSON fixtures, row counts, and data quality handling.
- [04_METHODOLOGY.md](docs/guide/04_METHODOLOGY.md): Experimental protocol, baselines, statistical tests, and falsification rules.
- [05_CODE_WALKTHROUGH.md](docs/guide/05_CODE_WALKTHROUGH.md): Step-by-step module breakdown by pipeline stage.
- [06_RUNBOOK.md](docs/guide/06_RUNBOOK.md): Windows setup, pipeline commands, expected outputs, and troubleshooting.
- [07_TESTING.md](docs/guide/07_TESTING.md): Test suite map, coverage, offline skip mechanisms, and adding tests.
- [08_DESIGN_DECISIONS.md](docs/guide/08_DESIGN_DECISIONS.md): ADRs for 10 major architectural choices.
- [09_RESULTS_AND_LIMITATIONS.md](docs/guide/09_RESULTS_AND_LIMITATIONS.md): Uncertainties, null/negative results, and validity threats.
- [GLOSSARY.md](docs/guide/GLOSSARY.md): Definitions for all terms and acronyms.

---

## 10. License

This project is licensed under the MIT License.
