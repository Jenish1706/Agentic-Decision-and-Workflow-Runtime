# Agentic Decision & Workflow Runtime

A deterministic execution runtime for LLM workflows, document retrieval (RAG), and policy-governed decision pipelines.

> **Core Invariant:** *The LLM proposes actions and reasoning; deterministic code verifies facts and enforces policy.*

For full conceptual documentation, mathematical formulations, architecture diagrams, and runbooks, see the [Public Documentation Guide](docs/guide/00_START_HERE.md).

---

## 1. The Problem & Approach

Generative language models produce plausible but probabilistic tokens: they can invent figures, swap percentages between segments, invert directions, and hallucinate citations. In high-stakes business decisions (such as credit underwriting, production incident response, or feature rollouts), probabilistic models cannot be granted unconstrained authorization authority.

This runtime enforces a strict trust boundary:
- **Immutable Evidence:** Observations from tool calls (SQL databases, record APIs, document retrieval) are deep-frozen in an append-only `EvidenceLedger`.
- **Permission Asymmetry:** Unverified or model-shaped queries (such as free-form SQL) may restrict (block or escalate) an action, but only whitelisted tools (`permit_tools`) can autonomously permit one.
- **Fail-Closed Target Binding:** Proposed decisions must target the entity requested in the user prompt, blocking confused-deputy attacks.
- **Deterministic Numerical Verification:** Quantitative claims in generated answers are split by clause and verified against evidence bounds, domain synonyms, and opposing metric definitions.
- **Citation Entailment:** Citations are guarded against negation swaps, numerical conflicts, and semantic opposition via local NLI cross-encoders with offline lexical fallbacks.

---

## 2. Headline Verified Results

All tables below are generated directly from verified measurement scripts (`scripts/render_readme.py --check`).

### Numerical Claim Verification
Evaluated across 7 categories including 6,000 synthetic trials (plausible shifts, pp gaps, and relative falls) and baseline correct answers (the current verifier accepts 2.9%):

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
| Held-out adversarial probes (Set 1, 20 trials) | 20/20 (100.0%) | 0/20 (0.0%) |
| Held-out adversarial probes (Set 2, 20 trials) | 20/20 (100.0%) | 0/20 (0.0%) |
| Held-out correct paraphrases (16 trials) | 16/16 (100.0%) | 16/16 (100.0%) |
| Direction reversal ("fell" for +1.33 pp) | accepted | rejected |
| Version number in factual sentence ("iOS 17") | accepted | accepted |
| Correct answers accepted (3 benchmark cases) | 3/3 (100.0%) | 3/3 (100.0%) |
<!-- END_VERIFIER_TABLE -->

### Citation Entailment Verification
Measured across 60 hand-labelled benchmark pairs and 20 held-out pairs (written before tuning thresholds):

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

## 3. Quickstart

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

## 4. Documentation Guide

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

## 5. Known Limitations

- **In-Memory Vector Search:** The ~31-document knowledge base uses exact dot-product matrix search. Larger corpora (>100k chunks) require approximate nearest-neighbor indices (HNSW).
- **Simulated Multimodal Extraction:** The multimodal domain processes pre-extracted JSON fixtures rather than running raw vision model OCR in-process.
- **Air-Gapped Embedding Downloads:** Full neural cross-encoder verification requires cached weights; without cached weights, tests gracefully fall back to lexical verification.
- **Hand-Crafted Benchmark Scale:** Benchmark sets are designed to isolate category failures rather than compute broad multi-thousand question leaderboards.

---

## 6. License

This project is licensed under the MIT License.
