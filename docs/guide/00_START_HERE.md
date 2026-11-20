# Start Here: Agent Runtime Guide

## 1. What This Project Is

This repository implements a deterministic runtime for agentic workflows, document retrieval (RAG), and policy-governed autonomous decision-making. 

Large Language Models (LLMs) are probabilistic token generators: they can hallucinate facts, invent numbers, misattribute claims, and become confused deputies when prompted maliciously. In enterprise decision pipelines—such as credit underwriting, production incident remediation, invoice payments, and feature rollouts—probabilistic models cannot be granted unconstrained authority to authorize real-world actions.

This runtime enforces a strict trust boundary:
- The **LLM proposes** actions, reasoning, and answers.
- An immutable **Evidence Ledger** captures actual observations from external tools (SQL databases, record APIs, document retrieval).
- Deterministic **Verifiers** check that all numerical claims and document citations in the generated answer match ground-truth evidence.
- A deterministic **Policy Engine** evaluates the proposed decision against domain thresholds and entity-bound facts.

## 2. The Core Question It Answers

> **Can an autonomous agent make auditable, safe enterprise decisions without allowing generative hallucinations or prompt injections to forge their own authorization?**

Yes—by enforcing five foundational invariants:
1. **Separation of Proposal from Authorization:** Decisions are evaluated strictly by deterministic rule code, never by the LLM self-scoring its own output.
2. **Entity-Scoped Evidence:** Facts are bound strictly to the entity requested by the user, derived from domain manifests (`domain_manifest.yaml`), rejecting target substitution attacks.
3. **The Permission Asymmetry:** Weak or unverified evidence (such as free-form SQL projections) can restrict (block or escalate) an action, but only authoritative whitelisted sources (`permit_tools`) can autonomously permit one.
4. **Independent Numerical Verification:** Every quantitative claim in the generated text is split by grammatical clause and verified against evidence bounds, domain synonyms, and opposing metric definitions.
5. **Entailment Citation Verification:** Citations are guarded against negation swaps, number contradictions, and semantic opposition using local NLI cross-encoders with offline lexical fallbacks.

## 3. Honest Headline Results

All measurements are verified and rendered directly by `scripts/render_readme.py`:
- **Numerical Claim Verification:** Correct answers are accepted at **100.0%** across baseline test suites, while rejecting 100% of entity swaps, metric swaps, and opposite-direction claims. Across 6,000 synthetic trials (plausible percentage shifts, percentage-point gaps, and relative falls), the verifier detects ungrounded numbers. See full breakdown in [README.md#numerical-verification-benchmark](../../README.md#numerical-verification-benchmark) and [EVALUATION.md](../../EVALUATION.md).
- **Citation Entailment Verification:** The verifier eliminates false-positive citations on contradiction probes (such as negation inversions and number substitutions). In offline lexical fallback mode, precision is 85.7% (benchmark) and 83.3% (held-out); with local NLI enabled, precision reaches 90.3% (benchmark) and 100.0% (held-out). See [README.md#citation-verification-benchmark](../../README.md#citation-verification-benchmark).
- **Deterministic Policy Enforcement:** Tested across four business domains (Credit Underwriting, Enterprise Systems Operations, Multimodal Business Operations, and Product Growth Analytics), the policy engine achieves 0 false autonomous approvals on spoofed records and malicious SQL projections.

## 4. Reading Order

1. [00_START_HERE.md](00_START_HERE.md) (This file) — Overview, headline results, and 10-minute quickstart.
2. [01_CONCEPTS.md](01_CONCEPTS.md) — Intuition, mathematical formulations, worked numeric examples, and code references for every core concept.
3. [02_ARCHITECTURE.md](02_ARCHITECTURE.md) — Mermaid architecture diagram, component matrix, and end-to-end request lifecycle.
4. [03_DATA.md](03_DATA.md) — Data schemas, row counts, licenses, and quality handling across domains.
5. [04_METHODOLOGY.md](04_METHODOLOGY.md) — Experimental protocols, baselines, statistical tests, and falsification conditions.
6. [05_CODE_WALKTHROUGH.md](05_CODE_WALKTHROUGH.md) — Detailed pipeline breakdown by stage, function signatures, and tricky implementation lines.
7. [06_RUNBOOK.md](06_RUNBOOK.md) — Step-by-step setup, test execution, benchmark regeneration, and troubleshooting.
8. [07_TESTING.md](07_TESTING.md) — Test suite architecture, coverage map, offline skip mechanisms, and how to write tests.
9. [08_DESIGN_DECISIONS.md](08_DESIGN_DECISIONS.md) — Architectural Decision Records (ADRs) for the 10 most critical design choices.
10. [09_RESULTS_AND_LIMITATIONS.md](09_RESULTS_AND_LIMITATIONS.md) — Complete experimental results, confidence intervals, negative results, and threat model boundaries.
11. [GLOSSARY.md](GLOSSARY.md) — Definitions for all abbreviations, acronyms, and specialized terminology.

## 5. 10-Minute Quickstart

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or standard `pip` / `venv`
- Git

### Installation
Clone the repository and sync the virtual environment:
```bash
git clone https://github.com/org/agent-runtime.git
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
