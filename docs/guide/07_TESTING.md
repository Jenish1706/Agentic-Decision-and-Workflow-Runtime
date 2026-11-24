# Testing Architecture & Coverage Guide

This document describes the test suite architecture, module coverage map, design omissions, offline execution mechanisms, and guidelines for adding new tests.

---

## 1. Test Suite Architecture

The test suite is built on `pytest` and `pytest-asyncio`. It is designed to be **hermetic, fast, and deterministic**:
- **Execution Speed:** ~272 tests complete in 90–120 seconds.
- **Zero Remote Dependencies:** All LLM decisions in standard tests are driven by deterministic scripted fakes (`FakeLLMClient`).
- **Real Code Under Test:** Unlike mocked suites, tests execute real database queries against DuckDB, real evidence ledger immutability freezes, real domain policy modules, and real numerical verifiers.

---

## 2. Module Coverage Map

| Test Module | Tests | Primary Subsystem Covered | Key Behaviors Verified |
| :--- | :--- | :--- | :--- |
| `tests/test_agent_loop.py` | 19 | `AgentRuntime`, `AgentState` | Immediate final answers, multi-step loops, step timeouts, error handling |
| `tests/test_budget.py` | 9 | `BudgetController`, `ActionBudget` | Per-action ceilings, exhaustion errors, state immutability across runs |
| `tests/test_calculator.py` | 7 | `CalculatorTool` | Expression evaluation, operator constraints, division-by-zero handling |
| `tests/test_chunking.py` | 6 | `rag/chunking.py` | Boundary splits, word boundaries, overlap consistency, oversized token handling |
| `tests/test_credit_risk_flow.py` | 17 | `domains/credit_risk/` | Underwriting rules, DTI bounds, spoofed tool rejection, target entity binding |
| `tests/test_enterprise_operations_flow.py` | 15 | `domains/enterprise_operations/` | Active P0 freezes, blast-radius caps, TLS baseline, debug port detection |
| `tests/test_multimodal_flow.py` | 12 | `domains/multimodal_investigation/` | Invoice math reconciliation, duplicate invoice detection, SLA uptime breaches |
| `tests/test_product_analytics_flow.py` | 9 | `domains/product_analytics/` | Statistical significance gating, conversion p-values, traffic caps |
| `tests/test_cross_domain_pipeline.py` | 5 | Cross-domain execution | Uniform runtime pipeline driving all four distinct business domains |
| `tests/test_evidence.py` | 8 | `EvidenceLedger`, `EvidenceItem` | Immutability, deep freezing of dicts/lists, numeric flattening, ID sequence |
| `tests/test_facts.py` | 16 | `agent/facts.py` | Scoped entity attribution, CONFLICT/MISSING states, permit capability checks |
| `tests/test_mcp_retriever.py` | 2 | `agent/mcp_retriever.py` | MCP knowledge server integration, unique chunk ID generation, offline skips |
| `tests/test_mcp_servers.py` | 8 | `mcp_layer/` | Tool registration, stdio transports, schema exposure, duckdb read-only guards |
| `tests/test_memory.py` | 7 | `agent/memory/` | Short-term turn eviction, long-term vector similarity search, persistence |
| `tests/test_multi_agent.py` | 9 | Multi-agent `Supervisor` | Worker routing, task decomposition, synthesis verification |
| `tests/test_numerical_verification.py` | 17 | `agent/verification.py` | Section 0.2 probes, entity swaps, metric oppositions, clause attribution |
| `tests/test_policy.py` | 29 | `agent/policy.py` | Section 0.2 target binding probes, delimiter normalization, fail-closed rules |
| `tests/test_readme_tables.py` | 2 | `scripts/tooling/render_readme.py` | Documentation table synchronization check |
| `tests/test_vector_index.py` | 5 | `rag/vector_index.py` | Top-k similarity search, vector normalization, clipping bounds |
| `tests/test_verification.py` | 8 | `agent/verification.py` | Citation checking, negation contradiction guards, number contradiction guards |

---

## 3. Deliberate Omissions (What We Don't Test Here)

1. **Stochastic LLM Token Generation:** Testing live generative endpoints in CI introduces flaky non-deterministic failures, latency, and token costs. Generative quality is benchmarked separately in `scripts/experiments/run_evaluation.py`.
2. **External Network Transports:** No live internet connections or remote Hugging Face endpoints are called during testing.
3. **Live UI / Frontend Elements:** User interfaces belong to Stage B (`BRIEF_B.md`) and are out of scope.

---

## 4. Fresh-Clone & Offline Skip Mechanics

In an air-gapped environment or when cloning without internet connectivity:
- Pretrained weights for dense embedders or cross-encoders may not be cached locally.
- `tests/test_mcp_retriever.py` wraps server importation in an offline-safe check:
  ```python
  try:
      from agent_runtime.agent.mcp_retriever import MCPRetriever
      from agent_runtime.mcp_layer.servers import knowledge_server
  except Exception as exc:
      pytest.skip(f"Knowledge server models cannot be loaded offline: {exc}", allow_module_level=True)
  ```
- Result: Pytest skips the test gracefully rather than reporting an unhandled crash, guaranteeing 0 test errors in air-gapped CI.

---

## 5. How to Add a New Test

### Adding a Policy Rule Test
1. Create a `DecisionContext` containing an `EvidenceLedger` populated with test evidence.
2. Formulate a `ProposedDecision` with target parameters.
3. Call `engine.evaluate(proposal, context)` and assert on `result.final_status` and `rule_results`.

Example:
```python
@pytest.mark.asyncio
async def test_custom_underwriting_rule():
    ledger = EvidenceLedger(permit_tools=frozenset({"get_account_profile"}))
    ledger.add(
        source=EvidenceSource.RECORD_LOOKUP,
        query_or_input="ACCT-999",
        data={"account_id": "ACCT-999", "credit_score": 700},
        provenance="tool:get_account_profile",
        summary="Customer profile",
    )
    domain = load_domain("credit_risk")
    engine = domain.create_policy_engine()
    # evaluate ...
```
