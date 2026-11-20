# Evaluation Methodology & Statistical Protocols

This document defines the evaluation methodology, benchmarking protocols, metrics, statistical significance tests, experimental controls, and falsification criteria used across the project.

---

## 1. Benchmarking Datasets & Split Design

### 1.1 Knowledge Base Benchmark (`data/benchmark.json`)
Consists of 20 hand-crafted evaluation questions testing RAG retrieval and tool integration over the 31-document synthetic Nimbus corpus:
- **`pure_document` (5 questions):** Information retrieval and textual synthesis strictly from documentation chunks.
- **`structured_record` (4 questions):** Queries requiring entity lookup from structured database/records rather than free text.
- **`tool_arithmetic` (4 questions):** Numerical calculations requiring the calculator tool.
- **`multi_hop` (4 questions):** Questions requiring multi-source correlation (e.g. document spec combined with a database record).
- **`out_of_corpus` (3 questions):** Questions asking about unrecorded entities; evaluates appropriate refusal rather than hallucination.

### 1.2 Business Domain Benchmarks (`domains/*/benchmark.json`)
Each domain contains 8–12 questions with annotated `expected_numbers` and `expected_policy_status` (`ALLOWED`, `BLOCKED`, `REQUIRES_APPROVAL`):
- Credit Underwriting: 10 questions (`domains/credit_risk/benchmark.json`)
- Enterprise Operations: 10 questions (`domains/enterprise_operations/benchmark.json`)
- Multimodal Investigation: 9 questions (`domains/multimodal_investigation/benchmark.json`)
- Product Growth Analytics: 8 questions (`domains/product_analytics/benchmark.json`)

---

## 2. Metrics & Formal Definitions

### 2.1 Retrieval Metrics
- **Recall@k:** The proportion of queries where at least one ground-truth chunk appears in the top $k$ retrieved results.
  $$\text{Recall}@k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{rank}(q) \le k)$$
- **Mean Reciprocal Rank (MRR):**
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

### 2.2 Verifier Metrics
- **Precision:** The proportion of accepted claims/citations that are genuinely true/supported.
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall:** The proportion of genuinely true/supported claims/citations that are accepted.
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **Fabrication Acceptance Rate (False Positive Rate):**
  $$\text{FPR} = \frac{FP}{FP + TN}$$
- **Correct-Answer Acceptance Rate:** The verifier must not trade false-positive suppression for false rejections; correct ground-truth answers must achieve 100% acceptance.

### 2.3 Agent Performance Metrics
- **Task Success Rate:** Fraction of runs where the final answer satisfies all criteria evaluated by an independent judge.
- **Answer Verified Rate:** Fraction of runs where `AgentState.answer_verified == True`.
- **Policy Alignment:** Fraction of runs where the policy engine's verdict matches `expected_policy_status`.
- **Search Redundancy:** Ratio of repeated identical retrieval queries across steps:
  $$\text{Redundancy} = 1 - \frac{|\text{Unique Queries}|}{|\text{Total Queries}|}$$

---

## 3. Statistical Significance Testing

### 3.1 Two-Proportion Z-Test (A/B Experimentation)
- **Application:** Product analytics automated rollout decisions (`test_significance`).
- **Hypothesis:** $H_0: p_1 = p_2$ versus $H_1: p_2 > p_1$.
- **Test Statistic:**
  $$Z = \frac{\hat{p}_2 - \hat{p}_1}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$
- **Decision Rule:** Autonomous rollout requires $p < 0.05$ AND difference $> 0$.

### 3.2 Bootstrap Confidence Intervals
- **Application:** Performance benchmark comparisons between Single-Agent and Multi-Agent modes across seeds.
- **Protocol:** Resample evaluation runs $B = 10,000$ times with replacement. Calculate the 95% empirical percentile confidence interval $[\theta_{0.025}^*, \theta_{0.975}^*]$.

---

## 4. Experimental Controls & Hygiene

1. **Session Memory Reset:** Between independent benchmark questions, the harness must invoke `runtime.reset_session()`. Without this, short-term history leaks into subsequent prompts, contaminating evaluation results.
2. **Deterministic LLM Drivers in CI:** To make CI builds reliable and independent of external API quotas or GPU availability, the standard test suite uses scripted `FakeLLMClient` instances that replay realistic model outputs.
3. **Offline Hermeticity:** Model downloads must be caught cleanly (`pytest.skip(..., allow_module_level=True)`), ensuring tests never fail because of transient network or Hugging Face outages.
4. **Dynamic Table Rendering:** Results tables in `README.md` and `EVALUATION.md` must be rendered via `scripts/render_readme.py` and validated by `scripts/render_readme.py --check`. No number is entered manually.

---

## 5. Falsification Criteria

Every core claim of the runtime is falsifiable by automated tests:

| Claim | What Would Falsify It | Test Enforcing It |
| :--- | :--- | :--- |
| **Fail-Closed Target Binding** | If a proposed target does not match the request entity, or if no entity is found, the verdict must be `REQUIRES_APPROVAL`. | `tests/test_policy.py::test_target_binding_probes_reject_swaps` |
| **Numerical Verifier Grounding** | If an entity swap (e.g. desktop for mobile) or metric swap (purchases for checkouts) is accepted as valid. | `tests/test_numerical_verification.py::test_section_0_2_numerical_verifier_probes` |
| **Contradiction-Aware Citations** | If a cited document stating a 5s timeout accepts a claim of "no timeout and runs indefinitely". | `tests/test_verification.py::test_contradiction_guards_reject_opposing_numbers_and_negations` |
| **Permit-Tool Pinning** | If a spoofed MCP server declaring a free-SQL tool as `RECORD` produces an `ALLOWED` verdict. | `tests/test_credit_risk_flow.py::test_spoofed_server_declaring_free_sql_as_record_cannot_permit` |
| **Offline Test Safety** | If running pytest with an empty cache and no network raises unhandled network exceptions. | `tests/test_mcp_retriever.py` module-level skip |
