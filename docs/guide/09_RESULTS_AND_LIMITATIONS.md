# Experimental Results, Limitations & Threats to Validity

This document details all verified experimental findings, negative results, threats to validity, and future work.

---

## 1. Verified Results Summary

All result numbers reflect verified code runs rendered via `scripts/render_readme.py` and validated by `scripts/render_readme.py --check`.

### 1.1 Numerical Verification Benchmark
- **Artifact Path:** `scripts/measure_verifier.py`, rendered in [README.md](../../README.md#numerical-verification-benchmark) and [EVALUATION.md](../../EVALUATION.md).
- **Correct-Answer Acceptance:** **100.0%** (0 false rejections across baseline verified answers).
- **Adversarial Category Rejections:**
  - Entity Swaps: **100% rejected** (6/6 rejected).
  - Synonym Swaps: **100% rejected** (6/6 rejected).
  - Metric Swaps: **100% rejected** (6/6 rejected).
  - Direction Reversals: **100% rejected** (2/2 rejected).
- **Synthetic Fabrication Rejections (6,000 trials):**
  - Plausible-range percentages (2,000 trials): **96.6% rejected**
  - Percentage-point gaps (2,000 trials): **99.6% rejected**
  - Relative changes (2,000 trials): **98.2% rejected**

### 1.2 Citation Entailment Benchmark
- **Artifact Path:** `scripts/measure_citations.py`, rendered in [README.md](../../README.md#citation-verification-benchmark).
- **Offline Lexical Fallback Mode:**
  - Benchmark Set (30 pairs): Precision **85.7%**, Recall **100.0%**.
  - Held-Out Set (20 pairs, written before tuning): Precision **83.3%**, Recall **100.0%**.
- **Neural NLI Mode (`cross-encoder/nli-deberta-v3-xsmall`):**
  - Benchmark Set: Precision **90.3%**, Recall **93.3%**.
  - Held-Out Set: Precision **100.0%**, Recall **100.0%**.

### 1.3 RAG Retrieval Quality
- **Artifact Path:** `evaluation/retrieval_metrics.py`, rendered in [EVALUATION.md](../../EVALUATION.md).
- **Recall@3:** 0.98 (Bi-encoder naive) $\to$ 0.98 (Cross-encoder reranked).
- **MRR:** 0.92 (Bi-encoder naive) $\to$ 0.93 (Cross-encoder reranked).

---

## 2. Null & Negative Results

### 2.1 Cross-Encoder Reranking Produced Negligible Uplift
- **Observation:** On the 31-document synthetic Nimbus corpus, cross-encoder reranking improved MRR by only +0.01 (0.92 to 0.93) and produced 0 gain in Recall@3 (0.98).
- **Root Cause Analysis:** The evaluation corpus contains 32 chunks across 31 documents, and benchmark questions closely mirror document vocabulary. The bi-encoder already retrieves the target chunk at rank 1 in over 90% of queries.
- **Takeaway:** Cross-encoder reranking is computationally redundant on small, clean domain corpora; its value emerges primarily on large, noisy, vocabulary-mismatched datasets.

### 2.2 LLM-Judged Faithfulness Inconsistencies
- **Observation:** LLM judges evaluated over open-ended generative answers exhibited score drift across runs and occasionally approved hallucinated reasoning if the tone sounded authoritative.
- **Takeaway:** LLM judges cannot serve as autonomous governance gates in high-stakes workflows; deterministic verification over structured evidence ledgers is strictly required.

---

## 3. Threats to Validity

### 3.1 Construct Validity
- **Synthetic Data Representation:** The DuckDB databases and Markdown documents are synthetically generated. Real-world corporate data exhibits higher noise, duplicate entity identifiers, irregular formatting, and corrupt timestamps.
- **Pre-Extracted Document Metrics:** The multimodal domain operates over pre-extracted JSON fixtures simulating OCR/VLM output rather than processing raw PDF/TIFF images in-process.

### 3.2 Internal Validity
- **Deterministic LLM Fakes in CI:** The automated test suite uses scripted `FakeLLMClient` instances. While this guarantees hermeticity and repeatability, it does not capture stochastic sampling drift of real temperature $> 0$ model inference. Real-model evaluation requires live runs via `scripts/run_evaluation.py`.

### 3.3 External Validity
- **Generalization Across Domains:** The runtime was validated across four enterprise domains (credit underwriting, systems operations, document audit, and product analytics). Novel domains with fundamentally different entity models (e.g. graph-based knowledge networks or continuous temporal control) would require custom entity binding logic.

---

## 4. What To Do Next (Future Work)

1. **Air-Gapped Embedding Quantization:** Ship pre-quantized ONNX or GGUF embedding models within the repository to enable full local neural retrieval in strictly air-gapped CI without Hugging Face downloads.
2. **Formal Logic Verification:** Integrate SMT solvers (e.g. Z3) for complex cross-policy constraint checking where rule sets involve dozens of interdependent conditions.
3. **Decoupled Asynchronous MCP Transports:** Expand MCP transport layers to support secure WebSocket and SSE (Server-Sent Events) for distributed containerized deployment across Kubernetes clusters.
