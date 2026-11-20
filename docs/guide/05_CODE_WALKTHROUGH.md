# Codebase Walkthrough by Pipeline Stage

This document provides a thorough walkthrough of every important module, script, and configuration in the repository. For each stage, we cover:
- Purpose & Architecture
- Inputs and Outputs
- Key Functions & Signatures
- Non-obvious implementation details a reader or reviewer might trip on.

---

## 1. Retrieval-Augmented Generation (RAG) Stage

### `src/agent_runtime/rag/chunking.py`
- **What it does:** Breaks raw Markdown documents into overlapping token/character chunks without splitting words mid-sentence.
- **Inputs & Outputs:** `text: str, chunk_size: int, chunk_overlap: int -> list[str]`
- **Key Function:**
  ```python
  def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]: ...
  ```
- **Non-obvious detail:** When a single token or word is longer than `chunk_size`, the chunker yields the oversized word intact rather than entering an infinite while loop or corrupting UTF-8 characters.

### `src/agent_runtime/rag/embeddings.py`
- **What it does:** Wraps pretrained sentence-transformer models to generate dense float32 vector representations.
- **Inputs & Outputs:** `texts: list[str] -> np.ndarray` (shape: `(N, dim)`, dtype: `float32`)
- **Key Function:**
  ```python
  class Embedder:
      def __init__(self, model_name: str): ...
      def embed(self, texts: list[str]) -> np.ndarray: ...
  ```
- **Non-obvious detail:** Embeddings are L2-normalized on output (`normalize_embeddings=True`). This means cosine similarity simplifies to a simple matrix dot product in `VectorIndex`, eliminating per-query vector norm calculations.

### `src/agent_runtime/rag/vector_index.py`
- **What it does:** In-memory exact nearest-neighbor vector search over dense embeddings.
- **Inputs & Outputs:** `query_vector: np.ndarray, top_k: int -> list[tuple[str, float]]`
- **Key Function:**
  ```python
  class VectorIndex:
      def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
      def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]: ...
  ```
- **Non-obvious detail:** Dot product scores are clipped to `[-1.0, 1.0]` before calculating similarities to prevent floating-point inaccuracies from producing `cos(theta) > 1.0`.

### `src/agent_runtime/rag/reranker.py`
- **What it does:** Cross-encoder joint scoring of `(query, document)` pairs to reorder top candidates retrieved by bi-encoder search.
- **Inputs & Outputs:** `query: str, chunks: list[Chunk], top_k: int -> list[RetrievedChunk]`
- **Key Function:**
  ```python
  class Reranker:
      def rerank(self, query: str, chunks: list[Chunk], top_k: int = 3) -> list[RetrievedChunk]: ...
  ```
- **Non-obvious detail:** If `chunks` is empty or shorter than `top_k`, the reranker gracefully slices available candidates without raising indexing errors.

---

## 2. Tool Execution & MCP Layer

### `src/agent_runtime/agent/tools/registry.py`
- **What it does:** Manages tool schemas, namespace registration, and invocation dispatch.
- **Key Functions:**
  ```python
  class ToolRegistry:
      def register(self, tool: Tool) -> None: ...
      async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...
  ```
- **Non-obvious detail:** Registration enforces strict uniqueness. If two MCP servers attempt to register a tool with the same name (e.g. `execute_sql`), the registry raises immediately rather than silently overwriting, preventing silent tool hijacking.

### `src/agent_runtime/mcp_layer/client.py`
- **What it does:** Manages child process lifecycles for external domain MCP servers communicating over stdio JSON-RPC.
- **Key Class:** `MCPToolClient`
- **Non-obvious detail:** Windows asyncio subprocesses require graceful stdin/stdout teardown to avoid `BrokenPipeError` or lingering zombie Python processes on test termination.

### `src/agent_runtime/mcp_layer/duckdb_tools.py`
- **What it does:** Sanitizes and executes read-only SQL queries against DuckDB instances.
- **Key Function:**
  ```python
  def validate_read_only_sql(sql: str) -> None: ...
  ```
- **Non-obvious detail:** It disallows multiple semicolons and keyword DDL mutations (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `ATTACH`, `COPY`), ensuring agents cannot alter state under the guise of analytical queries.

---

## 3. Evidence Ledger & Fact Resolution

### `src/agent_runtime/agent/evidence.py`
- **What it does:** Implements the append-only, deeply immutable log of all external observations.
- **Key Classes:** `EvidenceItem`, `EvidenceLedger`
- **Key Functions:**
  ```python
  def freeze(value: Any) -> Any: ...
  def thaw(value: Any) -> Any: ...
  ```
- **Non-obvious detail:** `EvidenceItem` is a frozen dataclass, but `dataclasses.frozen=True` only prevents reassignment of top-level attributes—it does not prevent in-place mutation of nested `dict` or `list` values. `freeze()` recursively wraps all mappings in `MappingProxyType` and converts lists to tuples, enforcing true immutability.

### `src/agent_runtime/agent/facts.py`
- **What it does:** Resolves candidate facts for specific entities from the ledger, tracking attribution and permit capabilities.
- **Key Function:**
  ```python
  def resolve_fact(
      ledger: EvidenceLedger | None,
      field: str,
      *,
      entity_key: str | None = None,
      entity_value: Any = None,
  ) -> ResolvedFact: ...
  ```
- **Non-obvious detail:** `can_permit` requires all backing items to be permit-capable (`all(item.can_permit for item in items)`). Free-form SQL queries or unpinned tools have `can_permit=False`. They can supply candidates to block or escalate an action, but `fact.conclusive` evaluates to `False`, preventing autonomous approval.

---

## 4. Agent Loop & Budget Controller

### `src/agent_runtime/agent/loop.py`
- **What it does:** Drives the iterative perception-action-synthesis cycle.
- **Key Class:** `AgentRuntime`
- **Key Function:**
  ```python
  async def run(self, query: str, *, budget: ActionBudget | None = None) -> AgentState: ...
  ```
- **Non-obvious detail:** If deterministic verifiers detect ungrounded numbers or unsupported citations in a draft answer, the runtime injects a synthetic corrective observation into the conversation and continues the loop, giving the model an explicit opportunity to self-correct before final termination.

### `src/agent_runtime/agent/budget.py`
- **What it does:** Tracks per-action budget counters and checks admissibility.
- **Key Class:** `BudgetController`
- **Non-obvious detail:** Counters are isolated per run in `BudgetController`; the immutable `ActionBudget` configuration is never mutated, preventing budget state leakage across sequential queries.

---

## 5. Verification & Entailment

### `src/agent_runtime/agent/verification.py`
- **What it does:** Performs deterministic numerical claim verification and citation entailment checks.
- **Key Functions:**
  ```python
  def verify_numerical_grounding(
      answer: str,
      ledger: EvidenceLedger,
      *,
      query: str = "",
      reference_values: Mapping[str, float] | None = None,
  ) -> VerificationResult: ...

  def verify_citations(
      answer: str,
      citation_map: dict[str, Chunk],
      *,
      has_non_retrieval_evidence: bool = False,
  ) -> CitationVerificationResult: ...
  ```
- **Non-obvious details:**
  1. **Clause Splitting:** Sentences with multiple numbers (e.g. *"Mobile was 4.1% while desktop fell to 1.6%"*) are split into clauses by conjunctions (`while`, `vs`) and punctuation, ensuring `desktop` does not cross-contaminate the `mobile` claim.
  2. **Negation Guards:** Before running heavy NLI models, polarity tokens (`no`, `not`, `without`) are compared. If the cited document is affirmative while the sentence is negated, citation support is rejected deterministically in $O(1)$ time.

---

## 6. Policy Governance & Domain Loading

### `src/agent_runtime/agent/policy.py`
- **What it does:** Deterministic rule engine evaluating proposed decisions against ledger facts.
- **Key Classes:** `PolicyEngine`, `ProposedDecision`, `DecisionContext`
- **Non-obvious detail:** Pre-evaluation enforces `target_entity_binding`: if `proposal.parameters[target]` does not match the canonical entity in the user query, it fails closed to `REQUIRES_APPROVAL`, thwarting prompt injection confused-deputy attacks.

### `src/agent_runtime/domains.py`
- **What it does:** Parses `domain_manifest.yaml` and links domain policies with pinned `permit_tools`.
- **Key Function:**
  ```python
  def load_domain(name: str) -> DomainSpec: ...
  ```
- **Non-obvious detail:** Manifest validation asserts that every documented rule name in YAML exactly matches the registered rule names in Python code. If code and documentation drift, `load_domain()` raises `DomainConfigError` at startup.
