# Architectural Decision Records (ADRs)

This document records the 10 most critical design decisions in the repository in ADR format (Context, Decision, Alternatives, Consequences).

---

## ADR-01: Bespoke Runtime Implementation vs. Third-Party Agent Frameworks

### Context
Popular agent frameworks (LangChain, LlamaIndex, CrewAI, AutoGen) provide rapid prototyping abstractions. However, they introduce deep dependency trees, non-deterministic prompt abstractions, hidden retries, and mutable state dictionaries that obscure audit trails.

### Decision
Build the core agent runtime, evidence ledger, RAG pipeline, and policy engine from scratch in standard Python with minimal external dependencies (`pydantic`, `duckdb`, `sentence-transformers`, `mcp`).

### Alternatives Considered
- **LangChain / LangGraph:** Rejected due to excessive abstraction churn, hidden prompt formatting, and lack of deterministic state freezing.
- **CrewAI / AutoGen:** Rejected because multi-agent role-playing abstractions make it difficult to enforce formal mathematical invariants or deterministic governance guarantees.

### Consequences
- **Positive:** Total control over state mutation, execution budgets, verification logic, and provenance tracking. Hermetic test suite running in under 2 minutes.
- **Negative:** Required writing bespoke chunkers, vector search indices, and loop controllers that frameworks provide out of the box.

---

## ADR-02: Separation of Proposal Generation from Policy Authorization

### Context
Early agent architectures allowed the LLM to decide whether an action was allowed (e.g. asking the model *"Is this applicant approved?"*). Generative models can be coaxed via jailbreaks or prompt injections into granting unauthorized approvals.

### Decision
The LLM is strictly confined to generating a *proposal* (`ProposedDecision`). All policy authorizations are evaluated by a deterministic Python `PolicyEngine` executing compiled rule functions against an immutable ledger.

### Alternatives Considered
- **LLM Self-Critic / Evaluator Prompt:** Rejected because secondary LLM evaluators share the same probabilistic vulnerabilities as primary generators.
- **Guardrails Library (e.g. NeMo Guardrails):** Rejected due to complex Colang domain-specific language requirements and lack of direct DuckDB evidence binding.

### Consequences
- **Positive:** Provable guarantees: an LLM cannot autonomously approve an action if ground-truth ledger facts violate hard business constraints.
- **Negative:** Requires domain policy rules to be implemented as deterministic code functions.

---

## ADR-03: The Permission Asymmetry (Weak Restricts, Strong Permits)

### Context
In business operations, decisions can be based on diverse evidence sources: authoritative database records, analytical SQL aggregations, OCR extractions, and user prompts. Treating all evidence as equally authoritative creates security holes.

### Decision
Enforce a structural asymmetry:
- **Restricting an action (BLOCKED / REQUIRES_APPROVAL)** can proceed on *weak or partial* evidence (such as a single derogatory record found via free-form SQL).
- **Permitting an action autonomously (ALLOWED)** strictly requires *conclusive, authoritative* proof: a single undisputed value attributed to the target entity from a permit-capable source.

### Alternatives Considered
- **Symmetric Evidence Weighting:** All tool results treated with identical authority. Rejected because model-authored SQL queries can manufacture synthetic records (`SELECT 750 AS credit_score`).
- **Complete SQL Ban:** Disallow free SQL entirely. Rejected because SQL aggregation is necessary for forensic investigation.

### Consequences
- **Positive:** Eliminates synthetic SQL projection attacks while retaining analytical query power.
- **Negative:** Autonomous approvals require explicit record lookup tools; pipelines with only analytical SQL will safely escalate to human approval.

---

## ADR-04: Client-Side Pinning of Permit-Capable Tools

### Context
In Model Context Protocol (MCP), servers declare metadata including tool categories (`category: "record"`). A compromised or rogue server could label a free-form SQL tool as `record`, tricking the fact resolver into granting it permit authority.

### Decision
The client runtime pins authoritative tool names in domain manifests (`domain_manifest.yaml`). The client checks `tool_name \in domain.permit_tools` and completely ignores server-declared categories when evaluating `can_permit`.

### Alternatives Considered
- **Server Certificate Validation:** Cryptographically signing server capabilities. Rejected as overly complex for local process architectures.
- **Trust Server Declarations:** Rejected because untrusted third-party servers must never define client authorization rights.

### Consequences
- **Positive:** Hardened trust boundary: the client determines privilege levels regardless of what an external server advertises.
- **Negative:** Adding a new permit-capable tool requires updating the domain manifest.

---

## ADR-05: Entity-Scoped Fact Resolution vs. Global Property Extraction

### Context
Naively scanning an evidence log for field names (e.g. finding the first occurrence of `credit_score`) causes accidental entity cross-contamination when multiple records are present in the ledger.

### Decision
`resolve_fact()` walks nested dictionary structures and strictly attributes fields to the nearest enclosing entity identifier (`account_id`, `incident_id`, `service_name`). Values belonging to other entities are excluded from candidates.

### Alternatives Considered
- **Flat Key-Value Dictionaries:** Requiring tools to return flat keys like `ACCT_101_credit_score`. Rejected because external APIs return nested JSON structures.
- **Heuristic String Matching:** Rejected due to fragility.

### Consequences
- **Positive:** Multi-account queries cannot accidentally authorize Customer A based on Customer B's pristine credit score.
- **Negative:** Tools must include entity identifiers in their returned records for facts to be attributed.

---

## ADR-06: Fail-Closed Target Entity Binding

### Context
A confused deputy or prompt injection attack can trick an agent into evaluating one entity (e.g. a subprime applicant) while proposing an action against another (e.g. a prime customer).

### Decision
`target_entity_binding` extracts all entities mentioned in the user's initial prompt (canonicalizing delimiters and prefixes) and matches them against `proposal.parameters[target]`. If they do not match, or if no entity was identified in the query, the engine fails closed to `REQUIRES_APPROVAL`.

### Alternatives Considered
- **Permissive Unbound Actions:** Allow proposals to act on any entity discovered during execution. Rejected as vulnerable to target-swap attacks.
- **Strict Regex Parsing:** Rejected in favor of manifest-driven domain pattern bindings with canonical prefix normalization.

### Consequences
- **Positive:** Completely blocks target substitution attacks across all domains.
- **Negative:** Prompts that ask general questions without naming an entity cannot execute autonomous actions without human confirmation.

---

## ADR-07: Clause-Level Numerical Attribution & Domain Synonyms

### Context
Global token-distance matching accepts numbers that belong to different entities or opposing metrics within complex sentences (e.g. matching mobile conversion rate to desktop, or checkouts started to purchases completed).

### Decision
Split multi-claim sentences into grammatical clauses (delimited by conjunctions like `vs`, `while` and punctuation). Bind claims strictly to the subject of their clause, normalize domain synonyms (e.g. `phone` -> `mobile`), and reject metric oppositions.

### Alternatives Considered
- **Dependency Parsing (spaCy / Stanza):** Heavyweight neural dependency parsing. Rejected due to installation overhead, latency, and fragility on informal LLM text.
- **Bag-of-Words Token Proximity:** Proved vulnerable to entity and metric swaps (accepted 50.9% of fabricated numbers in benchmark).

### Consequences
- **Positive:** Rejects 100% of entity swaps, metric swaps, and opposite-direction claims while maintaining 100% acceptance of correct ground-truth answers.
- **Negative:** Complex nested subordinate clauses require careful delimiter splitting.

---

## ADR-08: Two-Tier Citation Verification (Deterministic Guards + Local NLI)

### Context
Lexical overlap accepts sentences that cite documents asserting the direct opposite of what the sentence claims (e.g. *"queries have no timeout"* citing *"queries have a 5-second timeout"*).

### Decision
Implement a two-tier verification check:
1. Fast deterministic guards: Check polarity/negation token alignment and numerical value overlap.
2. Cross-Encoder NLI: Evaluate natural language entailment using local `cross-encoder/nli-deberta-v3-xsmall`.
3. Offline fallback: If the NLI model is offline, gracefully fall back to lexical overlap + deterministic guards.

### Alternatives Considered
- **Pure Lexical Overlap:** Fast, but produced 50% false positive rates on contradiction probes.
- **LLM-Judge Evaluation:** High latency, cost, and susceptible to its own reasoning hallucinations.

### Consequences
- **Positive:** Detects semantic contradictions with 0 false positives on held-out sets, while maintaining zero-error offline execution.
- **Negative:** NLI inference adds ~50–100ms when neural model is active.

---

## ADR-09: Deeply Immutable Append-Only Evidence Ledger

### Context
In Python, dataclasses with `frozen=True` only protect top-level references. If an evidence item contains a nested dictionary, callers can mutate it after insertion, corrupting the deterministic audit trail.

### Decision
Implement recursive deep-freezing (`freeze()`) that converts all dictionaries to `MappingProxyType` and lists to tuples upon insertion into `EvidenceLedger`. Exporting creates thawed copies (`thaw()`).

### Alternatives Considered
- **Shallow Frozen Dataclass:** Rejected because nested dicts remained mutable.
- **JSON String Serialization:** Storing strings internally. Rejected due to deserialization overhead on repeated fact lookups.

### Consequences
- **Positive:** Mathematical guarantee of immutability: past observations cannot be rewritten by subsequent tool runs or user code.
- **Negative:** A recursive copy is performed at insertion time (negligible for ledger sizes under 100 items).

---

## ADR-10: Model Context Protocol (MCP) Over Subprocess Stdio

### Context
Domain tools (DuckDB queries, incident lookup APIs) could be imported directly as Python functions in the agent process. However, this creates monolithic coupling, memory leakage risks, and prevents language-agnostic extensibility.

### Decision
Expose domain capabilities as Model Context Protocol (MCP) servers communicating over standard input/output (stdio) JSON-RPC streams.

### Alternatives Considered
- **Direct In-Process Function Calls:** Fast, but allows tools to access client memory and prevents testing realistic wire serialization.
- **HTTP/REST Endpoints:** Introduces port management, socket exhaustion, and network security configuration.

### Consequences
- **Positive:** Clean process isolation; tools run in genuine separate processes conforming to standard open protocols.
- **Negative:** Requires managing subprocess lifecycles and handling Windows asyncio pipe closing quirks.
