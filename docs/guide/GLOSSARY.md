# Glossary of Terms and Acronyms

This glossary provides definitions for every acronym, technical term, and concept used across the repository.

---

| Term / Acronym | Full Form | Definition in This Project |
| :--- | :--- | :--- |
| **ADR** | Architectural Decision Record | A document capturing an important architectural decision made on a software project, along with its context and consequences. |
| **Bi-Encoder** | Dual-Encoder Neural Model | An embedding architecture that encodes queries and documents independently into dense vectors, enabling rapid vector similarity search via dot product. |
| **Cross-Encoder** | Joint Attention Neural Model | A model architecture that takes query and document text concatenated together, attending across both simultaneously to compute relevance or entailment. Highly accurate but computationally expensive. |
| **DDL** | Data Definition Language | SQL commands that define or modify database schemas (`CREATE`, `ALTER`, `DROP`). Explicitly blocked by `validate_read_only_sql()`. |
| **DTI** | Debt-to-Income Ratio | The percentage of a consumer's monthly gross income that goes toward paying debts. Expressed as a fraction (e.g. $0.43 = 43\%$). Hard ceiling enforced in credit risk policy. |
| **Evidence Ledger** | Append-Only Evidence Log | The immutable, deep-frozen data store collecting observations from tool calls and retrieval during an agent run. Serves as the single source of truth for deterministic verifiers and policy rules. |
| **FICO** | Fair Isaac Corporation | Standardized consumer credit score ranging from 300 to 850. |
| **JSON-RPC** | JSON Remote Procedure Call | A stateless, light-weight remote procedure call protocol using JSON data formatting. Used as the underlying wire protocol for Model Context Protocol (MCP). |
| **MCP** | Model Context Protocol | An open standard protocol created by Anthropic that standardizes how applications provide context (tools, prompts, resources) to LLMs over stdio or HTTP transports. |
| **MRR** | Mean Reciprocal Rank | A statistical measure evaluating search and retrieval systems, calculated as the mean of the reciprocal ranks of the first relevant document across all queries. |
| **NLI** | Natural Language Inference | The task of determining whether a hypothesis is supported (entailed), contradicted, or undetermined (neutral) by a given premise. Used in citation verification. |
| **OCR** | Optical Character Recognition | Technology converting images of typed, handwritten or printed text into machine-encoded text. |
| **Permission Asymmetry** | Architectural Governance Principle | The rule that weak or unverified evidence (such as free-form SQL queries) can restrict (block or escalate) an autonomous action, but only strong, authoritative evidence (`permit_tools`) can permit one. |
| **RAG** | Retrieval-Augmented Generation | An architecture that optimizes LLM generation by retrieving relevant documents from an external knowledge base before generating a response. |
| **Recall@k** | Recall at Rank k | The fraction of evaluation queries for which at least one relevant document was retrieved among the top $k$ positions. |
| **SLA** | Service Level Agreement | A contractual commitment between a service provider and a client defining expected service availability, performance, and uptime thresholds. |
| **SLO** | Service Level Objective | A target metric or performance level within an SLA (e.g. 99.9% uptime). |
| **SMT** | Satisfiability Modulo Theories | Mathematical logic solvers determining whether mathematical formulas or constraint sets are satisfiable. |
| **Target Binding** | Confused-Deputy Security Guard | A policy engine rule verifying that the entity targeted by an agent's proposal matches the entity requested in the user prompt, preventing target-swap attacks. |
| **TLS** | Transport Layer Security | Cryptographic protocol providing communications security over a computer network. Production services are audited for minimum TLS 1.3 baseline. |
| **VLM** | Vision-Language Model | A multimodal model capable of processing both visual images/documents and natural language text. |
| **Z-Test** | Two-Proportion Z-Test | A statistical hypothesis test used to determine whether the difference between two independent proportions is statistically significant. |
