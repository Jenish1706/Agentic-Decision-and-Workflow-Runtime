# Data Sources, Schemas & Quality Governance

This document describes all corpora, databases, and structured fixtures used in the runtime, including schemas, exact row counts, provenance, license terms, quality issues, and exclusion criteria.

---

## 1. Data Sources & Licenses

All data in this repository is synthetic and specifically constructed for deterministic testing, evaluation, and red-teaming. No proprietary, customer, PII, or real financial data is stored.

- **Corpus & Fixtures License:** MIT License (matching repository license).
- **Domain Generation Scripts:** Reproducible data generation scripts are included in `domains/credit_risk/data/generate_data.py` and `domains/product_analytics/data/generate_data.py`.

---

## 2. Tabular Databases (DuckDB)

### 2.1 Credit Risk Underwriting (`domains/credit_risk/data/credit_risk.duckdb`)

Enforces underwriting credit policies across 6 synthetic consumer accounts.

#### Table: `accounts` (6 rows)
| Column | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `account_id` | VARCHAR | Unique customer identifier (e.g. `ACCT-101`) | Primary Key |
| `name` | VARCHAR | Customer full name | Not Null |
| `credit_score` | INTEGER | FICO-equivalent credit score (300–850) | Nullable (thin-file accounts) |
| `current_limit` | DECIMAL(10,2) | Active credit line in USD | $> 0$ |
| `account_status` | VARCHAR | Status (`active`, `frozen`, `closed`) | Not Null |

#### Table: `credit_history` (6 rows)
| Column | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `account_id` | VARCHAR | Foreign key referencing `accounts` | Not Null |
| `delinquencies_90d` | INTEGER | Count of 90+ day delinquencies | $\ge 0$ |
| `dti_ratio` | DECIMAL(5,4) | Debt-to-income ratio ($0.0000$ to $1.0000$) | $\ge 0$ |
| `bankruptcy_flag` | BOOLEAN | Historical bankruptcy indicator | Not Null |

#### Table: `limit_requests` (6 rows)
| Column | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `request_id` | VARCHAR | Unique request tracking code | Primary Key |
| `account_id` | VARCHAR | Foreign key referencing `accounts` | Not Null |
| `requested_amount`| DECIMAL(10,2) | Requested dollar increase amount | $> 0$ |
| `request_date` | DATE | Date request was submitted | Not Null |

#### Table: `bureau_inquiries` (6 rows)
| Column | Type | Description | Constraints |
| :--- | :--- | :--- | :--- |
| `inquiry_id` | VARCHAR | Unique bureau pull reference | Primary Key |
| `account_id` | VARCHAR | Foreign key referencing `accounts` | Not Null |
| `inquiry_count` | INTEGER | Hard pulls in preceding 12 months | $\ge 0$ |

---

### 2.2 Product Analytics (`domains/product_analytics/data/product_analytics.duckdb`)

Captures e-commerce checkout funnel events and A/B test telemetry.

#### Table: `experiments` (1 row)
| Column | Type | Description |
| :--- | :--- | :--- |
| `experiment_id` | VARCHAR | Identifier (`EXP-204`) |
| `name` | VARCHAR | Human-readable test name (`one_click_checkout`) |
| `status` | VARCHAR | Lifecycle status (`active`, `concluded`) |

#### Table: `experiment_results` (2 rows: Control vs Treatment)
| Column | Type | Description |
| :--- | :--- | :--- |
| `experiment_id` | VARCHAR | Foreign key referencing `experiments` |
| `variant` | VARCHAR | Variant identifier (`variant_a` control, `variant_b` treatment) |
| `visitors` | INTEGER | Sample size $n$ |
| `conversions` | INTEGER | Successful checkout conversions $x$ |
| `conversion_rate` | DECIMAL(5,4) | Empirical conversion proportion $\hat{p} = x/n$ |

#### Table: `funnel_dropoff` (3 rows)
Tracks conversion stages: `cart_view`, `checkout_start`, `payment_submit`.

#### Table: `daily_checkout_metrics` (12 rows)
Daily device-segmented checkout metrics (desktop vs. mobile) spanning 6 days.

---

## 3. Structured JSON Fixtures

### 3.1 Enterprise Systems Operations (`domains/enterprise_operations/data/`)
- `incidents.json` (4 records): Telemetry for production incidents across services (`auth-service`, `checkout-api`, `recommendation-worker`, `legacy-billing`), tracking severity (`P0`, `P1`, `P2`), blast radius percentage, and commander override authorization flags.
- `service_configs.json` (4 records): Security baselines including configured TLS version (`1.2` vs `1.3`) and debug port states (`debug_port_open: true/false`).

### 3.2 Multimodal Business Operations (`domains/multimodal_investigation/data/`)
- `invoices/` (4 JSON fixtures: `inv-2026-001` through `inv-2026-004`): Pre-extracted structured representations of invoice PDFs, detailing line items, subtotal, tax amount, billed total, duplicate flags, and settlement statuses.
- `charts/` (2 JSON fixtures: `latency_sla_september`, `conversion_funnel_redesign`): Telemetry chart extraction data detailing monthly uptime averages, SLA contract commitments, and funnel step drops.

---

## 4. Engineering Knowledge Base (`data/documents/`)

31 Markdown documents (averaging 600–1,000 words each) categorized across:
- **API References (5 docs):** `api-anchor`, `api-aurora`, `api-compass`, `api-ferry`, `api-lighthouse`
- **Architecture Specs (7 docs):** System topologies, rate-limiting algorithms, and queue buffers
- **Operational Runbooks (5 docs):** DB failover, deployment rollbacks, on-call escalations, storage cleanup
- **Postmortems (4 docs):** Real-world incident retro analyses with timestamps, root causes, and action items
- **Company Policies (4 docs):** Access control, data retention, deployment authorization, on-call schedules
- **Internal FAQs (6 docs):** Token expiration, SLO definitions, alert noise reduction

---

## 5. Known Quality Issues & Handling

1. **Thin-File Entities (`credit_score: null`):**
   - *Issue:* Account `ACCT-105` has no bureau credit score on file.
   - *Handling:* `resolve_fact` returns `status=MISSING`. Policy rule `minimum_credit_score` treats missing score as non-conclusive and escalates to `REQUIRES_APPROVAL` (fails closed).
2. **String Number Formatting in Tool Outputs:**
   - *Issue:* LLM parameters and JSON files format currency and percentages with decorations (e.g. `"$5,000"`, `"48%"`).
   - *Handling:* `as_number()` strictly parses these by stripping commas and symbols while asserting float finiteness, explicitly rejecting booleans (`True` is not converted to `1.0`).
3. **Boolean Parsing Ambiguities:**
   - *Issue:* Models output string representations `"false"` or `"0"` for flags like `override_authorized`.
   - *Handling:* Python's `bool("false")` evaluates to `True`. `as_bool()` explicitly matches token whitelists (`{"true", "yes", "1"}` vs `{"false", "no", "0"}`), returning `None` for ambiguous strings.
4. **Synthetic Projection Queries in Free SQL:**
   - *Issue:* DuckDB allows `SELECT 750 AS credit_score` without querying tables.
   - *Handling:* Governed by the Permission Asymmetry: free SQL evidence receives `can_permit=False`, preventing autonomous authorization.

---

## 6. Exclusions & Scoping

- **No Public Internet / Live Web Scraping:** All documentation and data is stored locally to maintain strict determinism and eliminate external network failure in CI.
- **No Direct Image / Binary Storage:** Multimodal document extractions are stored as pre-extracted JSON fixtures simulating OCR/VLM output, focusing evaluation on agentic reasoning and deterministic policy enforcement rather than vision model fine-tuning.
