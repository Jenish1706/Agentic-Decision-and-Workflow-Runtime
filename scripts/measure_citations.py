"""Measures precision and recall of citation support verification on a benchmark
of 60 supported vs unsupported claim-chunk pairs.

Run: uv run python scripts/measure_citations.py
"""

import json
from pathlib import Path

from agent_runtime.agent.verification import chunk_supports_sentence, _get_nli_model
from agent_runtime.rag.documents import Chunk

BENCHMARK_DATA = [
    # ---------------- 30 SUPPORTED PAIRS ----------------
    (
        "Aurora is Nimbus's event ingestion service and entry point for telemetry data.",
        "Aurora is Nimbus's event ingestion service. All telemetry and event streams enter through Aurora.",
        True,
    ),
    (
        "Aurora enforces a default rate limit of 5,000 events/s per tenant.",
        "Aurora enforces a default rate limit of 5,000 events/s. Tenants exceeding this limit receive HTTP 429.",
        True,
    ),
    (
        "Tenants receive HTTP 429 responses when exceeding ingestion thresholds.",
        "When an account exceeds the configured throughput, Aurora responds with HTTP 429 Too Many Requests.",
        True,
    ),
    (
        "The buffer subsystem utilizes Redis streams for temporary persistence.",
        "Event buffering is implemented using Redis streams before writing to DuckDB.",
        True,
    ),
    (
        "Boreas is the analytics query engine running DuckDB analytical queries.",
        "Boreas handles OLAP analytics queries executing against DuckDB storage.",
        True,
    ),
    (
        "Zephyr is the machine learning model inference microservice.",
        "Zephyr coordinates model serving and real-time LLM inference requests.",
        True,
    ),
    (
        "The maximum credit limit increase allowed autonomously is $3,000.",
        "Autonomous credit limit increases are capped at $3,000. Higher amounts require senior underwriter review.",
        True,
    ),
    (
        "Borrowers must maintain a minimum credit score of 660 for automated approval.",
        "Policy rule minimum_credit_score requires a verified FICO score of at least 660.",
        True,
    ),
    (
        "The debt-to-income ratio ceiling is strictly 43.0%.",
        "dti_ceiling rule rejects or escalates any loan applicant with debt-to-income ratio exceeding 43%.",
        True,
    ),
    (
        "Accounts with 90-day delinquencies are blocked from automated limit increases.",
        "hard_delinquency_cutoff blocks any borrower who has recorded a 90-day delinquency in the past 24 months.",
        True,
    ),
    (
        "Soft pull inquiries do not impact underwriting risk scores.",
        "Bureau inquiry records indicate that soft pull inquiries do not affect applicant risk tier.",
        True,
    ),
    (
        "The SLO target for API latency is 99.9% of requests under 200ms.",
        "Service Level Objective (SLO): 99.9% of API requests must complete in under 200ms.",
        True,
    ),
    (
        "Incident severity P0 requires an immediate incident commander assignment.",
        "For P0 outages, on-call paging triggers immediate incident commander mobilization.",
        True,
    ),
    (
        "Database replicas synchronize asynchronously with a maximum replication lag of 5 seconds.",
        "Read replicas sync asynchronously from the primary node; alerting triggers if replication lag exceeds 5s.",
        True,
    ),
    (
        "The vector search index uses cosine distance for dense embedding similarity.",
        "Vector search uses cosine similarity over 384-dimensional dense sentence embeddings.",
        True,
    ),
    (
        "Authentication requires JWT bearer tokens signed with RS256.",
        "API gateway validates incoming requests using RS256-signed JWT bearer tokens.",
        True,
    ),
    (
        "Cache eviction follows a Least Recently Used (LRU) policy with a 1GB limit.",
        "In-memory caching implements LRU eviction when memory consumption reaches 1GB.",
        True,
    ),
    (
        "DuckDB query timeout is configured to 5 seconds to prevent runaway queries.",
        "Analytical queries are constrained by a 5-second execution timeout.",
        True,
    ),
    (
        "Audit logging records all administrative operations to an immutable ledger.",
        "All admin actions and configuration changes are appended to the immutable audit ledger.",
        True,
    ),
    (
        "The webhook dispatcher retries failed deliveries using exponential backoff.",
        "Webhook notification failures trigger exponential backoff retries up to 5 attempts.",
        True,
    ),
    (
        "Customer accounts are flagged inactive after 12 months without activity.",
        "An account is marked inactive if no logins or transactions occur within 12 consecutive months.",
        True,
    ),
    (
        "Data exports are encrypted using AES-256 before upload to cloud storage.",
        "Exported data archives are encrypted at rest using AES-256 keys.",
        True,
    ),
    (
        "Circuit breakers open after 5 consecutive downstream service failures.",
        "Downstream client circuit breaker transitions to OPEN after 5 consecutive request failures.",
        True,
    ),
    (
        "Rate limiting uses a token bucket algorithm with burst capacity of 100 requests.",
        "Token bucket rate limiting allows burst traffic up to 100 tokens per client IP.",
        True,
    ),
    (
        "The ETL pipeline runs nightly at 02:00 UTC to aggregate metrics.",
        "Daily batch ETL executes at 02:00 UTC to aggregate platform usage tables.",
        True,
    ),
    (
        "Prometheus scrapes metrics from the /metrics endpoint every 15 seconds.",
        "Metric collection is configured with Prometheus scraping /metrics on a 15s interval.",
        True,
    ),
    (
        "Underwriters can override automated policy declines with written justification.",
        "Human underwriters have override authority for declined applications upon documenting justification.",
        True,
    ),
    (
        "Kafka partition keys are derived from customer_id to preserve event order.",
        "Events are partitioned by customer_id to guarantee in-order delivery across Kafka topics.",
        True,
    ),
    (
        "Session cookies expire after 8 hours of inactivity.",
        "Web application session cookies are invalidated after 8 hours of user inactivity.",
        True,
    ),
    (
        "Microservices communicate using gRPC over HTTP/2 for internal RPCs.",
        "Internal service communication utilizes gRPC with protobuf definitions over HTTP/2.",
        True,
    ),

    # ---------------- 30 UNSUPPORTED PAIRS ----------------
    (
        "The Eiffel Tower was completed in 1889 in Paris, France.",
        "Aurora is Nimbus's event ingestion service. All telemetry enters through Aurora.",
        False,
    ),
    (
        "Napoleon Bonaparte was defeated at the Battle of Waterloo in 1815.",
        "Boreas handles OLAP analytics queries executing against DuckDB storage.",
        False,
    ),
    (
        "Photosynthesis in plants converts solar energy into chemical sugars.",
        "dti_ceiling rule rejects or escalates any loan applicant with debt-to-income ratio exceeding 43%.",
        False,
    ),
    (
        "Quantum computers leverage superposition and entanglement for computation.",
        "hard_delinquency_cutoff blocks any borrower who has recorded a 90-day delinquency in the past 24 months.",
        False,
    ),
    (
        "The Pacific Ocean is the largest and deepest oceanic division on Earth.",
        "Policy rule minimum_credit_score requires a verified FICO score of at least 660.",
        False,
    ),
    (
        "Mount Everest is the highest mountain peak above sea level in the Himalayas.",
        "When an account exceeds the configured throughput, Aurora responds with HTTP 429 Too Many Requests.",
        False,
    ),
    (
        "The speed of light in vacuum is approximately 299,792 kilometers per second.",
        "Vector search uses cosine similarity over 384-dimensional dense sentence embeddings.",
        False,
    ),
    (
        "DNA consists of two polynucleotide chains that coil around each other.",
        "Downstream client circuit breaker transitions to OPEN after 5 consecutive request failures.",
        False,
    ),
    (
        "William Shakespeare wrote Hamlet and Romeo and Juliet in Elizabethan England.",
        "Token bucket rate limiting allows burst traffic up to 100 tokens per client IP.",
        False,
    ),
    (
        "The Great Barrier Reef is the world's largest coral reef system in Australia.",
        "All admin actions and configuration changes are appended to the immutable audit ledger.",
        False,
    ),
    (
        "Aurora stores customer credit card details in unencrypted plaintext files.",
        "Aurora is Nimbus's event ingestion service. All telemetry enters through Aurora.",
        False,
    ),
    (
        "Boreas executes machine learning model inference using GPU tensor cores.",
        "Boreas handles OLAP analytics queries executing against DuckDB storage.",
        False,
    ),
    (
        "Applicants with credit scores below 500 receive instant automatic approval.",
        "Policy rule minimum_credit_score requires a verified FICO score of at least 660.",
        False,
    ),
    (
        "All database queries are executed directly on the production primary without replicas.",
        "Read replicas sync asynchronously from the primary node; alerting triggers if replication lag exceeds 5s.",
        False,
    ),
    (
        "Users can log in without passwords or tokens using guest anonymous sessions.",
        "API gateway validates incoming requests using RS256-signed JWT bearer tokens.",
        False,
    ),
    (
        "The maximum allowable debt-to-income ratio is 95% for prime borrowers.",
        "dti_ceiling rule rejects or escalates any loan applicant with debt-to-income ratio exceeding 43%.",
        False,
    ),
    (
        "Aurora rate limits allow unlimited throughput of 100 million transactions per minute.",
        "Aurora enforces a default rate limit of 5,000 events/s. Tenants exceeding this limit receive HTTP 429.",
        False,
    ),
    (
        "Underwriters are prohibited from viewing customer credit history dossiers.",
        "Human underwriters have override authority for declined applications upon documenting justification.",
        False,
    ),
    (
        "Analytical queries have no timeout and run indefinitely until system reboot.",
        "Analytical queries are constrained by a 5-second execution timeout.",
        False,
    ),
    (
        "Redis streams are cleared immediately without buffering incoming event payloads.",
        "Event buffering is implemented using Redis streams before writing to DuckDB.",
        False,
    ),
    (
        "The vector search engine relies on lexical BM25 matching rather than dense vectors.",
        "Vector search uses cosine similarity over 384-dimensional dense sentence embeddings.",
        False,
    ),
    (
        "Customer accounts are deleted permanently after 48 hours of inactivity.",
        "An account is marked inactive if no logins or transactions occur within 12 consecutive months.",
        False,
    ),
    (
        "Internal services communicate exclusively over unencrypted telnet sockets.",
        "Internal service communication utilizes gRPC with protobuf definitions over HTTP/2.",
        False,
    ),
    (
        "Kafka partitions events randomly across worker nodes regardless of user ID.",
        "Events are partitioned by customer_id to guarantee in-order delivery across Kafka topics.",
        False,
    ),
    (
        "Prometheus metrics are pushed once per year during annual maintenance windows.",
        "Metric collection is configured with Prometheus scraping /metrics on a 15s interval.",
        False,
    ),
    (
        "ETL aggregation batches are triggered manually by customer email requests.",
        "Daily batch ETL executes at 02:00 UTC to aggregate platform usage tables.",
        False,
    ),
    (
        "Session authentication cookies never expire and remain valid forever.",
        "Web application session cookies are invalidated after 8 hours of user inactivity.",
        False,
    ),
    (
        "Data exports are transmitted in cleartext HTTP without encryption.",
        "Exported data archives are encrypted at rest using AES-256 keys.",
        False,
    ),
    (
        "P0 outages are logged in a spreadsheet and reviewed at quarterly town halls.",
        "For P0 outages, on-call paging triggers immediate incident commander mobilization.",
        False,
    ),
    (
        "Webhook deliveries are discarded immediately on failure without retrying.",
        "Webhook notification failures trigger exponential backoff retries up to 5 attempts.",
        False,
    ),
]

HELDOUT_PAIRS = [
    # ---------------- 10 SUPPORTED HELDOUT PAIRS ----------------
    (
        "Aurora persists incoming ingestion records into DuckDB partition tables.",
        "Ingested telemetry records are written to partitioned DuckDB tables by the Aurora pipeline.",
        True,
    ),
    (
        "Credit approvals exceeding $5,000 require vice president sign-off.",
        "Policy escalation threshold: limit requests above $5,000 require approval from VP of Underwriting.",
        True,
    ),
    (
        "The minimum debt service coverage ratio for commercial applicants is 1.25.",
        "Commercial credit risk underwriting requires a debt service coverage ratio (DSCR) of at least 1.25.",
        True,
    ),
    (
        "Service mesh mutual TLS uses certificates rotated every 30 days.",
        "Internal mTLS encryption enforces a 30-day certificate rotation lifecycle.",
        True,
    ),
    (
        "Boreas SQL query planner optimizes filter pushdowns to parquet files.",
        "The query planner pushes predicate filters directly down into Parquet scan operators.",
        True,
    ),
    (
        "The credit bureau API call times out after 3,000 milliseconds.",
        "External credit bureau requests are terminated if response time exceeds 3000ms.",
        True,
    ),
    (
        "Database failover promotes the secondary replica to leader within 30 seconds.",
        "Automated failover protocol promotes the synchronous standby replica to primary within 30s.",
        True,
    ),
    (
        "Invoices over $10,000 require two managerial approvals.",
        "Accounts payable policy mandates two levels of management approval for invoices greater than $10,000.",
        True,
    ),
    (
        "Redis cache TTL for user permissions is set to 300 seconds.",
        "Authorization permission cache entries in Redis expire after a 300s TTL.",
        True,
    ),
    (
        "The checkout service circuit breaker trips when error rate reaches 20%.",
        "Circuit breaker automatically trips to open state if downstream service error rate reaches 20%.",
        True,
    ),

    # ---------------- 10 UNSUPPORTED HELDOUT PAIRS ----------------
    (
        "Aurora persists incoming telemetry directly into MongoDB document collections.",
        "Ingested telemetry records are written to partitioned DuckDB tables by the Aurora pipeline.",
        False,
    ),
    (
        "Credit limit increases of $5,000 are approved automatically without escalation.",
        "Policy escalation threshold: limit requests above $5,000 require approval from VP of Underwriting.",
        False,
    ),
    (
        "Commercial applicants are approved with a debt service coverage ratio of 0.85.",
        "Commercial credit risk underwriting requires a debt service coverage ratio (DSCR) of at least 1.25.",
        False,
    ),
    (
        "Service mesh mutual TLS is disabled and communications use unencrypted HTTP.",
        "Internal mTLS encryption enforces a 30-day certificate rotation lifecycle.",
        False,
    ),
    (
        "Boreas disables filter pushdown and always performs full table scans.",
        "The query planner pushes predicate filters directly down into Parquet scan operators.",
        False,
    ),
    (
        "Credit bureau inquiries wait indefinitely with no timeout configured.",
        "External credit bureau requests are terminated if response time exceeds 3000ms.",
        False,
    ),
    (
        "Database failover requires manual intervention and replicas are never promoted automatically.",
        "Automated failover protocol promotes the synchronous standby replica to primary within 30s.",
        False,
    ),
    (
        "Invoices over $10,000 are paid immediately without managerial review.",
        "Accounts payable policy mandates two levels of management approval for invoices greater than $10,000.",
        False,
    ),
    (
        "Redis permissions cache has no TTL and never expires.",
        "Authorization permission cache entries in Redis expire after a 300s TTL.",
        False,
    ),
    (
        "The checkout circuit breaker allows 100% traffic during total downstream failure.",
        "Circuit breaker automatically trips to open state if downstream service error rate reaches 20%.",
        False,
    ),
]


def _evaluate_pairs(dataset: list[tuple[str, str, bool]], *, use_nli: bool = True) -> dict:
    tp = fp = tn = fn = 0
    failures = []

    for idx, (claim, chunk_text, expected_support) in enumerate(dataset):
        chunk = Chunk(
            chunk_id=f"doc::{idx}",
            doc_id="doc",
            doc_title="Doc",
            position=0,
            text=chunk_text,
        )
        supported, explanation = chunk_supports_sentence(claim, chunk, use_nli=use_nli)

        if expected_support and supported:
            tp += 1
        elif not expected_support and not supported:
            tn += 1
        elif not expected_support and supported:
            fp += 1
            failures.append(f"FALSE POSITIVE: Claim: '{claim}' | Chunk: '{chunk_text}' | Expl: {explanation}")
        elif expected_support and not supported:
            fn += 1
            failures.append(f"FALSE NEGATIVE: Claim: '{claim}' | Chunk: '{chunk_text}' | Expl: {explanation}")

    total = len(dataset)
    supported_total = sum(1 for _, _, exp in dataset if exp)
    unsupported_total = total - supported_total
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total": total,
        "supported_total": supported_total,
        "unsupported_total": unsupported_total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "failures": failures,
    }


def is_nli_available() -> bool:
    try:
        return _get_nli_model() is not None
    except Exception:
        return False


def load_committed_citation_measurements() -> dict | None:
    artifact_path = Path(__file__).resolve().parent.parent / "data" / "citation_measurements.json"
    if artifact_path.exists():
        try:
            return json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def measure(*, save_artifact: bool = True) -> dict:
    nli_active = is_nli_available()
    b_lex = _evaluate_pairs(BENCHMARK_DATA, use_nli=False)
    b_nli = _evaluate_pairs(BENCHMARK_DATA, use_nli=True)
    h_lex = _evaluate_pairs(HELDOUT_PAIRS, use_nli=False)
    h_nli = _evaluate_pairs(HELDOUT_PAIRS, use_nli=True)

    result = {
        **b_nli,  # Default top-level for backward compatibility
        "backend_used": "nli" if nli_active else "lexical_fallback",
        "benchmark_lexical": b_lex,
        "benchmark_nli": b_nli,
        "heldout_lexical": h_lex,
        "heldout_nli": h_nli,
    }

    if save_artifact and nli_active:
        artifact_path = Path(__file__).resolve().parent.parent / "data" / "citation_measurements.json"
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        except Exception:
            pass

    return result


def main() -> None:
    results = measure()
    b_lex = results["benchmark_lexical"]
    b_nli = results["benchmark_nli"]
    h_lex = results["heldout_lexical"]
    h_nli = results["heldout_nli"]
    backend = results.get("backend_used", "nli")
    nli_label = "NLI Mode" if backend == "nli" else "NLI Mode (UNAVAILABLE - fell back to lexical)"

    print("================ CITATION SUPPORT MEASUREMENT ================")
    print(f"Backend used for NLI evaluation: {backend}")
    print(f"Benchmark Set (60 pairs):")
    print(f"  Lexical Mode:  Recall {b_lex['recall']:.1%} ({b_lex['tp']}/{b_lex['supported_total']}), "
          f"Precision {b_lex['precision']:.1%} ({b_lex['tp']}/{b_lex['tp'] + b_lex['fp']}), "
          f"False Positives {b_lex['fp']}/{b_lex['unsupported_total']}")
    print(f"  {nli_label}: Recall {b_nli['recall']:.1%} ({b_nli['tp']}/{b_nli['supported_total']}), "
          f"Precision {b_nli['precision']:.1%} ({b_nli['tp']}/{b_nli['tp'] + b_nli['fp']}), "
          f"False Positives {b_nli['fp']}/{b_nli['unsupported_total']}")
    print(f"\nHeld-Out Set (20 pairs):")
    print(f"  Lexical Mode:  Recall {h_lex['recall']:.1%} ({h_lex['tp']}/{h_lex['supported_total']}), "
          f"Precision {h_lex['precision']:.1%} ({h_lex['tp']}/{h_lex['tp'] + h_lex['fp']})")
    print(f"  {nli_label}: Recall {h_nli['recall']:.1%} ({h_nli['tp']}/{h_nli['supported_total']}), "
          f"Precision {h_nli['precision']:.1%} ({h_nli['tp']}/{h_nli['tp'] + h_nli['fp']})")
    print("==============================================================")


if __name__ == "__main__":
    main()
