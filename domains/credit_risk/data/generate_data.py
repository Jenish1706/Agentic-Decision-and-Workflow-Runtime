"""Builds data/credit_risk.duckdb for the credit risk domain.

Tables (schema in ../schema.sql, which this script executes - it is the
single source of truth; the previous version re-declared every table inline,
so the two could drift):
    accounts          score, limit, income, debt obligations and DTI per borrower
    credit_history    delinquencies, utilization and payment history
    limit_requests    pending credit limit increase requests
    bureau_inquiries  hard and soft credit inquiries

Each account is hand-authored to exercise one decision path of
domains/credit_risk/policies.py (thresholds from the domain manifest):
    ACCT-101  score 740, DTI 28%, no delinquencies, +$3,000     -> ALLOWED
    ACCT-102  score 610, one 90-day delinquency                 -> BLOCKED
    ACCT-103  score 710, DTI 32%, +$10,000 (over $5,000)        -> REQUIRES_APPROVAL
    ACCT-104  score 670, DTI 48% (over 43%)                     -> BLOCKED
    ACCT-105  no credit score on file, 2 months on books        -> REQUIRES_APPROVAL (fact missing)
    ACCT-106  SMB line, DTI 48%, +$7,000                        -> BLOCKED

dti_ratio is stored as a fraction and matches monthly_debt_obligations /
(annual_income / 12) to three decimals for every account. (An old comment
described ACCT-106's records as contradictory; they are not -
5,600 / (140,000 / 12) = 0.48.)

Run: uv run python domains/credit_risk/data/generate_data.py
"""

from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parent
DB_PATH = DATA_DIR / "credit_risk.duckdb"
SCHEMA_PATH = DATA_DIR.parent / "schema.sql"

# (account_id, customer_name, account_type, credit_score, current_limit, annual_income,
#  monthly_debt_obligations, dti_ratio, employment_status, account_status)
ACCOUNTS = [
    ("ACCT-101", "Alice Walker", "consumer_revolving", 740, 5000.0, 95000.0, 2216.0, 0.280, "employed", "active"),
    ("ACCT-102", "Bob Martinez", "consumer_revolving", 610, 3000.0, 48000.0, 1400.0, 0.350, "employed", "active"),
    ("ACCT-103", "Charlie Davis", "consumer_revolving", 710, 10000.0, 120000.0, 3200.0, 0.320, "employed", "active"),
    ("ACCT-104", "Diana Prince", "consumer_revolving", 670, 4000.0, 60000.0, 2400.0, 0.480, "employed", "active"),
    ("ACCT-105", "Evan Stone", "consumer_revolving", None, 1000.0, 32000.0, 500.0, 0.187, "employed", "active"),
    ("ACCT-106", "Fiona Gallagher", "smb_credit_line", 680, 8000.0, 140000.0, 5600.0, 0.480, "self_employed", "active"),
]

# (account_id, delinquencies_30d, delinquencies_60d, delinquencies_90d, revolving_utilization,
#  on_time_payment_pct, months_on_books) - on_time_payment_pct is a fraction despite its name.
CREDIT_HISTORY = [
    ("ACCT-101", 0, 0, 0, 0.22, 1.00, 36),
    ("ACCT-102", 2, 1, 1, 0.88, 0.85, 24),
    ("ACCT-103", 0, 0, 0, 0.30, 0.99, 48),
    ("ACCT-104", 1, 0, 0, 0.65, 0.94, 18),
    ("ACCT-105", 0, 0, 0, 0.10, 1.00, 2),
    ("ACCT-106", 0, 0, 0, 0.42, 0.98, 14),
]

# (request_id, account_id, requested_limit, increase_amount, request_date, request_reason)
LIMIT_REQUESTS = [
    ("REQ-001", "ACCT-101", 8000.0, 3000.0, "2024-11-10", "Home renovation expenses"),
    ("REQ-002", "ACCT-102", 5000.0, 2000.0, "2024-11-12", "Debt consolidation"),
    ("REQ-003", "ACCT-103", 20000.0, 10000.0, "2024-11-14", "Business travel buffer"),
    ("REQ-004", "ACCT-104", 6000.0, 2000.0, "2024-11-15", "Emergency auto repair"),
    ("REQ-005", "ACCT-105", 3000.0, 2000.0, "2024-11-16", "Credit building"),
    ("REQ-006", "ACCT-106", 15000.0, 7000.0, "2024-11-17", "Working capital expansion"),
]

# (inquiry_id, account_id, inquiry_date, inquiry_type, inquiry_purpose)
BUREAU_INQUIRIES = [
    ("INQ-001", "ACCT-101", "2024-05-12", "soft_pull", "Account review"),
    ("INQ-002", "ACCT-102", "2024-10-01", "hard_pull", "Auto loan application"),
    ("INQ-003", "ACCT-103", "2024-08-20", "soft_pull", "Promotional review"),
    ("INQ-004", "ACCT-104", "2024-09-15", "hard_pull", "Personal loan application"),
    ("INQ-005", "ACCT-105", "2024-11-01", "hard_pull", "Credit card account opening"),
    ("INQ-006", "ACCT-106", "2024-07-11", "hard_pull", "Equipment lease inquiry"),
]


def generate_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)  # rebuild from scratch; the data is fully defined above

    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        connection.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ACCOUNTS)
        connection.executemany("INSERT INTO credit_history VALUES (?, ?, ?, ?, ?, ?, ?)", CREDIT_HISTORY)
        connection.executemany("INSERT INTO limit_requests VALUES (?, ?, ?, ?, ?, ?)", LIMIT_REQUESTS)
        connection.executemany("INSERT INTO bureau_inquiries VALUES (?, ?, ?, ?, ?)", BUREAU_INQUIRIES)
    finally:
        connection.close()
    print(f"Generated {db_path}")


if __name__ == "__main__":
    generate_database()
