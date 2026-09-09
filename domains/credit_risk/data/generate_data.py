"""Data generator for Credit Risk & Underwriting domain.

Generates realistic credit underwriting database:
- `accounts`: Financial, income, score, and DTI profiles.
- `credit_history`: Delinquency records, revolving utilization, payment history.
- `limit_requests`: Pending credit line expansion requests.
- `bureau_inquiries`: Hard and soft credit inquiries.
"""

from pathlib import Path
import duckdb

DATA_DIR = Path(__file__).parent
DB_PATH = DATA_DIR / "credit_risk.duckdb"


def generate_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = duckdb.connect(str(db_path))

    # 1. Accounts table
    conn.execute("""
        CREATE TABLE accounts (
            account_id VARCHAR PRIMARY KEY,
            customer_name VARCHAR,
            account_type VARCHAR,
            credit_score INTEGER,
            current_limit DOUBLE,
            annual_income DOUBLE,
            monthly_debt_obligations DOUBLE,
            dti_ratio DOUBLE,
            employment_status VARCHAR,
            account_status VARCHAR
        )
    """)

    # 2. Credit history table
    conn.execute("""
        CREATE TABLE credit_history (
            account_id VARCHAR PRIMARY KEY,
            delinquencies_30d INTEGER,
            delinquencies_60d INTEGER,
            delinquencies_90d INTEGER,
            revolving_utilization DOUBLE,
            on_time_payment_pct DOUBLE,
            months_on_books INTEGER
        )
    """)

    # 3. Limit requests table
    conn.execute("""
        CREATE TABLE limit_requests (
            request_id VARCHAR PRIMARY KEY,
            account_id VARCHAR,
            requested_limit DOUBLE,
            increase_amount DOUBLE,
            request_date DATE,
            request_reason VARCHAR
        )
    """)

    # 4. Bureau inquiries table
    conn.execute("""
        CREATE TABLE bureau_inquiries (
            inquiry_id VARCHAR PRIMARY KEY,
            account_id VARCHAR,
            inquiry_date DATE,
            inquiry_type VARCHAR,
            inquiry_purpose VARCHAR
        )
    """)

    # Seed concrete borrower profiles representing each business decision path:
    accounts_data = [
        # ACCT-101: Clean prime borrower, low DTI, $3k increase -> ALLOWED
        ('ACCT-101', 'Alice Walker', 'consumer_revolving', 740, 5000.0, 95000.0, 2216.0, 0.280, 'employed', 'active'),
        # ACCT-102: Subprime borrower, 610 score, 90d delinquency -> BLOCKED
        ('ACCT-102', 'Bob Martinez', 'consumer_revolving', 610, 3000.0, 48000.0, 1400.0, 0.350, 'employed', 'active'),
        # ACCT-103: Prime borrower, high requested increase ($10k > $5k cap) -> REQUIRES_APPROVAL
        ('ACCT-103', 'Charlie Davis', 'consumer_revolving', 710, 10000.0, 120000.0, 3200.0, 0.320, 'employed', 'active'),
        # ACCT-104: High DTI borrower (48% > 43% cap) -> BLOCKED
        ('ACCT-104', 'Diana Prince', 'consumer_revolving', 670, 4000.0, 60000.0, 2400.0, 0.480, 'employed', 'active'),
        # ACCT-105: Thin-file borrower (unrated score, null / 0) -> INSUFFICIENT_EVIDENCE
        ('ACCT-105', 'Evan Stone', 'consumer_revolving', None, 1000.0, 32000.0, 500.0, 0.187, 'employed', 'active'),
        # ACCT-106: Contradictory records (annual income listed $140k but monthly debt calculation based on $30k) -> AUDIT_FLAG
        ('ACCT-106', 'Fiona Gallagher', 'smb_credit_line', 680, 8000.0, 140000.0, 5600.0, 0.480, 'self_employed', 'active'),
    ]
    for row in accounts_data:
        conn.execute("INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)

    history_data = [
        ('ACCT-101', 0, 0, 0, 0.22, 1.00, 36),
        ('ACCT-102', 2, 1, 1, 0.88, 0.85, 24),
        ('ACCT-103', 0, 0, 0, 0.30, 0.99, 48),
        ('ACCT-104', 1, 0, 0, 0.65, 0.94, 18),
        ('ACCT-105', 0, 0, 0, 0.10, 1.00, 2),
        ('ACCT-106', 0, 0, 0, 0.42, 0.98, 14),
    ]
    for row in history_data:
        conn.execute("INSERT INTO credit_history VALUES (?, ?, ?, ?, ?, ?, ?)", row)

    requests_data = [
        ('REQ-001', 'ACCT-101', 8000.0, 3000.0, '2024-11-10', 'Home renovation expenses'),
        ('REQ-002', 'ACCT-102', 5000.0, 2000.0, '2024-11-12', 'Debt consolidation'),
        ('REQ-003', 'ACCT-103', 20000.0, 10000.0, '2024-11-14', 'Business travel buffer'),
        ('REQ-004', 'ACCT-104', 6000.0, 2000.0, '2024-11-15', 'Emergency auto repair'),
        ('REQ-005', 'ACCT-105', 3000.0, 2000.0, '2024-11-16', 'Credit building'),
        ('REQ-006', 'ACCT-106', 15000.0, 7000.0, '2024-11-17', 'Working capital expansion'),
    ]
    for row in requests_data:
        conn.execute("INSERT INTO limit_requests VALUES (?, ?, ?, ?, ?, ?)", row)

    inquiries_data = [
        ('INQ-001', 'ACCT-101', '2024-05-12', 'soft_pull', 'Account review'),
        ('INQ-002', 'ACCT-102', '2024-10-01', 'hard_pull', 'Auto loan application'),
        ('INQ-003', 'ACCT-103', '2024-08-20', 'soft_pull', 'Promotional review'),
        ('INQ-004', 'ACCT-104', '2024-09-15', 'hard_pull', 'Personal loan application'),
        ('INQ-005', 'ACCT-105', '2024-11-01', 'hard_pull', 'Credit card account opening'),
        ('INQ-006', 'ACCT-106', '2024-07-11', 'hard_pull', 'Equipment lease inquiry'),
    ]
    for row in inquiries_data:
        conn.execute("INSERT INTO bureau_inquiries VALUES (?, ?, ?, ?, ?)", row)

    conn.close()
    print(f"Generated Credit Risk DuckDB database at {db_path} successfully.")


if __name__ == "__main__":
    generate_database()
