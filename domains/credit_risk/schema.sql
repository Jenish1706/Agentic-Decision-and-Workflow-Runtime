-- Schema for Credit Risk & Lending Domain Database (credit_risk.duckdb)

CREATE TABLE accounts (
    account_id VARCHAR PRIMARY KEY,
    customer_name VARCHAR,
    account_type VARCHAR,          -- 'consumer_revolving', 'smb_credit_line'
    credit_score INTEGER,          -- FICO / Bureau score (300 - 850)
    current_limit DOUBLE,
    annual_income DOUBLE,
    monthly_debt_obligations DOUBLE,
    dti_ratio DOUBLE,              -- Debt-to-Income (monthly_debt / (annual_income / 12))
    employment_status VARCHAR,
    account_status VARCHAR         -- 'active', 'restricted', 'closed'
);

CREATE TABLE credit_history (
    account_id VARCHAR PRIMARY KEY,
    delinquencies_30d INTEGER,
    delinquencies_60d INTEGER,
    delinquencies_90d INTEGER,
    revolving_utilization DOUBLE,  -- Balance / Current Limit
    on_time_payment_pct DOUBLE,
    months_on_books INTEGER
);

CREATE TABLE limit_requests (
    request_id VARCHAR PRIMARY KEY,
    account_id VARCHAR,
    requested_limit DOUBLE,
    increase_amount DOUBLE,
    request_date DATE,
    request_reason VARCHAR
);

CREATE TABLE bureau_inquiries (
    inquiry_id VARCHAR PRIMARY KEY,
    account_id VARCHAR,
    inquiry_date DATE,
    inquiry_type VARCHAR,          -- 'hard_pull', 'soft_pull'
    inquiry_purpose VARCHAR
);
