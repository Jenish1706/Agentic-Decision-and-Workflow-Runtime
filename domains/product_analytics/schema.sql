-- Schema for Product Analytics Domain Database (product_analytics.duckdb)

CREATE TABLE experiments (
    experiment_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    target_segment VARCHAR,
    hypothesis VARCHAR,
    status VARCHAR,
    start_date DATE,
    end_date DATE
);

CREATE TABLE daily_checkout_metrics (
    metric_date DATE,
    period VARCHAR,            -- 'baseline_week' vs 'last_week'
    device VARCHAR,            -- 'mobile', 'desktop', 'tablet'
    traffic_source VARCHAR,    -- 'organic', 'paid_search', 'direct'
    checkouts_started INTEGER,
    purchases_completed INTEGER,
    conversion_rate DOUBLE
);

CREATE TABLE funnel_dropoff (
    funnel_step VARCHAR,
    step_order INTEGER,
    mobile_dropoff_rate DOUBLE,
    desktop_dropoff_rate DOUBLE
);

CREATE TABLE experiment_results (
    experiment_id VARCHAR,
    variant VARCHAR,           -- 'A' (control) vs 'B' (treatment)
    users_assigned INTEGER,
    conversions INTEGER,
    conversion_rate DOUBLE
);
