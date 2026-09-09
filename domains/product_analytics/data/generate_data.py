"""Data generator for Product Analytics domain.

Generates realistic e-commerce analytics data:
- `users`: User demographic & acquisition attributes.
- `sessions`: Session telemetry (device, traffic source, duration, date).
- `funnel_events`: Step-by-step funnel stages (home, search, product_page, cart, checkout, purchase).
- `experiments`: A/B testing metadata (EXP-204 mobile checkout redesign).
- `experiment_assignments`: Per-user variant allocation and conversion outcome.

The data embodies a real business phenomenon:
- Mobile checkout conversion experienced an unexpected drop last week (down from 2.8% to 1.6%).
- Desktop conversion remained healthy (4.0% - 4.1%).
- Experiment EXP-204 tested a streamlined mobile checkout flow:
  Variant A (Control) = 1.6% conversion
  Variant B (Treatment) = 2.9% conversion (p < 0.05, statistically significant recovery).
"""

from pathlib import Path
import random
import duckdb

DATA_DIR = Path(__file__).parent
DB_PATH = DATA_DIR / "product_analytics.duckdb"


def generate_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = duckdb.connect(str(db_path))
    random.seed(42)

    # 1. Experiments
    conn.execute("""
        CREATE TABLE experiments (
            experiment_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            target_segment VARCHAR,
            hypothesis VARCHAR,
            status VARCHAR,
            start_date DATE,
            end_date DATE
        )
    """)
    conn.execute("""
        INSERT INTO experiments VALUES (
            'EXP-204',
            'Mobile One-Page Checkout Redesign',
            'mobile',
            'Eliminating 3-step checkout reduces mobile friction and recovers conversion drop.',
            'completed',
            '2024-11-01',
            '2024-11-14'
        )
    """)

    # 2. Daily Checkout Metrics (Summary Aggregations)
    conn.execute("""
        CREATE TABLE daily_checkout_metrics (
            metric_date DATE,
            period VARCHAR, -- 'baseline_week' vs 'last_week'
            device VARCHAR, -- 'mobile', 'desktop', 'tablet'
            traffic_source VARCHAR, -- 'organic', 'paid_search', 'direct'
            checkouts_started INTEGER,
            purchases_completed INTEGER,
            conversion_rate DOUBLE
        )
    """)

    # Populate daily metrics:
    # Baseline week: 7 days
    # Desktop: ~4.1%
    # Mobile: ~2.8%
    # Last week: 7 days
    # Desktop: ~4.0% (steady)
    # Mobile: ~1.6% (sharp drop!)
    rows = [
        # Baseline week (Total checkouts: ~10,000, 70% mobile)
        ('2024-10-25', 'baseline_week', 'mobile', 'organic', 2500, 70, 0.028),
        ('2024-10-25', 'baseline_week', 'mobile', 'paid_search', 2000, 56, 0.028),
        ('2024-10-25', 'baseline_week', 'desktop', 'organic', 1200, 50, 0.0417),
        ('2024-10-25', 'baseline_week', 'desktop', 'paid_search', 800, 32, 0.040),
        ('2024-10-28', 'baseline_week', 'mobile', 'organic', 2600, 73, 0.0281),
        ('2024-10-28', 'baseline_week', 'desktop', 'organic', 1300, 53, 0.0408),
        
        # Last week (Post-release drop on mobile)
        ('2024-11-04', 'last_week', 'mobile', 'organic', 2700, 43, 0.0159),
        ('2024-11-04', 'last_week', 'mobile', 'paid_search', 2100, 34, 0.0162),
        ('2024-11-04', 'last_week', 'desktop', 'organic', 1250, 51, 0.0408),
        ('2024-11-04', 'last_week', 'desktop', 'paid_search', 850, 34, 0.0400),
        ('2024-11-07', 'last_week', 'mobile', 'organic', 2650, 42, 0.0158),
        ('2024-11-07', 'last_week', 'desktop', 'organic', 1300, 52, 0.0400),
    ]
    for r in rows:
        conn.execute("INSERT INTO daily_checkout_metrics VALUES (?, ?, ?, ?, ?, ?, ?)", r)

    # 3. Funnel Steps Table (where in funnel did users drop?)
    conn.execute("""
        CREATE TABLE funnel_dropoff (
            funnel_step VARCHAR,
            step_order INTEGER,
            mobile_dropoff_rate DOUBLE,
            desktop_dropoff_rate DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO funnel_dropoff VALUES
        ('cart_to_shipping', 1, 0.15, 0.14),
        ('shipping_to_payment', 2, 0.18, 0.16),
        ('payment_to_confirmation', 3, 0.48, 0.12) -- Mobile catastrophic failure on payment step!
    """)

    # 4. Experiment Assignments & Results for EXP-204
    conn.execute("""
        CREATE TABLE experiment_results (
            experiment_id VARCHAR,
            variant VARCHAR, -- 'A' (control) vs 'B' (treatment)
            users_assigned INTEGER,
            conversions INTEGER,
            conversion_rate DOUBLE
        )
    """)
    # 24 / 1500 = 1.6% vs 44 / 1500 = 2.93%
    conn.execute("""
        INSERT INTO experiment_results VALUES
        ('EXP-204', 'A', 1500, 24, 0.0160),
        ('EXP-204', 'B', 1500, 44, 0.0293)
    """)

    conn.close()
    print(f"Generated DuckDB database at {db_path} successfully.")


if __name__ == "__main__":
    generate_database()
