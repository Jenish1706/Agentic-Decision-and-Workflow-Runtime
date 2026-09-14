"""Builds data/product_analytics.duckdb for the product analytics domain.

Tables (schema in ../schema.sql, which this script executes - it is the
single source of truth; the previous version re-declared every table inline,
so the two could drift):
    daily_checkout_metrics  checkout funnel outcomes by date, week, device and traffic source
    funnel_dropoff          drop-off rate per checkout step, mobile vs desktop
    experiments             A/B experiment metadata (EXP-204)
    experiment_results      per-variant users and conversions

The rows are hand-authored, not random, so every benchmark answer can be
checked by reading them. They encode one business story:
    - Mobile checkout conversion fell from about 2.8% (baseline_week) to about
      1.6% (last_week); desktop stayed at about 4.0-4.1%.
    - The drop concentrates at the payment_to_confirmation step (48% mobile
      drop-off vs 12% on desktop).
    - Experiment EXP-204 (one-page mobile checkout) converted 24/1500 = 1.60%
      in control (A) and 44/1500 = 2.93% in treatment (B); a two-proportion
      z-test gives p of about 0.014.

(The previous docstring described `users`, `sessions`, `funnel_events` and
`experiment_assignments` tables that this script never created, and seeded a
random generator it never used.)

Run: uv run python domains/product_analytics/data/generate_data.py
"""

from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).resolve().parent
DB_PATH = DATA_DIR / "product_analytics.duckdb"
SCHEMA_PATH = DATA_DIR.parent / "schema.sql"

# (metric_date, period, device, traffic_source, checkouts_started, purchases_completed, conversion_rate)
# Two sample days per week, not seven: enough rows to aggregate and segment.
DAILY_CHECKOUT_METRICS = [
    ("2024-10-25", "baseline_week", "mobile", "organic", 2500, 70, 0.028),
    ("2024-10-25", "baseline_week", "mobile", "paid_search", 2000, 56, 0.028),
    ("2024-10-25", "baseline_week", "desktop", "organic", 1200, 50, 0.0417),
    ("2024-10-25", "baseline_week", "desktop", "paid_search", 800, 32, 0.040),
    ("2024-10-28", "baseline_week", "mobile", "organic", 2600, 73, 0.0281),
    ("2024-10-28", "baseline_week", "desktop", "organic", 1300, 53, 0.0408),
    # last_week: mobile conversion drops after a release; desktop holds steady.
    ("2024-11-04", "last_week", "mobile", "organic", 2700, 43, 0.0159),
    ("2024-11-04", "last_week", "mobile", "paid_search", 2100, 34, 0.0162),
    ("2024-11-04", "last_week", "desktop", "organic", 1250, 51, 0.0408),
    ("2024-11-04", "last_week", "desktop", "paid_search", 850, 34, 0.0400),
    ("2024-11-07", "last_week", "mobile", "organic", 2650, 42, 0.0158),
    ("2024-11-07", "last_week", "desktop", "organic", 1300, 52, 0.0400),
]

# (funnel_step, step_order, mobile_dropoff_rate, desktop_dropoff_rate)
FUNNEL_DROPOFF = [
    ("cart_to_shipping", 1, 0.15, 0.14),
    ("shipping_to_payment", 2, 0.18, 0.16),
    ("payment_to_confirmation", 3, 0.48, 0.12),  # the mobile failure point
]

EXPERIMENTS = [
    (
        "EXP-204",
        "Mobile One-Page Checkout Redesign",
        "mobile",
        "Eliminating 3-step checkout reduces mobile friction and recovers conversion drop.",
        "completed",
        "2024-11-01",
        "2024-11-14",
    ),
]

# (experiment_id, variant, users_assigned, conversions, conversion_rate)
EXPERIMENT_RESULTS = [
    ("EXP-204", "A", 1500, 24, 0.0160),
    ("EXP-204", "B", 1500, 44, 0.0293),
]


def generate_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)  # rebuild from scratch; the data is fully defined above

    connection = duckdb.connect(str(db_path))
    try:
        connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        connection.executemany("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?)", EXPERIMENTS)
        connection.executemany("INSERT INTO daily_checkout_metrics VALUES (?, ?, ?, ?, ?, ?, ?)", DAILY_CHECKOUT_METRICS)
        connection.executemany("INSERT INTO funnel_dropoff VALUES (?, ?, ?, ?)", FUNNEL_DROPOFF)
        connection.executemany("INSERT INTO experiment_results VALUES (?, ?, ?, ?, ?)", EXPERIMENT_RESULTS)
    finally:
        connection.close()
    print(f"Generated {db_path}")


if __name__ == "__main__":
    generate_database()
