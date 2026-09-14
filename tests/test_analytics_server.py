import pytest

from domains.product_analytics.server import describe_tables, execute_sql
from domains.product_analytics.server import test_significance as significance_test


def test_describe_tables():
    result = describe_tables()
    assert {"daily_checkout_metrics", "experiments", "experiment_results", "funnel_dropoff"} <= set(result.tables)
    assert result.summary


def test_select_and_cte_queries():
    count = execute_sql("SELECT COUNT(*) AS total_rows FROM daily_checkout_metrics")
    assert count.ok and count.row_count == 1 and count.columns == ["total_rows"]

    cte = execute_sql(
        """
        WITH mobile AS (
            SELECT device, AVG(conversion_rate) AS avg_conv FROM daily_checkout_metrics
            WHERE device = 'mobile' GROUP BY device
        )
        SELECT * FROM mobile
        """
    )
    assert cte.ok and cte.rows[0]["device"] == "mobile"


def test_writes_and_non_select_statements_are_rejected():
    for statement in (
        "DROP TABLE experiments",
        "DELETE FROM daily_checkout_metrics WHERE 1=1",
        "INSERT INTO experiments VALUES ('EXP-999')",
        "UPDATE experiments SET name='Hacked'",
        "PRAGMA version",
    ):
        assert not execute_sql(statement).ok, statement


def test_significance_matches_the_experiment_data():
    result = significance_test("EXP-204")
    assert (result.sample_size_a, result.conversions_a, result.sample_size_b, result.conversions_b) == (1500, 24, 1500, 44)
    assert (result.control_variant, result.treatment_variant) == ("A", "B")
    assert result.rate_a == 0.016 and result.rate_b == 0.0293
    assert result.difference_pp == pytest.approx(1.333, abs=1e-3)
    assert 0.01 < result.p_value < 0.05
    assert result.significant and result.treatment_better
    low, high = result.confidence_interval_95
    assert low < result.difference_pp < high


def test_significance_input_validation():
    with pytest.raises(ValueError, match="exactly 2 variants"):
        significance_test("EXP-999")
    with pytest.raises(ValueError, match="alpha"):
        significance_test("EXP-204", alpha=0.0)
    with pytest.raises(TypeError):
        significance_test()  # type: ignore[call-arg]  # no silent default experiment any more
