import pytest
from agent_runtime.mcp_layer.servers.analytics_server import (
    describe_tables,
    execute_sql,
    test_significance as run_significance_test,
)


def test_describe_tables():
    res = describe_tables()
    assert "daily_checkout_metrics" in res.tables
    assert "experiments" in res.tables
    assert "experiment_results" in res.tables
    assert "funnel_dropoff" in res.tables
    assert len(res.summary) > 0


def test_execute_sql_valid_select_and_cte():
    # Basic SELECT
    res = execute_sql("SELECT COUNT(*) as total_rows FROM daily_checkout_metrics")
    assert res.ok
    assert res.row_count == 1
    assert "total_rows" in res.columns

    # CTE query
    cte_query = """
    WITH mobile_data AS (
        SELECT device, AVG(conversion_rate) as avg_conv
        FROM daily_checkout_metrics
        WHERE device = 'mobile'
        GROUP BY device
    )
    SELECT * FROM mobile_data
    """
    cte_res = execute_sql(cte_query)
    assert cte_res.ok
    assert cte_res.row_count == 1
    assert cte_res.rows[0]["device"] == "mobile"


def test_execute_sql_rejects_destructive_and_external_operations():
    # DROP
    drop_res = execute_sql("DROP TABLE experiments")
    assert not drop_res.ok
    assert "Disallowed operations" in drop_res.error

    # DELETE
    del_res = execute_sql("DELETE FROM daily_checkout_metrics WHERE 1=1")
    assert not del_res.ok
    assert "Disallowed operations" in del_res.error

    # INSERT
    ins_res = execute_sql("INSERT INTO experiments VALUES ('EXP-999')")
    assert not ins_res.ok

    # UPDATE
    upd_res = execute_sql("UPDATE experiments SET name='Hacked'")
    assert not upd_res.ok

    # Non-SELECT / Non-WITH statement
    bad_stmt = execute_sql("PRAGMA version")
    assert not bad_stmt.ok


def test_hypothesis_testing_significance():
    sig = run_significance_test("EXP-204", alpha=0.05)
    assert sig.experiment_id == "EXP-204"
    assert sig.sample_size_a == 1500
    assert sig.sample_size_b == 1500
    assert sig.conversions_a == 24
    assert sig.conversions_b == 44
    assert sig.rate_a == 0.016
    assert round(sig.rate_b, 3) == 0.029
    assert sig.difference_pp > 1.0
    assert sig.p_value < 0.05
    assert sig.significant is True
