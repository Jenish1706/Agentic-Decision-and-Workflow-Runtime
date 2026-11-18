import pytest

from agent_runtime.mcp_layer.duckdb_tools import MAX_ROWS, execute_read_only, validate_read_only_sql
from domains.product_analytics.server import DB_PATH


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "  with t as (select 1 as x) select * from t;  ",
        "DESCRIBE experiments",
        # Regression: blocked keywords inside literals and comments were false
        # positives ('call' is a blocked keyword; this filter is harmless).
        "SELECT * FROM bureau_inquiries WHERE inquiry_purpose = 'call center; update'",
        "SELECT 1 -- drop table accounts",
        "SELECT 'it''s fine' AS s",
    ],
)
def test_read_only_queries_are_accepted(query):
    assert validate_read_only_sql(query) is None


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("SELECT 1; DROP TABLE experiments", "single SQL statement"),
        ("SELECT 1; SELECT 2", "single SQL statement"),
        ("ATTACH 'other.db' AS o", "Disallowed"),
        ("COPY experiments TO 'out.csv'", "Disallowed"),
        ("INSTALL httpfs", "Disallowed"),
        ("VALUES (1)", "read-only SELECT"),
        ("", "read-only SELECT"),
    ],
)
def test_unsafe_or_multiple_statements_are_rejected(query, reason):
    error = validate_read_only_sql(query)
    assert error is not None and reason in error


def test_results_report_truncation():
    # Regression: only the returned row count was reported, so a capped result
    # was indistinguishable from a complete one.
    capped = execute_read_only(DB_PATH, "SELECT * FROM range(500)", max_rows=10_000)
    assert capped.ok and capped.row_count == MAX_ROWS and capped.truncated

    complete = execute_read_only(DB_PATH, "SELECT * FROM range(5)", max_rows=10)
    assert complete.row_count == 5 and not complete.truncated


def test_file_access_is_disabled_even_for_a_select():
    result = execute_read_only(DB_PATH, "SELECT * FROM read_csv_auto('pyproject.toml')")
    assert not result.ok


def test_missing_database_is_reported(tmp_path):
    result = execute_read_only(tmp_path / "missing.duckdb", "SELECT 1")
    assert not result.ok and "generate_data.py" in result.error


def test_double_quoted_identifiers_do_not_trip_blocklist():
    assert validate_read_only_sql('SELECT "set" FROM experiments') is None
    assert validate_read_only_sql('SELECT "call" AS "use" FROM experiments') is None


def test_runaway_query_times_out():
    result = execute_read_only(DB_PATH, "SELECT count(*) FROM range(10000000000)", timeout_seconds=0.5)
    assert not result.ok
    assert "timed out" in result.error.lower()

