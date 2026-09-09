"""MCP Analytics Server: Read-only DuckDB SQL query execution, schema discovery,
and statistical hypothesis testing over local product analytics tables.

Security & Safety Controls:
1. Connection-level isolation: `read_only=True` and `enable_external_access=False`
   prevent write operations, external file loading, and network calls.
2. Defense-in-depth query validation: Disallows destructive DDL/DML, requires SELECT/WITH,
   and restricts output size to avoid blowing up context windows.
"""

from pathlib import Path
import re
from typing import Any
import duckdb
from pydantic import BaseModel
import scipy.stats as stats

from mcp.server.mcpserver import MCPServer
from ...config import REPO_ROOT

server = MCPServer("product-analytics")

# Path to local product analytics database
DB_PATH = REPO_ROOT / "domains" / "product_analytics" / "data" / "product_analytics.duckdb"

_PROHIBITED_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "attach",
    "detach", "copy", "install", "load", "pragma", "checkpoint", "export",
    "import", "vacuum", "call", "write_parquet", "write_csv",
}


def _get_connection() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        # Auto-initialize if database file does not exist yet
        from domains.product_analytics.data.generate_data import generate_database
        generate_database(DB_PATH)

    return duckdb.connect(
        str(DB_PATH),
        read_only=True,
        config={"enable_external_access": False, "allow_unsigned_extensions": False},
    )


class SqlQueryResult(BaseModel):
    ok: bool
    query: str
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    row_count: int = 0
    error: str | None = None


class SchemaDescriptionResult(BaseModel):
    tables: dict[str, list[dict[str, str]]]
    summary: str


class SignificanceResult(BaseModel):
    ok: bool = True
    experiment_id: str
    metric: str
    statistic: float
    p_value: float
    sample_size_a: int
    sample_size_b: int
    conversions_a: int
    conversions_b: int
    rate_a: float
    rate_b: float
    difference_pp: float
    confidence_interval_95: list[float]
    significant: bool
    alpha: float
    test_name: str
    error: str | None = None


@server.tool()
def execute_sql(query: str, max_rows: int = 50) -> SqlQueryResult:
    """Execute a read-only SQL query against the product analytics database.

    Supports SELECT, WITH CTEs, aggregations, filters, and joins over:
    - `daily_checkout_metrics` (date, period, device, traffic_source, conversion_rate)
    - `funnel_dropoff` (funnel_step, step_order, mobile_dropoff_rate, desktop_dropoff_rate)
    - `experiments` (experiment_id, name, target_segment, hypothesis, status)
    - `experiment_results` (experiment_id, variant, users_assigned, conversions, conversion_rate)

    Destructive operations (INSERT/UPDATE/DROP/ALTER) and external file reads are strictly prohibited.
    """
    clean_q = query.strip().rstrip(";").strip()
    tokens = re.findall(r"\b[A-Za-z_]+\b", clean_q.lower())

    # 1. Reject destructive / external keywords
    disallowed = [t for t in tokens if t in _PROHIBITED_KEYWORDS]
    if disallowed:
        return SqlQueryResult(
            ok=False,
            query=query,
            error=f"Disallowed operations detected in query: {sorted(set(disallowed))}. Read-only SELECT queries only.",
        )

    # 2. Must start with SELECT, WITH, or EXPLAIN
    if not tokens or tokens[0] not in ("select", "with", "explain", "describe", "show"):
        return SqlQueryResult(
            ok=False,
            query=query,
            error="Query must be a read-only SELECT or WITH statement.",
        )

    try:
        conn = _get_connection()
        cursor = conn.execute(clean_q)
        col_names = [desc[0] for desc in cursor.description]
        fetched_rows = cursor.fetchmany(max_rows)
        row_dicts = [dict(zip(col_names, row)) for row in fetched_rows]
        conn.close()

        return SqlQueryResult(
            ok=True,
            query=query,
            columns=col_names,
            rows=row_dicts,
            row_count=len(row_dicts),
        )
    except Exception as exc:
        return SqlQueryResult(
            ok=False,
            query=query,
            error=f"SQL execution error: {exc}",
        )


@server.tool()
def describe_tables() -> SchemaDescriptionResult:
    """Discover available table schemas, column names, and data types in the product analytics database."""
    conn = _get_connection()
    tables: dict[str, list[dict[str, str]]] = {}

    try:
        table_rows = conn.execute("SHOW TABLES").fetchall()
        for (tbl_name,) in table_rows:
            schema_rows = conn.execute(f"DESCRIBE {tbl_name}").fetchall()
            tables[tbl_name] = [
                {"column": col[0], "type": col[1]}
                for col in schema_rows
            ]
        conn.close()

        summary_lines = []
        for t_name, cols in tables.items():
            col_list = ", ".join(f"{c['column']} ({c['type']})" for c in cols)
            summary_lines.append(f"Table '{t_name}': {col_list}")

        return SchemaDescriptionResult(
            tables=tables,
            summary="\n".join(summary_lines),
        )
    except Exception as exc:
        conn.close()
        return SchemaDescriptionResult(tables={}, summary=f"Error discovering schemas: {exc}")


@server.tool()
def test_significance(experiment_id: str = "EXP-204", alpha: float = 0.05) -> SignificanceResult:
    """Run a two-proportion hypothesis z-test to determine if variant B outperforms variant A.

    Calculates z-statistic, two-tailed p-value, percentage point difference, and 95% confidence interval.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT variant, users_assigned, conversions, conversion_rate "
            "FROM experiment_results WHERE experiment_id = ? ORDER BY variant",
            [experiment_id],
        ).fetchall()
        conn.close()
    except Exception:
        conn.close()
        raise

    if len(rows) < 2:
        raise ValueError(f"Experiment '{experiment_id}' requires at least 2 variants for comparison.")

    # rows[0] is A (control), rows[1] is B (treatment)
    var_a, n_a, conv_a, rate_a = rows[0]
    var_b, n_b, conv_b, rate_b = rows[1]

    # Two-proportion pooled z-test
    p1 = conv_a / n_a
    p2 = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = (p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b)) ** 0.5

    z_stat = (p2 - p1) / se_pool if se_pool > 0 else 0.0
    p_val = 2.0 * stats.norm.sf(abs(z_stat))

    diff_pp = (p2 - p1) * 100.0
    se_diff = ((p1 * (1 - p1) / n_a) + (p2 * (1 - p2) / n_b)) ** 0.5
    margin = 1.96 * se_diff * 100.0
    ci_95 = [round(diff_pp - margin, 3), round(diff_pp + margin, 3)]

    return SignificanceResult(
        experiment_id=experiment_id,
        metric="conversion_rate",
        statistic=round(float(z_stat), 4),
        p_value=round(float(p_val), 5),
        sample_size_a=n_a,
        sample_size_b=n_b,
        conversions_a=conv_a,
        conversions_b=conv_b,
        rate_a=round(float(rate_a), 4),
        rate_b=round(float(rate_b), 4),
        difference_pp=round(float(diff_pp), 3),
        confidence_interval_95=ci_95,
        significant=bool(p_val < alpha),
        alpha=alpha,
        test_name="two_proportion_z_test",
    )


if __name__ == "__main__":
    server.run()
