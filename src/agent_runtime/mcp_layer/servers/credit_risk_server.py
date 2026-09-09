"""MCP Credit Risk Server: Read-only DuckDB SQL execution, schema discovery,
and consolidated account dossier retrieval for underwriting workflows.

Security & Safety Controls:
1. Connection-level isolation: `read_only=True` and `enable_external_access=False`
   prevent write operations, external file loading, and unauthorized network calls.
2. Defense-in-depth query validation: Blocks destructive DDL/DML (DROP, ALTER, UPDATE, DELETE).
"""

from pathlib import Path
import re
from typing import Any
import duckdb
from pydantic import BaseModel

from mcp.server.mcpserver import MCPServer
from ...config import REPO_ROOT

server = MCPServer("credit-risk")

# Path to local credit risk database
DB_PATH = REPO_ROOT / "domains" / "credit_risk" / "data" / "credit_risk.duckdb"

_PROHIBITED_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "attach",
    "detach", "copy", "install", "load", "pragma", "checkpoint", "export",
    "import", "vacuum", "call", "write_parquet", "write_csv",
}


def _get_connection() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        from domains.credit_risk.data.generate_data import generate_database
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


class AccountProfileResult(BaseModel):
    ok: bool
    account_id: str
    account_info: dict[str, Any] = {}
    credit_history: dict[str, Any] = {}
    open_requests: list[dict[str, Any]] = []
    inquiries: list[dict[str, Any]] = []
    error: str | None = None


@server.tool()
def execute_sql(query: str, max_rows: int = 50) -> SqlQueryResult:
    """Execute a read-only SQL query against the credit risk underwriting database.

    Available tables:
    - `accounts` (account_id, customer_name, account_type, credit_score, current_limit, annual_income, monthly_debt_obligations, dti_ratio, employment_status, account_status)
    - `credit_history` (account_id, delinquencies_30d, delinquencies_60d, delinquencies_90d, revolving_utilization, on_time_payment_pct, months_on_books)
    - `limit_requests` (request_id, account_id, requested_limit, increase_amount, request_date, request_reason)
    - `bureau_inquiries` (inquiry_id, account_id, inquiry_date, inquiry_type, inquiry_purpose)

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

    # 2. Must start with SELECT, WITH, EXPLAIN, DESCRIBE, or SHOW
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
    """Discover table schemas, column names, and data types in the credit risk underwriting database."""
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
def get_account_profile(account_id: str) -> AccountProfileResult:
    """Fetch complete underwriting profile for a given account_id, joining account details,
    credit history, recent limit requests, and bureau inquiries into a unified dossier.
    """
    clean_id = account_id.strip()
    conn = _get_connection()
    try:
        # Accounts
        acc_cur = conn.execute("SELECT * FROM accounts WHERE account_id = ?", [clean_id])
        acc_cols = [d[0] for d in acc_cur.description]
        acc_row = acc_cur.fetchone()
        if not acc_row:
            conn.close()
            return AccountProfileResult(
                ok=False,
                account_id=clean_id,
                error=f"Account '{clean_id}' not found in portfolio database.",
            )
        acc_dict = dict(zip(acc_cols, acc_row))

        # Credit History
        hist_cur = conn.execute("SELECT * FROM credit_history WHERE account_id = ?", [clean_id])
        hist_cols = [d[0] for d in hist_cur.description]
        hist_row = hist_cur.fetchone()
        hist_dict = dict(zip(hist_cols, hist_row)) if hist_row else {}

        # Limit Requests
        req_cur = conn.execute("SELECT * FROM limit_requests WHERE account_id = ?", [clean_id])
        req_cols = [d[0] for d in req_cur.description]
        req_rows = [dict(zip(req_cols, r)) for r in req_cur.fetchall()]

        # Inquiries
        inq_cur = conn.execute("SELECT * FROM bureau_inquiries WHERE account_id = ?", [clean_id])
        inq_cols = [d[0] for d in inq_cur.description]
        inq_rows = [dict(zip(inq_cols, r)) for r in inq_cur.fetchall()]

        conn.close()

        return AccountProfileResult(
            ok=True,
            account_id=clean_id,
            account_info=acc_dict,
            credit_history=hist_dict,
            open_requests=req_rows,
            inquiries=inq_rows,
        )
    except Exception as exc:
        conn.close()
        return AccountProfileResult(
            ok=False,
            account_id=clean_id,
            error=f"Error fetching profile: {exc}",
        )


if __name__ == "__main__":
    server.run()
