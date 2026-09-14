"""MCP server for the Credit Risk & Lending Underwriting domain.

Tools
    execute_sql          read-only SQL over the underwriting database
    describe_tables      schema discovery
    get_account_profile  one account's full dossier in a single call

Every tool declares a `category` in its MCP metadata for budget accounting
and evidence provenance. The dossier's nested records all carry `account_id`,
which is what lets policy rules attribute each fact to the right borrower
(agent/facts.py).
"""

from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from agent_runtime.mcp_layer.duckdb_tools import (
    SchemaDescriptionResult,
    SqlQueryResult,
    describe_database,
    execute_read_only,
    read_only_connection,
)

DB_PATH = Path(__file__).resolve().parent / "data" / "credit_risk.duckdb"

server = MCPServer("credit-risk")


@server.tool(meta={"category": "sql"})
def execute_sql(query: str, max_rows: int = 50) -> SqlQueryResult:
    """Execute one read-only SQL query against the credit risk underwriting database.

    Tables:
    - accounts (account_id, customer_name, account_type, credit_score, current_limit, annual_income,
      monthly_debt_obligations, dti_ratio, employment_status, account_status)
    - credit_history (account_id, delinquencies_30d, delinquencies_60d, delinquencies_90d,
      revolving_utilization, on_time_payment_pct, months_on_books)
    - limit_requests (request_id, account_id, requested_limit, increase_amount, request_date, request_reason)
    - bureau_inquiries (inquiry_id, account_id, inquiry_date, inquiry_type, inquiry_purpose)

    dti_ratio is a fraction (0.48 means 48%). Include account_id in SELECT lists so results can be
    attributed to the right account. At most 200 rows are returned; check `truncated`.
    """
    return execute_read_only(DB_PATH, query, max_rows)


@server.tool(meta={"category": "record"})
def describe_tables() -> SchemaDescriptionResult:
    """List the tables, columns and column types in the credit risk database."""
    return describe_database(DB_PATH)


class AccountProfileResult(BaseModel):
    ok: bool
    account_id: str
    account_info: dict[str, Any] = Field(default_factory=dict)
    credit_history: dict[str, Any] = Field(default_factory=dict)
    open_requests: list[dict[str, Any]] = Field(default_factory=list)
    inquiries: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


def _rows_as_dicts(connection, sql: str, account_id: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, [account_id])
    columns = [description[0] for description in cursor.description]
    return [
        {column: (value.isoformat() if hasattr(value, "isoformat") else value) for column, value in zip(columns, row, strict=True)}
        for row in cursor.fetchall()
    ]


@server.tool(meta={"category": "record"})
def get_account_profile(account_id: str) -> AccountProfileResult:
    """Fetch the complete underwriting dossier for one account: account details, credit history,
    limit-increase requests, and bureau inquiries."""
    clean_id = account_id.strip().upper()
    try:
        with read_only_connection(DB_PATH) as connection:
            accounts = _rows_as_dicts(connection, "SELECT * FROM accounts WHERE account_id = ?", clean_id)
            if not accounts:
                return AccountProfileResult(
                    ok=False, account_id=clean_id, error=f"Account '{clean_id}' not found in portfolio database."
                )
            history = _rows_as_dicts(connection, "SELECT * FROM credit_history WHERE account_id = ?", clean_id)
            requests = _rows_as_dicts(connection, "SELECT * FROM limit_requests WHERE account_id = ?", clean_id)
            inquiries = _rows_as_dicts(connection, "SELECT * FROM bureau_inquiries WHERE account_id = ?", clean_id)
    except Exception as exc:  # noqa: BLE001 - reported to the agent as a failed lookup
        return AccountProfileResult(ok=False, account_id=clean_id, error=f"Error fetching profile: {exc}")

    return AccountProfileResult(
        ok=True,
        account_id=clean_id,
        account_info=accounts[0],
        credit_history=history[0] if history else {},
        open_requests=requests,
        inquiries=inquiries,
    )


if __name__ == "__main__":
    server.run()
