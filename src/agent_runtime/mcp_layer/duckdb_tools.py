"""Shared, read-only DuckDB query toolkit for MCP servers.

WHY THIS FILE EXISTS
    The product-analytics and credit-risk servers each carried their own
    near-identical copy of this logic. Duplicated safety code is a defect in
    its own right: a hardening fix applied to one copy silently misses the
    other. Both servers now call these functions.

DEFENCE IN DEPTH, IN ORDER OF STRENGTH
    1. The connection is opened `read_only=True` with external access and
       unsigned extensions disabled. This is the real guarantee: even a query
       that slipped past every check below cannot write, read local files, or
       reach the network.
    2. A single statement only. "SELECT 1; DROP TABLE x" is refused before it
       reaches DuckDB.
    3. A keyword blocklist, applied after string literals and comments are
       removed. The previous version tokenised the raw query, so a harmless
       filter such as `WHERE inquiry_purpose = 'call'` was rejected because
       "call" is a blocked keyword - a false positive with no security value.
    4. The statement must begin with a read-only verb.

OUTPUT BOUNDS
    Row counts are capped server-side (`MAX_ROWS`). The previous version let
    the caller choose `max_rows` freely and reported only the number of rows
    *returned*, so an agent could not tell a complete 50-row result from a
    truncated one - and would then summarise a partial result as if it were
    the whole table. Results now carry `truncated`.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, Field

MAX_ROWS = 200

_PROHIBITED_KEYWORDS = frozenset({
    "insert", "update", "delete", "drop", "alter", "create", "attach", "detach", "copy", "install",
    "load", "pragma", "checkpoint", "export", "import", "vacuum", "call", "set", "reset", "use",
})
_READ_ONLY_VERBS = frozenset({"select", "with", "describe", "show", "summarize"})


class SqlQueryResult(BaseModel):
    ok: bool
    query: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: str | None = None


class SchemaDescriptionResult(BaseModel):
    tables: dict[str, list[dict[str, str]]]
    summary: str


@contextmanager
def read_only_connection(db_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a locked-down connection and guarantee it is closed.

    Using `with` fixes a handle leak: previously a query that raised skipped
    `conn.close()`, leaking a file handle in a long-lived server process
    (on Windows that can also keep the database file locked).
    """
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Generate it with the domain's data/generate_data.py."
        )
    connection = duckdb.connect(
        str(db_path),
        read_only=True,
        config={"enable_external_access": False, "allow_unsigned_extensions": False},
    )
    try:
        yield connection
    finally:
        connection.close()


def _strip_literals_and_comments(query: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    without_line_comments = re.sub(r"--[^\n]*", " ", without_block_comments)
    # '' inside a literal is an escaped quote in SQL, so match it as part of one.
    return re.sub(r"'(?:[^']|'')*'", "''", without_line_comments)


def validate_read_only_sql(query: str) -> str | None:
    """Return an error message if `query` is not an acceptable single
    read-only statement, else None."""
    statement = query.strip().rstrip(";").strip()
    code = _strip_literals_and_comments(statement)
    if ";" in code:
        return "Only a single SQL statement is allowed."
    tokens = re.findall(r"\b[a-z_]+\b", code.lower())
    disallowed = sorted({token for token in tokens if token in _PROHIBITED_KEYWORDS})
    if disallowed:
        return f"Disallowed operations detected in query: {disallowed}. Read-only SELECT queries only."
    if not tokens or tokens[0] not in _READ_ONLY_VERBS:
        return "Query must be a read-only SELECT or WITH statement."
    return None


def _json_safe(value: Any) -> Any:
    """DuckDB returns dates and decimals; MCP results must be JSON."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def execute_read_only(db_path: Path, query: str, max_rows: int = 50) -> SqlQueryResult:
    """Validate and run one read-only query, returning at most MAX_ROWS rows."""
    error = validate_read_only_sql(query)
    if error:
        return SqlQueryResult(ok=False, query=query, error=error)

    row_limit = max(1, min(int(max_rows), MAX_ROWS))
    statement = query.strip().rstrip(";").strip()
    try:
        with read_only_connection(db_path) as connection:
            cursor = connection.execute(statement)
            if cursor.description is None:
                return SqlQueryResult(ok=False, query=query, error="Statement returned no result set.")
            columns = [description[0] for description in cursor.description]
            # Fetch one extra row: if it arrives, the result was cut off.
            fetched = cursor.fetchmany(row_limit + 1)
    except Exception as exc:  # noqa: BLE001 - SQL errors are reported to the agent, not raised
        return SqlQueryResult(ok=False, query=query, error=f"SQL execution error: {exc}")

    truncated = len(fetched) > row_limit
    rows = [
        {column: _json_safe(value) for column, value in zip(columns, row, strict=True)}
        for row in fetched[:row_limit]
    ]
    return SqlQueryResult(ok=True, query=query, columns=columns, rows=rows, row_count=len(rows), truncated=truncated)


def describe_database(db_path: Path) -> SchemaDescriptionResult:
    """Every table with its columns and types."""
    try:
        with read_only_connection(db_path) as connection:
            tables: dict[str, list[dict[str, str]]] = {}
            for (table_name,) in connection.execute("SHOW TABLES").fetchall():
                # Identifiers come from the database catalogue, not from the
                # caller; quoting them still keeps unusual names valid.
                columns = connection.execute(f'DESCRIBE "{table_name}"').fetchall()
                tables[table_name] = [{"column": column[0], "type": column[1]} for column in columns]
    except Exception as exc:  # noqa: BLE001
        return SchemaDescriptionResult(tables={}, summary=f"Error discovering schemas: {exc}")

    summary = "\n".join(
        f"Table '{name}': " + ", ".join(f"{column['column']} ({column['type']})" for column in columns)
        for name, columns in tables.items()
    )
    return SchemaDescriptionResult(tables=tables, summary=summary)
