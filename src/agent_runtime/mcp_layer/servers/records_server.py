"""MCP Records Server: exact structured lookup over data/records.json
(Nimbus services + teams). Deliberately contrasts with the Knowledge
Server's semantic search - this tool does field equality matching over a
small JSON store, no embeddings involved, because not every knowledge
source is prose worth chunking and embedding."""

import json
from typing import Any

from pydantic import BaseModel

from mcp.server.mcpserver import MCPServer

from ...config import settings

server = MCPServer("nimbus-records")
_raw = json.loads(settings.records_path.read_text(encoding="utf-8"))
_all_records: dict[str, dict[str, Any]] = {row["id"]: row for group in _raw.values() for row in group}


class RecordResult(BaseModel):
    found: bool
    record: dict[str, Any] | None = None


class SearchResult(BaseModel):
    matches: list[dict[str, Any]]


@server.tool()
def lookup_record(record_id: str) -> RecordResult:
    """Look up one service or team record by its exact id, e.g. 'aurora' or 'nimbus-data'."""
    record = _all_records.get(record_id)
    return RecordResult(found=record is not None, record=record)


@server.tool()
def search_records(field: str, value: str) -> SearchResult:
    """Find all records where `field` equals `value`, e.g. field='owner_team', value='nimbus-data'.
    Useful fields: type (service|team), owner_team, tier, region."""
    matches = [record for record in _all_records.values() if str(record.get(field, "")).lower() == value.lower()]
    return SearchResult(matches=matches)


if __name__ == "__main__":
    server.run()
