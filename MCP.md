# MCP in this project

## What problem MCP solves

Without a protocol, "giving an LLM a tool" means writing a Python function
and registering it directly in whatever process the agent runs in. That's
fine until the tool needs to live somewhere else - a different language, a
different machine, a service someone else owns and maintains independently
of your agent code. MCP (Model Context Protocol) standardizes the boundary
between "a process that wants to call tools" (a client) and "a process that
exposes tools" (a server): discovery (what tools exist, what do they take),
invocation (call one, get a structured result back), and transport
(stdio for a local subprocess, HTTP for a remote service) are all part of
one protocol instead of a bespoke integration per tool source.

This project uses the [official Python SDK](https://github.com/modelcontextprotocol/python-sdk)
(package `mcp`, v2). It does not reimplement the wire protocol; that would
teach nothing. What it builds is everything around the protocol: six real
servers, a client wrapper, and the adapter that lets an MCP tool sit in the
same registry as a local one.

## What's actually running

Every server is a genuinely separate OS process, built with `MCPServer` (the
SDK's decorator-based server class - `@server.tool()`, `@server.resource(uri)`),
which derives each tool's JSON Schema from Python type hints.

**Knowledge base** (connected by default, `src/agent_runtime/mcp_layer/servers/`):

- **`knowledge`** ([knowledge_server.py](src/agent_runtime/mcp_layer/servers/knowledge_server.py)) -
  `search_documents(query, top_k)` runs the RAG pipeline (embed, brute-force
  cosine search, cross-encoder rerank), plus a `documents://list` **resource**
  (a distinct MCP primitive from a tool: read-only, no arguments, listed and
  fetched rather than "called").
- **`records`** ([records_server.py](src/agent_runtime/mcp_layer/servers/records_server.py)) -
  `lookup_record(record_id)` and `search_records(field, value)` over a small
  JSON store (`data/records.json`). Exact field matching - a deliberately
  different tool shape from semantic search.

**Business domains** (opt-in, one server per folder under `domains/`):

| Server | Tools |
|---|---|
| [product_analytics](domains/product_analytics/server.py) | `execute_sql`, `describe_tables`, `test_significance` |
| [credit_risk](domains/credit_risk/server.py) | `execute_sql`, `describe_tables`, `get_account_profile` |
| [multimodal_investigation](domains/multimodal_investigation/server.py) | `extract_document_metrics`, `analyze_chart_data`, `list_visual_artifacts` |
| [enterprise_operations](domains/enterprise_operations/server.py) | `get_incident_telemetry`, `list_service_incidents`, `inspect_service_config`, `simulate_remediation_action` |

Domain servers live with their data and policies rather than inside the
generic runtime package; `agent_runtime.domains.load_domain(name).server_modules`
hands their file paths to the client. The multimodal server's "vision"
tools read pre-extracted JSON fixtures - no image model runs (see the
server's docstring for why that is still a meaningful test of the runtime).

None of the servers is a mock: `search_documents` really builds an embedding
index and calls a cross-encoder; the SQL tools really query DuckDB.

## Client side: discovery and invocation

[mcp_layer/client.py](src/agent_runtime/mcp_layer/client.py)'s `MCPToolClient`
is the one place in the codebase that speaks the MCP client protocol. On
`__aenter__` it launches each server as a subprocess over stdio - a dotted
module name runs as `python -m <module>`, a `.py` path runs as a script,
both with the repository root as working directory - and keeps the sessions
open for the client's lifetime, so the model-loading cost inside `knowledge`
is paid once per session, not once per query. If the third of three servers
fails to start, the two already running are shut down before the error
propagates, rather than leaked as orphan processes.

```python
async with MCPToolClient() as client:                     # knowledge + records
    tools = await client.discover_tools()                 # names, descriptions, schemas, metadata
    result = await client.call_tool("records", "lookup_record", {"record_id": "ferry"})

spec = load_domain("credit_risk")
async with MCPToolClient(server_modules=spec.server_modules) as client:
    ...
```

`discover_tools()` calls `list_tools()` on every connected server - genuine
runtime discovery: nothing in the agent's code hard-codes what tools exist or
what arguments they take. `call_tool()` returns a `CallToolResult` with
`structured_content` (a tool's typed return value as JSON, which becomes the
evidence payload verbatim) or plain text blocks as a fallback.

## Tool categories travel in MCP metadata

Budgets and evidence provenance need to know what *kind* of tool ran: a SQL
query consumes the SQL budget and is recorded as `analytics_sql` evidence; a
document extraction consumes the vision budget. The earlier version guessed
this from the tool's *name* ("sql" in the name means SQL, "document" means
vision), which misfiled tools in every direction and never enforced the
vision budget at all.

Each server now declares it:

```python
@server.tool(meta={"category": "sql"})
def execute_sql(query: str, max_rows: int = 50) -> SqlQueryResult: ...
```

`meta` is part of the MCP tool definition, so the category arrives with
discovery; [agent/tools/mcp_tool.py](src/agent_runtime/agent/tools/mcp_tool.py)
maps it onto `ToolCategory` (unknown values fall back to `record`), and local
tools set the same attribute directly. `tests/test_mcp_servers.py` checks
every domain server's declarations over the real protocol.

## One safe SQL implementation, shared

Both database-backed servers use
[mcp_layer/duckdb_tools.py](src/agent_runtime/mcp_layer/duckdb_tools.py)
instead of each carrying a copy (a hardening fix applied to one copy would
silently miss the other). In order of strength:

1. Connections are opened `read_only=True` with external access disabled, so
   even a query that slipped past every check below cannot write, read local
   files, or reach the network.
2. One statement only - `SELECT 1; DROP TABLE x` is refused.
3. A keyword blocklist applied *after* string literals and comments are
   stripped (the old tokenizer rejected `WHERE purpose = 'call'`, because
   `call` is a blocked keyword).
4. The statement must start with a read-only verb.

Results are capped server-side and carry `truncated`, so the model can tell a
complete 50-row result from a cut-off one.

## The agent's `retrieve` action is itself MCP-mediated

It would be easy to read this project as "MCP handles some side tools,
while the real retrieval path stays local." That was true in an earlier
version and turned out to be a bug: see ARCHITECTURE.md's "Why `retrieve`
is MCP-mediated" for the full story. In the current version,
[agent/mcp_retriever.py](src/agent_runtime/agent/mcp_retriever.py)'s
`MCPRetriever` is what the single agent's dedicated `retrieve` action
actually calls - which itself calls `knowledge`'s `search_documents` tool
through the MCP client. Retrieval, the agent's most central capability, is
protocol-accessed, not just the auxiliary lookups.

(The multi-agent `RetrieverWorker` is the one exception - it calls
`RagPipeline` directly, in-process. That's a deliberate, different
trade-off made once the MCP-vs-local contrast is already established
elsewhere - see multi_agent/workers.py - not an oversight.)

## Local tool vs. MCP tool: the actual contrast

[agent/tools/calculator.py](src/agent_runtime/agent/tools/calculator.py)
(local) and [agent/tools/mcp_tool.py](src/agent_runtime/agent/tools/mcp_tool.py)
(MCP) both implement the same `Tool` interface
([agent/tools/base.py](src/agent_runtime/agent/tools/base.py)), and both
end up in the same `ToolRegistry`. The agent loop's decision-making never
branches on `origin` - it picks a tool by name and calls
`registry.execute(name, arguments)` identically either way. The difference
is entirely inside `run()`:

- `CalculatorTool.run()` calls a Python function directly, in-process, in
  microseconds.
- `MCPTool.run()` calls `self._client.call_tool(server_name, self.name, kwargs)`
  - a message over stdio to a different OS process, with real
  (de)serialization and IPC latency (milliseconds, not microseconds).

Tool names form one flat namespace, so the registry refuses a second tool
with an existing name. It used to overwrite silently - and two domain
servers both expose `execute_sql`, so connecting both would have sent every
query to whichever registered last, against the wrong database.

`scripts/demo_mcp.py` exercises the client directly with no agent
involved, specifically so this protocol path is visible on its own rather
than only ever seen wrapped inside agent reasoning.

## Why not implement the protocol from scratch

The MCP spec covers JSON-RPC framing, capability negotiation, transport
abstraction (stdio/SSE/streamable-HTTP), and versioning. None of that is
what this project is trying to teach; the official SDK already implements
it correctly and is the thing worth learning to use well. What *is* worth
building here - and is built here - is the layer above the SDK: turning
"a tool the SDK discovered" into "a tool the agent loop can pick, alongside
locally-registered ones, without caring where it actually runs."

## A known simplification

Each server subprocess builds its own state independently at startup - the
`knowledge` server rebuilds its RAG index from `data/documents/` every time
it's launched (a few seconds), because it shares no process memory with
whatever spawned it. That's not a shortcut around MCP; it's what "a
separate process" genuinely implies. A production deployment would run
`knowledge` as a long-lived service instead of a subprocess spawned per
session, so that cost is paid once, not once per client session.
