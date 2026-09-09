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
(package `mcp`, v2 - the current rework for the 2026-07-28 spec). It does
not reimplement the wire protocol; that would teach nothing. What it builds
is everything around the protocol: two real servers, a client wrapper, and
the adapter that lets an MCP tool sit in the same registry as a local one.

## What's actually running

Two servers, each a genuinely separate OS process:

- **`knowledge` server** ([mcp_layer/servers/knowledge_server.py](src/agent_runtime/mcp_layer/servers/knowledge_server.py)) -
  exposes `search_documents(query, top_k)` (runs the RAG pipeline - embed,
  brute-force cosine search, cross-encoder rerank) and a `documents://list`
  **resource** (a distinct MCP primitive from a tool: read-only, no
  arguments, meant to be listed/fetched rather than "called").
- **`records` server** ([mcp_layer/servers/records_server.py](src/agent_runtime/mcp_layer/servers/records_server.py)) -
  exposes `lookup_record(record_id)` and `search_records(field, value)` over
  a small JSON store (`data/records.json`) of Nimbus services and teams.
  No embeddings involved - exact field matching, a deliberately different
  tool shape from semantic search.

Both are built with `MCPServer` (the SDK's ergonomic, decorator-based
server class - `@server.tool()`, `@server.resource(uri)`), which derives
the JSON Schema for each tool's parameters from Python type hints
automatically. Neither server is a mock or a stub: `search_documents`
really does build an embedding index and call a cross-encoder;
`lookup_record` really does load and query the JSON file.

## Client side: discovery and invocation

[mcp_layer/client.py](src/agent_runtime/mcp_layer/client.py)'s `MCPToolClient`
is the one place in the codebase that speaks the MCP client protocol. On
`__aenter__` it launches each server as a subprocess (`StdioServerParameters(command=sys.executable, args=["-m", "<module>"])`)
and connects over stdio; it stays open for the process's lifetime rather
than reconnecting per call, so the model-loading cost inside `knowledge`
is paid once per session, not once per query.

```python
async with MCPToolClient() as client:
    tools = await client.discover_tools()       # tool discovery
    result = await client.call_tool("records", "lookup_record", {"record_id": "ferry"})
```

`discover_tools()` calls `list_tools()` on every connected server and
returns each tool's name, description, and JSON Schema - this is genuine
runtime discovery: nothing in the agent's code hard-codes what tools exist
or what arguments they take. `call_tool()` returns a `CallToolResult` with
either `structured_content` (when the tool's return type gives the SDK
enough to produce structured JSON, which is what this project's tools do)
or plain text content blocks as a fallback.

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
