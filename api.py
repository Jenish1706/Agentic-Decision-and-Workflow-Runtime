"""Thin FastAPI wrapper over the single-agent runtime.

Not the primary interface - see scripts/ for that. This exists because the
project studies FastAPI alongside RAG/agents, so exposing the runtime as a
couple of HTTP endpoints is a natural, low-cost extension of code that
already exists, not a production API: no auth, no database, no deployment
config. Run with:

    uv run uvicorn api:app --reload

EVERY REQUEST IS INDEPENDENT
    One runtime is built at startup and shared by all requests (building it
    launches MCP servers and loads models, far too slow per request). The
    runtime keeps conversation memory, and the previous version never reset
    it, so one caller's questions and answers appeared in the next caller's
    prompt - a privacy leak between users, not just noise. Each request now
    starts from an empty session, and long-term memory is disabled: it is a
    single local file with no notion of which user a memory belongs to.

    The reset and the memory read happen before the run's first `await`, so
    concurrent requests on the event loop cannot see each other's turns.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from agent_runtime.runtime_factory import build_single_agent_runtime

_runtime_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with build_single_agent_runtime(with_memory=False) as runtime:
        _runtime_holder["runtime"] = runtime
        yield
    _runtime_holder.clear()


app = FastAPI(
    title="Agentic RAG Runtime",
    description="A from-scratch agentic RAG runtime with MCP tool integration - see README.md.",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    # False means the answer was accepted on the last step WITHOUT passing
    # citation/number verification; clients must not present it as verified.
    answer_verified: bool | None
    citations: list[str]
    termination_reason: str
    step_count: int


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    runtime = _runtime_holder["runtime"]
    runtime.reset_session()
    state = await runtime.run(request.question)
    return QueryResponse(
        answer=state.final_answer or "",
        answer_verified=state.answer_verified,
        citations=state.citations,
        termination_reason=state.termination_reason.value if state.termination_reason else "unknown",
        step_count=len(state.steps),
    )
