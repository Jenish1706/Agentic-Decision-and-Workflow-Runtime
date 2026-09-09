"""Thin FastAPI wrapper over the single-agent runtime.

Not the primary interface - see scripts/ for that. This exists because the
project studies FastAPI alongside RAG/agents, so exposing the runtime as a
couple of HTTP endpoints is a natural, low-cost extension of code that
already exists, not a production API: no auth, no database, no deployment
config. Run with:

    uv run uvicorn api:app --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel

from agent_runtime.runtime_factory import build_single_agent_runtime

_runtime_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with build_single_agent_runtime() as runtime:
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
    citations: list[str]
    termination_reason: str
    step_count: int


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    runtime = _runtime_holder["runtime"]
    state = await runtime.run(request.question)
    return QueryResponse(
        answer=state.final_answer or "",
        citations=state.citations,
        termination_reason=state.termination_reason.value if state.termination_reason else "unknown",
        step_count=len(state.steps),
    )
