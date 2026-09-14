"""Wires the sub-systems (LLM, tools, MCP servers, memory, policy) into
ready-to-use runtimes.

Centralised so demo scripts, the FastAPI wrapper and every evaluation harness
build runtimes identically instead of each duplicating setup (and drifting).
"""

from contextlib import asynccontextmanager
from pathlib import Path

from .agent.budget import ActionBudget
from .agent.loop import AgentRuntime
from .agent.mcp_retriever import MCPRetriever
from .agent.memory.long_term import LongTermMemory
from .agent.memory.short_term import ShortTermMemory
from .agent.policy import PolicyEngine
from .agent.tools.calculator import CalculatorTool
from .agent.tools.mcp_tool import register_mcp_tools
from .agent.tools.registry import ToolRegistry, build_restricted_registry
from .config import settings
from .llm.base import LLMClient
from .llm.ollama_client import OllamaClient
from .mcp_layer.client import MCPToolClient
from .multi_agent.supervisor import Supervisor
from .multi_agent.synthesizer import Synthesizer
from .multi_agent.workers import ResearcherWorker, RetrieverWorker
from .rag.embeddings import Embedder
from .rag.ingest import build_rag_pipeline

# search_documents is never exposed as a generic callable tool: the agent's
# dedicated "retrieve" action already reaches it via agent/mcp_retriever.py,
# and that is the one path that feeds citation tracking and verification.
# Exposing it twice recreated a real verification-bypass bug found while
# testing this project (see the mcp_retriever.py docstring).
_TOOLS_EXCLUDED_FROM_AGENT_REGISTRY = {"search_documents"}


@asynccontextmanager
async def build_single_agent_runtime(
    *,
    with_memory: bool = True,
    max_retrievals: int | None = None,
    policy_engine: PolicyEngine | None = None,
    budget: ActionBudget | None = None,
    llm: LLMClient | None = None,
    server_modules: dict[str, str | Path] | None = None,
    include_rag: bool = True,
):
    """Build an AgentRuntime connected to MCP servers.

    `server_modules=None` connects the built-in knowledge base (knowledge +
    records). A business domain passes its own servers instead - see
    `agent_runtime.domains.load_domain(...).server_modules`.

    `include_rag` requires a connected `knowledge` server, because retrieval
    goes through it.
    """
    active_llm = llm or OllamaClient(model=settings.ollama_model, host=settings.ollama_host)

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    async with MCPToolClient(server_modules=server_modules) as mcp_client:
        await register_mcp_tools(mcp_client, registry)
        agent_registry = build_restricted_registry(registry, _TOOLS_EXCLUDED_FROM_AGENT_REGISTRY)

        if include_rag and "knowledge" not in mcp_client.server_names:
            raise ValueError("include_rag=True needs a 'knowledge' MCP server; pass include_rag=False for this domain.")

        long_term = (
            LongTermMemory(embedder=Embedder(settings.embedding_model), path=settings.long_term_memory_path)
            if with_memory
            else None
        )
        runtime = AgentRuntime(
            llm=active_llm,
            tools=agent_registry,
            rag=MCPRetriever(mcp_client) if include_rag else None,
            short_term=ShortTermMemory(),
            long_term=long_term,
            max_steps=settings.max_agent_steps,
            max_decision_retries=settings.max_decision_retries,
            max_retrievals=max_retrievals,
            policy_engine=policy_engine,
            budget=budget,
        )
        yield runtime


@asynccontextmanager
async def build_multi_agent_supervisor(*, llm: LLMClient | None = None):
    """Build the Supervisor/worker runtime over the built-in knowledge base."""
    active_llm = llm or OllamaClient(model=settings.ollama_model, host=settings.ollama_host)
    rag = build_rag_pipeline(embedder=Embedder(settings.embedding_model))

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    async with MCPToolClient() as mcp_client:
        await register_mcp_tools(mcp_client, registry)
        researcher_registry = build_restricted_registry(registry, _TOOLS_EXCLUDED_FROM_AGENT_REGISTRY)
        yield Supervisor(
            llm=active_llm,
            retriever=RetrieverWorker(rag=rag),
            researcher=ResearcherWorker(llm=active_llm, tools=researcher_registry),
            synthesizer=Synthesizer(llm=active_llm),
        )
