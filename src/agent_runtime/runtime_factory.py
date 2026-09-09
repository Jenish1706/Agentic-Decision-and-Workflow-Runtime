"""Wires the sub-systems (LLM, RAG, tools, MCP, memory) into ready-to-use
runtimes. Centralized so demo scripts, the FastAPI wrapper, and the
evaluation harness all build the same way instead of duplicating setup.
"""

from contextlib import asynccontextmanager

from .agent.loop import AgentRuntime
from .agent.mcp_retriever import MCPRetriever
from .agent.memory.long_term import LongTermMemory
from .agent.memory.short_term import ShortTermMemory
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

# search_documents is deliberately never exposed as a generic callable tool:
# the agent's dedicated "retrieve" action already reaches it via
# agent/mcp_retriever.py, and that is also the one path that feeds results
# into citation tracking / verification. Exposing it a second time as a
# regular tool would recreate a real bug found while testing this project -
# see the docstring in agent/mcp_retriever.py.
_TOOLS_EXCLUDED_FROM_AGENT_REGISTRY = {"search_documents"}


@asynccontextmanager
async def build_single_agent_runtime(
    with_memory: bool = True,
    max_retrievals: int | None = None,
    policy_engine=None,
    budget_controller=None,
    llm: LLMClient | None = None,
    server_modules: dict[str, str] | None = None,
    include_rag: bool = True,
):
    active_llm = llm or OllamaClient(model=settings.ollama_model, host=settings.ollama_host)
    embedder = Embedder(settings.embedding_model) if with_memory else None

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    async with MCPToolClient(server_modules=server_modules) as mcp_client:
        await register_mcp_tools(mcp_client, registry)
        agent_registry = build_restricted_registry(registry, _TOOLS_EXCLUDED_FROM_AGENT_REGISTRY)

        long_term = (
            LongTermMemory(embedder=embedder, path=settings.long_term_memory_path)
            if (with_memory and embedder)
            else None
        )

        retriever = MCPRetriever(mcp_client) if include_rag else None

        runtime = AgentRuntime(
            llm=active_llm,
            tools=agent_registry,
            rag=retriever,
            short_term=ShortTermMemory(),
            long_term=long_term,
            max_steps=settings.max_agent_steps,
            max_decision_retries=settings.max_decision_retries,
            max_retrievals=max_retrievals,
            policy_engine=policy_engine,
            budget_controller=budget_controller,
        )
        yield runtime


@asynccontextmanager
async def build_multi_agent_supervisor():
    llm = OllamaClient(model=settings.ollama_model, host=settings.ollama_host)
    embedder = Embedder(settings.embedding_model)
    rag = build_rag_pipeline(embedder=embedder)

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    async with MCPToolClient() as mcp_client:
        await register_mcp_tools(mcp_client, registry)

        researcher_registry = build_restricted_registry(registry, _TOOLS_EXCLUDED_FROM_AGENT_REGISTRY)
        supervisor = Supervisor(
            llm=llm,
            retriever=RetrieverWorker(rag=rag),
            researcher=ResearcherWorker(llm=llm, tools=researcher_registry),
            synthesizer=Synthesizer(llm=llm),
        )
        yield supervisor
