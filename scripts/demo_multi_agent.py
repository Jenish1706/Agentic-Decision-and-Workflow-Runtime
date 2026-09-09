"""Runs the same representative questions through the multi-agent
supervisor path (Supervisor -> Retriever/Researcher -> Verifier ->
Synthesizer) and prints the worker-level trace for each."""

import asyncio

from agent_runtime.runtime_factory import build_multi_agent_supervisor
from agent_runtime.tracing import print_multi_agent_trace

QUESTIONS = [
    "What is the default per-tenant rate limit for Aurora event ingestion?",
    "If Ferry's July 2026 storage quota issue happened again, which team owns the fix, and what does policy require before deleting anything older than 30 days?",
    "What was Nimbus's total revenue last quarter?",
]


async def main() -> None:
    async with build_multi_agent_supervisor() as supervisor:
        for question in QUESTIONS:
            result = await supervisor.run(question)
            print_multi_agent_trace(result)
            print()


if __name__ == "__main__":
    asyncio.run(main())
