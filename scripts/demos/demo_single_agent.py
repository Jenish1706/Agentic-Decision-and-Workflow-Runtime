"""Runs a handful of representative questions through the single-agent
runtime and prints a full execution trace for each - a pure document
question, a calculator question, an MCP records question, a multi-hop
question, and an out-of-corpus question that the agent should refuse rather
than hallucinate.

Requires Ollama running locally with the configured model pulled
(see .env.example / README.md).
"""

import asyncio

from agent_runtime.runtime_factory import build_single_agent_runtime
from agent_runtime.tracing import print_single_agent_trace

QUESTIONS = [
    "What is the default per-tenant rate limit for Aurora event ingestion?",
    "Beacon scrapes metrics every 30 seconds. How many scrapes happen in one hour?",
    "Which team owns the Ferry service?",
    "Who should be paged if the team that owns Cascade needs escalation, and how long until it escalates to secondary?",
    "What was Nimbus's total revenue last quarter?",
]


async def main() -> None:
    async with build_single_agent_runtime() as runtime:
        for question in QUESTIONS:
            # The questions are unrelated; without a reset each prompt would
            # carry the previous questions and answers as "conversation".
            runtime.reset_session()
            state = await runtime.run(question)
            print_single_agent_trace(state)
            print()


if __name__ == "__main__":
    asyncio.run(main())
