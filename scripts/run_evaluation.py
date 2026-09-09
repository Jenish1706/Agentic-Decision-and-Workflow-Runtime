"""Entry point for the evaluation harness: `uv run python scripts/run_evaluation.py`.
See agent_runtime/evaluation/run_eval.py for the actual implementation."""

import asyncio

from agent_runtime.evaluation.run_eval import main

if __name__ == "__main__":
    asyncio.run(main())
