"""Entry point for the Retrieval Budget & Search Efficiency experiment:
`uv run python scripts/experiments/run_budget_experiment.py`.
See agent_runtime/evaluation/budget_experiment.py for the hypothesis and
implementation."""

import asyncio

from agent_runtime.evaluation.budget_experiment import main

if __name__ == "__main__":
    asyncio.run(main())
