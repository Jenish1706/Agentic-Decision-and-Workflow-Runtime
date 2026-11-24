"""Entry point for the Product Analytics Action-Space & Budgeting Experiment (Phase 1.5).

Run with:
    uv run python scripts/experiments/run_product_analytics_experiment.py

Tests whether restricting an agent's action space improves business-task efficiency
without reducing decision quality across Unconstrained, Fixed, and Adaptive regimes.
"""

import asyncio

from agent_runtime.evaluation.product_analytics_experiment import main

if __name__ == "__main__":
    asyncio.run(main())
