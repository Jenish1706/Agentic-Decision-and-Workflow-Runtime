"""MCP server for the Product Analytics & Experimentation domain.

Tools
    execute_sql        read-only SQL over the checkout / experiment warehouse
    describe_tables    schema discovery
    test_significance  two-proportion z-test for an A/B experiment

The runtime launches this file as an MCP subprocess via
`agent_runtime.domains.load_domain("product_analytics").server_modules`.

Every tool declares a `category` in its MCP metadata; the agent loop uses it
for budget accounting and evidence provenance (see agent/tools/base.py).

WHY THE SIGNIFICANCE TEST RETURNS `treatment_better`
    A two-sided test answers "are the rates different?", not "is B better?".
    A significantly WORSE treatment also has p < alpha. Rollout governance
    (policies.py) therefore checks the p-value against its own alpha AND the
    sign of the difference; `treatment_better` states the same thing for the
    model so it does not have to infer direction from `significant`.
"""

import math
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel
from scipy import stats

from agent_runtime.mcp_layer.duckdb_tools import (
    SchemaDescriptionResult,
    SqlQueryResult,
    describe_database,
    execute_read_only,
    read_only_connection,
)

DB_PATH = Path(__file__).resolve().parent / "data" / "product_analytics.duckdb"

server = MCPServer("product-analytics")


@server.tool(meta={"category": "sql"})
def execute_sql(query: str, max_rows: int = 50) -> SqlQueryResult:
    """Execute one read-only SQL query against the product analytics database.

    Tables:
    - daily_checkout_metrics (metric_date, period, device, traffic_source, checkouts_started,
      purchases_completed, conversion_rate)
    - funnel_dropoff (funnel_step, step_order, mobile_dropoff_rate, desktop_dropoff_rate)
    - experiments (experiment_id, name, target_segment, hypothesis, status, start_date, end_date)
    - experiment_results (experiment_id, variant, users_assigned, conversions, conversion_rate)

    Rates are stored as fractions (0.0293 means 2.93%). At most 200 rows are returned; check
    `truncated` before treating a result as complete.
    """
    return execute_read_only(DB_PATH, query, max_rows)


@server.tool(meta={"category": "record"})
def describe_tables() -> SchemaDescriptionResult:
    """List the tables, columns and column types in the product analytics database."""
    return describe_database(DB_PATH)


class SignificanceResult(BaseModel):
    experiment_id: str
    metric: str
    control_variant: str
    treatment_variant: str
    test_name: str
    alternative: str
    alpha: float
    statistic: float
    p_value: float
    sample_size_a: int
    sample_size_b: int
    conversions_a: int
    conversions_b: int
    rate_a: float
    rate_b: float
    difference_pp: float
    relative_lift_pct: float | None
    confidence_interval_95: list[float]
    significant: bool
    treatment_better: bool


_CONTROL_LABELS = {"a", "control"}
_TREATMENT_LABELS = {"b", "treatment"}


@server.tool(meta={"category": "statistics"})
def test_significance(experiment_id: str, alpha: float = 0.05) -> SignificanceResult:
    """Two-proportion z-test comparing an experiment's treatment (variant B) against control (A).

    Returns the z statistic, a TWO-SIDED p-value, the signed difference in percentage points
    (treatment minus control), relative lift, and a confidence interval for the difference.
    `significant` means p < alpha in EITHER direction; `treatment_better` is true only when the
    difference is significant AND positive. A rollout should rely on `treatment_better`.

    Args:
        experiment_id: Experiment to test, e.g. "EXP-204". Required - there is no default.
        alpha: Significance level, between 0 and 0.5.
    """
    # experiment_id used to default to "EXP-204", so an agent that forgot the
    # argument silently received a different experiment's result.
    if not 0.0 < alpha <= 0.5:
        raise ValueError(f"alpha must be in (0, 0.5], got {alpha}.")

    with read_only_connection(DB_PATH) as connection:
        rows = connection.execute(
            "SELECT variant, users_assigned, conversions FROM experiment_results WHERE experiment_id = ?",
            [experiment_id],
        ).fetchall()

    # The old version accepted "at least 2" variants and compared the first
    # two rows, so a third variant was silently ignored.
    if len(rows) != 2:
        raise ValueError(f"Experiment '{experiment_id}' must have exactly 2 variants; found {len(rows)}.")
    by_label = {str(variant).strip().lower(): (str(variant), users, conversions) for variant, users, conversions in rows}
    control = next((by_label[label] for label in by_label if label in _CONTROL_LABELS), None)
    treatment = next((by_label[label] for label in by_label if label in _TREATMENT_LABELS), None)
    if control is None or treatment is None:
        # Previously the first row in alphabetical order was silently assumed
        # to be control, which is only right by coincidence of naming.
        raise ValueError(f"Cannot identify control (A) and treatment (B) variants among {sorted(by_label)}.")

    control_label, n_control, conv_control = control
    treatment_label, n_treatment, conv_treatment = treatment
    for label, n, conversions in (control, treatment):
        if n <= 0 or not 0 <= conversions <= n:
            raise ValueError(f"Variant {label} has invalid counts (users={n}, conversions={conversions}).")

    # Rates are computed from the counts rather than read from the stored
    # conversion_rate column (as the old version did), so every number
    # returned is consistent with the test statistic; the column is rounded
    # and could disagree.
    p_control = conv_control / n_control
    p_treatment = conv_treatment / n_treatment

    # Pooled standard error under H0 (the two rates are equal) for the test...
    pooled = (conv_control + conv_treatment) / (n_control + n_treatment)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    z_statistic = (p_treatment - p_control) / se_pooled if se_pooled > 0 else 0.0
    p_value = 2.0 * stats.norm.sf(abs(z_statistic))

    # ...and the unpooled standard error for the confidence interval (the
    # standard Wald construction). The interval is always 95%, as its field
    # name says, whatever `alpha` the caller chose for the test.
    se_unpooled = math.sqrt(p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_treatment)
    difference_pp = (p_treatment - p_control) * 100.0
    margin_pp = stats.norm.ppf(0.975) * se_unpooled * 100.0

    significant = bool(p_value < alpha)
    return SignificanceResult(
        experiment_id=experiment_id,
        metric="conversion_rate",
        control_variant=control_label,
        treatment_variant=treatment_label,
        test_name="two_proportion_z_test",
        alternative="two-sided",
        alpha=alpha,
        statistic=round(z_statistic, 4),
        p_value=round(float(p_value), 5),
        sample_size_a=n_control,
        sample_size_b=n_treatment,
        conversions_a=conv_control,
        conversions_b=conv_treatment,
        rate_a=round(p_control, 4),
        rate_b=round(p_treatment, 4),
        difference_pp=round(difference_pp, 3),
        relative_lift_pct=round((p_treatment - p_control) / p_control * 100.0, 2) if p_control > 0 else None,
        confidence_interval_95=[round(difference_pp - margin_pp, 3), round(difference_pp + margin_pp, 3)],
        significant=significant,
        treatment_better=significant and difference_pp > 0,
    )


if __name__ == "__main__":
    server.run()
