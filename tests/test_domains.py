"""The domain loader must refuse manifests that disagree with their code -
that disagreement (a documented 20% rollout threshold vs an enforced 50%, a
documented rule with no implementation) is exactly what it was built to stop."""

import shutil

import pytest

import agent_runtime.domains as domains_module
from agent_runtime.domains import DomainConfigError, list_domains, load_domain


def test_all_shipped_domains_are_consistent():
    for name in list_domains():
        spec = load_domain(name)
        engine = spec.create_policy_engine()
        assert dict(engine.thresholds) == dict(spec.thresholds)
        assert spec.server_script.is_file() and spec.benchmark_path.is_file()
        assert spec.server_modules == {name: spec.server_script}
        assert engine.guidance and engine.guidance == spec.decision_guide


def test_decision_guidance_reaches_the_prompt_and_only_its_own_domain():
    # Regression: the generic prompt listed every domain's action names, and a
    # real model copied "rolling_restart" into a product-analytics rollout.
    from agent_runtime.agent.loop import AgentRuntime
    from agent_runtime.agent.tools.registry import ToolRegistry

    from .fakes import FakeLLMClient

    runtime = AgentRuntime(llm=FakeLLMClient(), tools=ToolRegistry(), policy_engine=load_domain("product_analytics").create_policy_engine())
    prompt = runtime._system_prompt(max_steps=5)
    assert "rollout_variant_b" in prompt and "traffic_percentage" in prompt
    assert "rolling_restart" not in prompt and "account_id" not in prompt
    # The real limits are stated, so the model does not invent its own.
    assert "max_autonomous_traffic_pct=50" in prompt and "significance_alpha=0.05" in prompt


@pytest.fixture
def domain_copy(tmp_path, monkeypatch):
    """A private copy of the credit_risk domain that tests may corrupt."""
    shutil.copytree(domains_module.DOMAINS_DIR / "credit_risk", tmp_path / "credit_risk", ignore=shutil.ignore_patterns("data"))
    monkeypatch.setattr(domains_module, "DOMAINS_DIR", tmp_path)
    return tmp_path / "credit_risk" / "domain_manifest.yaml"


def _edit(manifest, old: str, new: str) -> None:
    text = manifest.read_text(encoding="utf-8")
    assert old in text
    manifest.write_text(text.replace(old, new), encoding="utf-8")


def test_missing_threshold_is_rejected(domain_copy):
    _edit(domain_copy, "  min_credit_score: 650\n", "")
    with pytest.raises(DomainConfigError, match="missing thresholds \\['min_credit_score'\\]"):
        load_domain("credit_risk")


def test_misspelt_threshold_is_rejected(domain_copy):
    _edit(domain_copy, "min_credit_score:", "min_credit_scor:")
    with pytest.raises(DomainConfigError, match="unknown thresholds \\['min_credit_scor'\\]"):
        load_domain("credit_risk")


def test_non_numeric_threshold_is_rejected(domain_copy):
    _edit(domain_copy, "min_credit_score: 650", "min_credit_score: true")
    with pytest.raises(DomainConfigError, match="must be a number"):
        load_domain("credit_risk")


def test_documented_rule_without_implementation_is_rejected(domain_copy):
    _edit(domain_copy, "  - name: dti_ceiling\n", "  - name: safe_automated_approval\n")
    spec = load_domain("credit_risk")
    with pytest.raises(DomainConfigError, match="implements"):
        spec.create_policy_engine()


def test_unknown_domain_is_rejected():
    with pytest.raises(DomainConfigError, match="Unknown domain"):
        load_domain("no_such_domain")
