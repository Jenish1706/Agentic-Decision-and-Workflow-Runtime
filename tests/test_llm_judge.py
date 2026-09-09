"""Covers the JudgeVerdict schema's tolerance for how small models actually
answer, as opposed to how the schema asks them to. Every case here was
observed for real running this project against Ollama, or is the direct
boundary of one."""

from agent_runtime.agent.llm_judge import JudgeVerdict


def test_confidence_in_range_is_untouched():
    assert JudgeVerdict(grounded=True, confidence=0.85).confidence == 0.85
    assert JudgeVerdict(grounded=True, confidence=1.0).confidence == 1.0
    assert JudgeVerdict(grounded=True, confidence=0.0).confidence == 0.0


def test_percentage_style_confidence_is_rescaled():
    # The real failure: model answered `70`, pydantic rejected it, and the
    # resulting LLMError aborted an entire multi-agent run.
    assert JudgeVerdict(grounded=True, confidence=70).confidence == 0.7
    assert JudgeVerdict(grounded=False, confidence=100).confidence == 1.0


def test_out_of_range_confidence_is_clamped_not_rejected():
    assert JudgeVerdict(grounded=True, confidence=150).confidence == 1.0
    assert JudgeVerdict(grounded=True, confidence=-5).confidence == 0.0


def test_issues_defaults_to_empty_list():
    assert JudgeVerdict(grounded=True, confidence=0.9).issues == []
