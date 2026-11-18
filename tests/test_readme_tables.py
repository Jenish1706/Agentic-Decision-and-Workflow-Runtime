"""Tests that documentation tables in README.md and EVALUATION.md are exactly
synchronized with benchmark and measurement artifacts.

Guarantees that hand-edited or hallucinated figures cannot survive.
"""

from pathlib import Path
import pytest
from scripts import render_readme

ROOT = Path(__file__).resolve().parent.parent


def test_readme_and_evaluation_tables_match_measurements():
    """Assert that documentation tables match measurement artifacts."""
    assert render_readme.render_all(check=True) is True


def test_render_all_detects_tampered_tables(monkeypatch, tmp_path):
    """Test that any discrepancy between measurements and markdown is caught."""
    fake_readme = tmp_path / "README.md"
    fake_eval = tmp_path / "EVALUATION.md"

    fake_readme.write_text(
        "Header\n<!-- BEGIN_VERIFIER_TABLE -->\n| old | 999 |\n<!-- END_VERIFIER_TABLE -->\n"
        "<!-- BEGIN_CITATION_TABLE -->\n<!-- END_CITATION_TABLE -->\n",
        encoding="utf-8",
    )
    fake_eval.write_text(
        "Header\n<!-- BEGIN_VERIFIER_TABLE -->\n<!-- END_VERIFIER_TABLE -->\n"
        "<!-- BEGIN_CITATION_TABLE -->\n<!-- END_CITATION_TABLE -->\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(render_readme, "ROOT", tmp_path)
    # With mismatched/empty tables, check=True must return False
    assert render_readme.render_all(check=True) is False
