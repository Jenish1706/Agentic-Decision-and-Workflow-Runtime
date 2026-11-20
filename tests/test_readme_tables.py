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
    fake_guide_dir = tmp_path / "docs" / "guide"
    fake_guide_dir.mkdir(parents=True)
    fake_guide09 = fake_guide_dir / "09_RESULTS_AND_LIMITATIONS.md"
    fake_guide00 = fake_guide_dir / "00_START_HERE.md"

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
    fake_guide09.write_text(
        "Header\n<!-- BEGIN_GUIDE09_VERIFIER -->\n<!-- END_GUIDE09_VERIFIER -->\n"
        "<!-- BEGIN_GUIDE09_CITATION -->\n<!-- END_GUIDE09_CITATION -->\n",
        encoding="utf-8",
    )
    fake_guide00.write_text(
        "Header\n<!-- BEGIN_GUIDE00_RESULTS -->\n<!-- END_GUIDE00_RESULTS -->\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(render_readme, "ROOT", tmp_path)
    # With mismatched/empty tables, check=True must return False
    assert render_readme.render_all(check=True) is False


def test_guide09_rates_match_measurement_json():
    """Fail if guide 09 states a rate that the measurement does not produce."""
    from scripts import measure_verifier

    guide_path = ROOT / "docs" / "guide" / "09_RESULTS_AND_LIMITATIONS.md"
    content = guide_path.read_text(encoding="utf-8")
    metrics = measure_verifier.measure()

    # Verify that the accepted percentages rendered in guide 09 match exact measurement output
    plaus_pct = f"{metrics['plausible_pct']['current_pct']:.1%}"
    assert f"{plaus_pct} accepted" in content, f"Expected {plaus_pct} accepted in guide 09"

    gaps_pct = f"{metrics['pp_gaps']['current_pct']:.1%}"
    assert f"{gaps_pct} accepted" in content, f"Expected {gaps_pct} accepted in guide 09"

    rels_pct = f"{metrics['relative_changes']['current_pct']:.1%}"
    assert f"{rels_pct} accepted" in content, f"Expected {rels_pct} accepted in guide 09"

    # Verify that false/hallucinated rejection figures are NOT in guide 09
    for bad_num in ["96.6%", "99.6%", "98.2%", "2/2"]:
        assert bad_num not in content, f"Found hallucinated figure {bad_num} in guide 09"
