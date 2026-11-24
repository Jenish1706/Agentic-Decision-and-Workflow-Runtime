"""Tests that documentation tables in README.md and EVALUATION.md are exactly
synchronized with benchmark and measurement artifacts.

Guarantees that hand-edited or hallucinated figures cannot survive.
"""

from pathlib import Path
import pytest
from scripts.tooling import render_readme

ROOT = Path(__file__).resolve().parent.parent


def test_readme_and_evaluation_tables_match_measurements():
    """Assert that documentation tables match measurement artifacts."""
    assert render_readme.render_all(check=True) is True


def test_render_all_detects_tampered_tables(monkeypatch, tmp_path):
    """Test that any discrepancy between measurements and markdown is caught."""
    fake_readme = tmp_path / "README.md"
    fake_eval = tmp_path / "docs" / "EVALUATION.md"
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
    from scripts.measurements import measure_verifier

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


def test_live_nli_citation_matches_committed_measurement():
    """Assert live NLI citation measurement matches committed artifact when model is available.
    Skips with a clear reason if the NLI model cannot be loaded offline."""
    from scripts.measurements import measure_citations

    live_metrics = measure_citations.measure(save_artifact=False)
    if live_metrics.get("backend_used") != "nli":
        pytest.skip("NLI cross-encoder model not cached or network blocked; skipping live NLI comparison")

    committed = measure_citations.load_committed_citation_measurements()
    if committed is None:
        pytest.skip("Committed citation measurement artifact not found")

    assert live_metrics["benchmark_nli"]["precision"] == committed["benchmark_nli"]["precision"]
    assert live_metrics["benchmark_nli"]["recall"] == committed["benchmark_nli"]["recall"]
    assert live_metrics["heldout_nli"]["precision"] == committed["heldout_nli"]["precision"]
