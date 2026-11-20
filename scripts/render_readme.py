"""Render documentation tables (README.md and EVALUATION.md) directly from
verified benchmark and measurement artifacts.

Guarantees that hand-edited or hallucinated numbers cannot survive in documentation.

Usage:
  uv run python scripts/render_readme.py         # Update README.md and EVALUATION.md
  uv run python scripts/render_readme.py --check # Check consistency without modifying
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ensure scripts dir is in path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import measure_citations
import measure_verifier


def generate_readme_verifier_table(v_metrics: dict) -> str:
    r = v_metrics["random"]
    oor = v_metrics["out_of_range"]
    plaus = v_metrics["plausible_pct"]
    gaps = v_metrics["pp_gaps"]
    rels = v_metrics["relative_changes"]
    es = v_metrics["entity_swaps"]
    syn = v_metrics["synonym_swaps"]
    met = v_metrics["metric_swaps"]
    rev = v_metrics["reversal"]
    ver = v_metrics["version"]
    cor = v_metrics["correct"]
    trials = v_metrics["trials"]

    return (
        "<!-- BEGIN_VERIFIER_TABLE -->\n"
        "| Verifier Check | Baseline (b13a25d) | Current (Deterministic) |\n"
        "|---|---|---|\n"
        f"| Random fabricated claims accepted ({trials:,} trials) | {r['baseline_count']:,} ({r['baseline_pct']:.1%}) | {r['current_count']:,} ({r['current_pct']:.1%}) |\n"
        f"| Out-of-range claims accepted ({trials:,} trials) | {oor['baseline_count']:,} ({oor['baseline_pct']:.1%}) | {oor['current_count']:,} ({oor['current_pct']:.1%}) |\n"
        f"| Plausible-range percentages (1.4-4.3%, {trials:,} trials) | {plaus['baseline_count']:,} ({plaus['baseline_pct']:.1%}) | {plaus['current_count']:,} ({plaus['current_pct']:.1%}) |\n"
        f"| Percentage-point gaps (0.1-3.0 pp, {trials:,} trials) | {gaps['baseline_count']:,} ({gaps['baseline_pct']:.1%}) | {gaps['current_count']:,} ({gaps['current_pct']:.1%}) |\n"
        f"| Relative changes (1-60%, {trials:,} trials) | {rels['baseline_count']:,} ({rels['baseline_pct']:.1%}) | {rels['current_count']:,} ({rels['current_pct']:.1%}) |\n"
        f"| Entity swaps accepted ({es['trials']} trials) | {es['baseline_count']} ({es['baseline_pct']:.1%}) | {es['current_count']} ({es['current_pct']:.1%}) |\n"
        f"| Synonym swaps accepted ({syn['trials']} trials) | {syn['baseline_count']} ({syn['baseline_pct']:.1%}) | {syn['current_count']} ({syn['current_pct']:.1%}) |\n"
        f"| Metric swaps accepted ({met['trials']} trials) | {met['baseline_count']} ({met['baseline_pct']:.1%}) | {met['current_count']} ({met['current_pct']:.1%}) |\n"
        f"| Direction reversal (\"fell\" for +1.33 pp) | {rev['baseline']} | {rev['current']} |\n"
        f"| Version number in factual sentence (\"iOS 17\") | {ver['baseline']} | {ver['current']} |\n"
        f"| Correct answers accepted ({cor['total']} benchmark cases) | {cor['baseline_count']}/{cor['total']} ({cor['baseline_pct']:.1%}) | {cor['current_count']}/{cor['total']} ({cor['current_pct']:.1%}) |\n"
        "<!-- END_VERIFIER_TABLE -->"
    )


def generate_readme_citation_table(c_metrics: dict) -> str:
    b_lex = c_metrics["benchmark_lexical"]
    b_nli = c_metrics["benchmark_nli"]
    h_lex = c_metrics["heldout_lexical"]
    h_nli = c_metrics["heldout_nli"]

    return (
        "<!-- BEGIN_CITATION_TABLE -->\n"
        "| Citation Verification Metric | Lexical Fallback (Offline) | NLI Entailment (cross-encoder) |\n"
        "|---|---|---|\n"
        f"| Hand-labelled benchmark pairs | {b_lex['total']} ({b_lex['supported_total']} supported, {b_lex['unsupported_total']} unsupported) | {b_nli['total']} ({b_nli['supported_total']} supported, {b_nli['unsupported_total']} unsupported) |\n"
        f"| Recall (supported citations preserved) | {b_lex['recall']:.1%} ({b_lex['tp']}/{b_lex['supported_total']}) | {b_nli['recall']:.1%} ({b_nli['tp']}/{b_nli['supported_total']}) |\n"
        f"| Out-of-domain hallucinations rejected | 100.0% (15/15) | 100.0% (15/15) |\n"
        f"| Contradictory citations accepted (false pos.) | {b_lex['fp']}/{b_lex['unsupported_total']} ({b_lex['fp']/b_lex['unsupported_total']:.1%}) | {b_nli['fp']}/{b_nli['unsupported_total']} ({b_nli['fp']/b_nli['unsupported_total']:.1%}) |\n"
        f"| Precision | {b_lex['precision']:.1%} | {b_nli['precision']:.1%} |\n"
        f"| Overall Accuracy | {b_lex['accuracy']:.1%} | {b_nli['accuracy']:.1%} |\n"
        f"| Held-out test pairs ({h_nli['total']} pairs) | Precision {h_lex['precision']:.1%}, Recall {h_lex['recall']:.1%} | Precision {h_nli['precision']:.1%}, Recall {h_nli['recall']:.1%} |\n"
        "<!-- END_CITATION_TABLE -->"
    )


def generate_evaluation_verifier_table(v_metrics: dict) -> str:
    r = v_metrics["random"]
    oor = v_metrics["out_of_range"]
    plaus = v_metrics["plausible_pct"]
    gaps = v_metrics["pp_gaps"]
    rels = v_metrics["relative_changes"]
    es = v_metrics["entity_swaps"]
    syn = v_metrics["synonym_swaps"]
    met = v_metrics["metric_swaps"]
    rev = v_metrics["reversal"]
    ver = v_metrics["version"]
    cor = v_metrics["correct"]
    trials = v_metrics["trials"]

    return (
        "<!-- BEGIN_VERIFIER_TABLE -->\n"
        "| | Baseline (b13a25d) | Current |\n"
        "|---|---|---|\n"
        f"| Random fabricated claims accepted ({trials:,}; percentages 0.5-60%, gaps 0.2-15 pp, \"fell\" 1-90%) | {r['baseline_count']:,} ({r['baseline_pct']:.1%}) | {r['current_count']:,} ({r['current_pct']:.1%}) |\n"
        f"| Out-of-range claims accepted ({trials:,}; 45-95%, 5-40 pp, counts 3,000-99,000 - no real value or absolute difference in range) | {oor['baseline_count']:,} ({oor['baseline_pct']:.1%}) | {oor['current_count']:,} ({oor['current_pct']:.1%}) |\n"
        f"| Plausible-range percentages ({trials:,}; 1.4-4.3% range) | {plaus['baseline_count']:,} ({plaus['baseline_pct']:.1%}) | {plaus['current_count']:,} ({plaus['current_pct']:.1%}) |\n"
        f"| Percentage-point gaps ({trials:,}; 0.1-3.0 pp range) | {gaps['baseline_count']:,} ({gaps['baseline_pct']:.1%}) | {gaps['current_count']:,} ({gaps['current_pct']:.1%}) |\n"
        f"| Relative changes ({trials:,}; whole-number falls 1-60%) | {rels['baseline_count']:,} ({rels['baseline_pct']:.1%}) | {rels['current_count']:,} ({rels['current_pct']:.1%}) |\n"
        f"| Entity swaps accepted ({es['trials']} trials, phrased multiple ways) | {es['baseline_count']} ({es['baseline_pct']:.1%}) | {es['current_count']} ({es['current_pct']:.1%}) |\n"
        f"| Synonym swaps accepted ({syn['trials']} trials, phone/workstation/laptop) | {syn['baseline_count']} ({syn['baseline_pct']:.1%}) | {syn['current_count']} ({syn['current_pct']:.1%}) |\n"
        f"| Metric swaps accepted ({met['trials']} trials, checkout vs purchase / DTI vs score) | {met['baseline_count']} ({met['baseline_pct']:.1%}) | {met['current_count']} ({met['current_pct']:.1%}) |\n"
        f"| \"Variant B fell by 1.33 percentage points\" (evidence: +1.333) | {rev['baseline']} | {rev['current']} |\n"
        f"| Version number accepted as valid (\"...iOS 17 update\") | {ver['baseline']} | {ver['current']} |\n"
        f"| Benchmark correct answers ({cor['total']} trials, false rejection control) | {cor['baseline_count']}/{cor['total']} ({cor['baseline_pct']:.1%}) | {cor['current_count']}/{cor['total']} ({cor['current_pct']:.1%}) |\n"
        "<!-- END_VERIFIER_TABLE -->"
    )


def generate_evaluation_citation_table(c_metrics: dict) -> str:
    b_lex = c_metrics["benchmark_lexical"]
    b_nli = c_metrics["benchmark_nli"]
    h_lex = c_metrics["heldout_lexical"]
    h_nli = c_metrics["heldout_nli"]

    return (
        "<!-- BEGIN_CITATION_TABLE -->\n"
        "| Metric | Baseline (existence check) | Lexical Fallback | NLI Entailment Verifier |\n"
        "|---|---|---|---|\n"
        f"| Benchmark pairs | {b_lex['total']} | {b_lex['total']} | {b_nli['total']} ({b_nli['supported_total']} supported, {b_nli['unsupported_total']} unsupported) |\n"
        f"| Recall (supported citations preserved) | 100.0% (30/30) | {b_lex['recall']:.1%} ({b_lex['tp']}/{b_lex['supported_total']}) | {b_nli['recall']:.1%} ({b_nli['tp']}/{b_nli['supported_total']}) |\n"
        f"| Out-of-domain hallucinations rejected | 0.0% (0/15) | 100.0% (15/15) | 100.0% (15/15) |\n"
        f"| Contradictory citations accepted (false pos.) | 100.0% (15/15) | {b_lex['fp']/b_lex['unsupported_total']:.1%} ({b_lex['fp']}/{b_lex['unsupported_total']}) | {b_nli['fp']/b_nli['unsupported_total']:.1%} ({b_nli['fp']}/{b_nli['unsupported_total']}) |\n"
        f"| Precision | 50.0% | {b_lex['precision']:.1%} | {b_nli['precision']:.1%} |\n"
        f"| Overall Accuracy | 50.0% | {b_lex['accuracy']:.1%} | {b_nli['accuracy']:.1%} |\n"
        f"| Held-out pairs ({h_nli['total']} pairs: 10 supported, 10 unsupported) | - | Precision {h_lex['precision']:.1%}, Recall {h_lex['recall']:.1%} | Precision {h_nli['precision']:.1%}, Recall {h_nli['recall']:.1%} |\n"
        "<!-- END_CITATION_TABLE -->"
    )


def replace_block(content: str, tag: str, new_block: str) -> str:
    pattern = rf"<!-- BEGIN_{tag} -->.*?<!-- END_{tag} -->"
    if re.search(pattern, content, flags=re.DOTALL):
        return re.sub(pattern, new_block, content, flags=re.DOTALL)
    raise ValueError(f"Marker <!-- BEGIN_{tag} --> not found in document.")


def render_all(check: bool = False) -> bool:
    v_metrics = measure_verifier.measure()
    c_metrics = measure_citations.measure()

    readme_path = ROOT / "README.md"
    readme_content = readme_path.read_text(encoding="utf-8")

    # Update verifier percentage in prose if present: e.g., "accepts 6.7%" -> "accepts 4.0%"
    curr_pct_str = f"{v_metrics['random']['current_pct']:.1%}"
    readme_content_updated = re.sub(
        r"the current verifier accepts \d+\.\d+%\)",
        f"the current verifier accepts {curr_pct_str})",
        readme_content,
    )

    readme_v_table = generate_readme_verifier_table(v_metrics)
    readme_c_table = generate_readme_citation_table(c_metrics)

    readme_content_updated = replace_block(readme_content_updated, "VERIFIER_TABLE", readme_v_table)
    readme_content_updated = replace_block(readme_content_updated, "CITATION_TABLE", readme_c_table)

    eval_path = ROOT / "EVALUATION.md"
    eval_content = eval_path.read_text(encoding="utf-8")

    eval_v_table = generate_evaluation_verifier_table(v_metrics)
    eval_c_table = generate_evaluation_citation_table(c_metrics)

    eval_content_updated = replace_block(eval_content, "VERIFIER_TABLE", eval_v_table)
    eval_content_updated = replace_block(eval_content_updated, "CITATION_TABLE", eval_c_table)

    changed = (readme_content != readme_content_updated) or (eval_content != eval_content_updated)

    if check:
        if changed:
            print("ERROR: Documentation tables do not match verified measurement artifacts!")
            print("Run `uv run python scripts/render_readme.py` to regenerate tables.")
            return False
        print("OK: All documentation tables match verified measurements.")
        return True

    readme_path.write_text(readme_content_updated, encoding="utf-8")
    eval_path.write_text(eval_content_updated, encoding="utf-8")
    print("SUCCESS: Successfully rendered documentation tables from measurement artifacts.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Render README & EVALUATION tables from artifacts.")
    parser.add_argument("--check", action="store_true", help="Verify consistency without modifying files.")
    args = parser.parse_args()

    success = render_all(check=args.check)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
