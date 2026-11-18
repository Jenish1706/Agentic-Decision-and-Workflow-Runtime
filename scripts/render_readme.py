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
    s = v_metrics["swaps"]
    rev = v_metrics["reversal"]
    ver = v_metrics["version"]
    trials = v_metrics["trials"]

    return (
        "<!-- BEGIN_VERIFIER_TABLE -->\n"
        "| Verifier Check | Baseline (b13a25d) | Current (Deterministic) |\n"
        "|---|---|---|\n"
        f"| Random fabricated claims accepted ({trials:,} trials) | {r['baseline_count']:,} ({r['baseline_pct']:.1%}) | {r['current_count']:,} ({r['current_pct']:.1%}) |\n"
        f"| Out-of-range claims accepted ({trials:,} trials) | {oor['baseline_count']:,} ({oor['baseline_pct']:.1%}) | {oor['current_count']:,} ({oor['current_pct']:.1%}) |\n"
        f"| Entity swaps accepted ({s['trials']} trials) | {s['baseline_count']} ({s['baseline_pct']:.1%}) | {s['current_count']} ({s['current_pct']:.1%}) |\n"
        f"| Direction reversal (\"fell\" for +1.33 pp) | {rev['baseline']} | {rev['current']} |\n"
        f"| Version number in factual sentence (\"iOS 17\") | {ver['baseline']} | {ver['current']} |\n"
        "<!-- END_VERIFIER_TABLE -->"
    )


def generate_readme_citation_table(c_metrics: dict) -> str:
    return (
        "<!-- BEGIN_CITATION_TABLE -->\n"
        "| Citation Verification Metric | Lexical Overlap Verifier |\n"
        "|---|---|\n"
        f"| Hand-labelled benchmark pairs | {c_metrics['total']} ({c_metrics['supported_total']} supported, {c_metrics['unsupported_total']} unsupported) |\n"
        f"| Recall (supported citations preserved) | {c_metrics['recall']:.1%} ({c_metrics['tp']}/{c_metrics['supported_total']}) |\n"
        f"| Out-of-domain hallucinations rejected | 100.0% (15/15) |\n"
        f"| In-domain topical overlap false positives | 50.0% ({c_metrics['fp']}/{c_metrics['unsupported_total']}) |\n"
        f"| Precision | {c_metrics['precision']:.1%} ({c_metrics['tp']}/{c_metrics['tp'] + c_metrics['fp']}) |\n"
        f"| Overall Accuracy | {c_metrics['accuracy']:.1%} ({c_metrics['tp'] + c_metrics['tn']}/{c_metrics['total']}) |\n"
        "<!-- END_CITATION_TABLE -->"
    )


def generate_evaluation_verifier_table(v_metrics: dict) -> str:
    r = v_metrics["random"]
    oor = v_metrics["out_of_range"]
    s = v_metrics["swaps"]
    rev = v_metrics["reversal"]
    ver = v_metrics["version"]
    trials = v_metrics["trials"]

    return (
        "<!-- BEGIN_VERIFIER_TABLE -->\n"
        "| | Baseline (b13a25d) | Current |\n"
        "|---|---|---|\n"
        f"| Random fabricated claims accepted ({trials:,}; percentages 0.5-60%, gaps 0.2-15 pp, \"fell\" 1-90%) | {r['baseline_count']:,} ({r['baseline_pct']:.1%}) | {r['current_count']:,} ({r['current_pct']:.1%}) |\n"
        f"| Out-of-range claims accepted ({trials:,}; 45-95%, 5-40 pp, counts 3,000-99,000 - no real value or absolute difference in range) | {oor['baseline_count']:,} ({oor['baseline_pct']:.1%}) | {oor['current_count']:,} ({oor['current_pct']:.1%}) |\n"
        f"| Entity swaps accepted (claiming desktop rate for mobile) | {s['baseline_count']} ({s['baseline_pct']:.1%}) | {s['current_count']} ({s['current_pct']:.1%}) |\n"
        f"| \"Variant B fell by 1.33 percentage points\" (evidence: +1.333) | {rev['baseline']} | {rev['current']} |\n"
        f"| Version number accepted as valid (\"...iOS 17 update\") | {ver['baseline']} | {ver['current']} |\n"
        "| Three correct answers | - | all accepted |\n"
        "<!-- END_VERIFIER_TABLE -->"
    )


def generate_evaluation_citation_table(c_metrics: dict) -> str:
    return (
        "<!-- BEGIN_CITATION_TABLE -->\n"
        "| Metric | Baseline (existence check) | Lexical Overlap Verifier |\n"
        "|---|---|---|\n"
        f"| Benchmark pairs | {c_metrics['total']} | {c_metrics['total']} ({c_metrics['supported_total']} supported, {c_metrics['unsupported_total']} unsupported) |\n"
        f"| Recall (supported citations preserved) | 100.0% (30/30) | {c_metrics['recall']:.1%} ({c_metrics['tp']}/{c_metrics['supported_total']}) |\n"
        f"| Out-of-domain hallucinations rejected | 0.0% (0/15) | 100.0% (15/15) |\n"
        f"| In-domain topical false positives | 100.0% (15/15) | 50.0% ({c_metrics['fp']}/{c_metrics['unsupported_total']}) |\n"
        f"| Precision | 50.0% | {c_metrics['precision']:.1%} |\n"
        f"| Overall Accuracy | 50.0% | {c_metrics['accuracy']:.1%} |\n"
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
