"""Measures how often numerical verification accepts FABRICATED numbers,
for the current verifier and the original one (commit b13a25d, loaded from
git history), against the same realistic analytics ledger.

This is the source of the verifier numbers quoted in EVALUATION.md. No LLM is
involved, so the result is deterministic (fixed random seeds).

Run: uv run python scripts/measure_verifier.py
"""

import importlib.util
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from agent_runtime.agent import verification as current
from agent_runtime.agent.evidence import EvidenceKind, EvidenceLedger, EvidenceSource

BASELINE_COMMIT = "b13a25d"
TRIALS = 2000
ROOT = Path(__file__).resolve().parent.parent


def load_baseline_verifier():
    """The original verification.py, with its two relative imports made absolute."""
    source = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:src/agent_runtime/agent/verification.py"],
        capture_output=True, check=True, cwd=ROOT,
    ).stdout.decode("utf-8")
    source = source.replace("from ..rag.documents import Chunk", "from agent_runtime.rag.documents import Chunk")
    source = source.replace("from .evidence import EvidenceLedger", "from agent_runtime.agent.evidence import EvidenceLedger")
    path = Path(tempfile.mkdtemp()) / "baseline_verification.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("baseline_verification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_ledger() -> EvidenceLedger:
    """12 daily checkout rows (as in the product analytics database) plus an A/B test result."""
    rows = [
        {"device": device, "period": period, "checkouts_started": started, "purchases_completed": done, "conversion_rate": rate}
        for device, period, started, done, rate in [
            ("mobile", "baseline_week", 2500, 70, 0.028), ("mobile", "baseline_week", 2000, 56, 0.028),
            ("desktop", "baseline_week", 1200, 50, 0.0417), ("desktop", "baseline_week", 800, 32, 0.040),
            ("mobile", "baseline_week", 2600, 73, 0.0281), ("desktop", "baseline_week", 1300, 53, 0.0408),
            ("mobile", "last_week", 2700, 43, 0.0159), ("mobile", "last_week", 2100, 34, 0.0162),
            ("desktop", "last_week", 1250, 51, 0.0408), ("desktop", "last_week", 850, 34, 0.0400),
            ("mobile", "last_week", 2650, 42, 0.0158), ("desktop", "last_week", 1300, 52, 0.0400),
        ]
    ]
    ledger = EvidenceLedger()
    ledger.add(source=EvidenceSource.ANALYTICS_SQL, query_or_input="SELECT ...", data={"rows": rows, "row_count": 12},
               provenance="db", summary="12 rows")
    ledger.add(source=EvidenceSource.STATISTICS, kind=EvidenceKind.CALCULATION, query_or_input="EXP-204",
               data={"rate_a": 0.016, "rate_b": 0.0293, "difference_pp": 1.333, "p_value": 0.01416},
               provenance="scipy", summary="z-test")
    return ledger


def random_claims() -> list[str]:
    """Random claims over wide ranges - some values will coincide with real ones."""
    rng = random.Random(7)
    claims = []
    for _ in range(TRIALS):
        kind = rng.choice(["pct", "pp", "fell_pct"])
        if kind == "pct":
            claims.append(f"Mobile conversion was {rng.uniform(0.5, 60):.1f}%.")
        elif kind == "pp":
            claims.append(f"The gap widened by {rng.uniform(0.2, 15):.1f} percentage points.")
        else:
            claims.append(f"Conversion fell {rng.uniform(1, 90):.1f}%.")
    return claims


def out_of_range_claims() -> list[str]:
    """Percentages of 45-95%, gaps of 5-40 pp and counts of 3,000-99,000: no value or
    absolute difference in the ledger falls in these ranges."""
    rng = random.Random(11)
    claims = []
    for _ in range(TRIALS):
        kind = rng.choice(["pct", "pp", "count"])
        if kind == "pct":
            claims.append(f"Mobile conversion was {rng.uniform(45, 95):.1f}%.")
        elif kind == "pp":
            claims.append(f"The gap widened by {rng.uniform(5, 40):.1f} percentage points.")
        else:
            claims.append(f"We recorded {rng.randint(3000, 99000):,} users.")
    return claims


def swap_claims() -> list[str]:
    """Entity swaps: claiming desktop's ~4.1% conversion rate for mobile."""
    return [
        "Mobile conversion was 4.1%.",
        "Mobile conversion rate was 4.0%.",
        "Mobile conversion was 4.2%.",
    ]


def main() -> None:
    baseline = load_baseline_verifier()
    ledger = build_ledger()

    def accepted(module, claims):
        count = sum(module.verify_numerical_grounding(claim, ledger).valid for claim in claims)
        return f"{count}/{len(claims)} ({count / len(claims):.1%})"

    print(f"Random fabricated claims accepted:  baseline {accepted(baseline, random_claims())}"
          f"   current {accepted(current, random_claims())}")
    print(f"Out-of-range claims accepted:       baseline {accepted(baseline, out_of_range_claims())}"
          f"   current {accepted(current, out_of_range_claims())}")
    print(f"Entity swaps accepted:              baseline {accepted(baseline, swap_claims())}"
          f"   current {accepted(current, swap_claims())}")

    reversal = "Variant B fell by 1.33 percentage points."  # evidence: +1.333 pp
    print(f"Direction reversal accepted:        baseline {baseline.verify_numerical_grounding(reversal, ledger).valid}"
          f"   current {current.verify_numerical_grounding(reversal, ledger).valid}")

    version_non_claim = "Mobile conversion dropped to 1.6% because of the iOS 17 update."
    print(f"Version number accepted as valid:   baseline {baseline.verify_numerical_grounding(version_non_claim, ledger).valid}"
          f"   current {current.verify_numerical_grounding(version_non_claim, ledger).valid}")

    correct = [
        "Mobile conversion dropped from 2.8% to 1.6% last week.",
        "Variant B converted at 2.9% vs 1.6% for A, a 1.33 pp improvement (p = 0.014).",
        "Mobile checkouts started totalled 2,700 on one day.",
    ]
    passing = sum(current.verify_numerical_grounding(answer, ledger).valid for answer in correct)
    print(f"Correct answers accepted (current): {passing}/{len(correct)}")
    sys.exit(0 if passing == len(correct) else 1)


if __name__ == "__main__":
    main()
