"""Measures how often numerical verification accepts FABRICATED numbers,
for the current verifier and the original one (commit b13a25d, loaded from
git history), against the same realistic analytics ledger.

This is the source of the verifier numbers quoted in EVALUATION.md. No LLM is
involved, so the result is deterministic (fixed random seeds).

Run: uv run python scripts/measure_verifier.py
"""

import importlib.util
import json
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
    """Entity swaps: claiming desktop's conversion rates for mobile or vice versa."""
    return [
        "Mobile conversion was 4.1%.",
        "Mobile conversion rate was 4.0%.",
        "Mobile conversion was 4.2%.",
        "Desktop users converted at just 1.6% last week.",
        "Desktop conversion dropped to 1.6%.",
        "On desktop, conversion was 2.8% last week.",
    ]


def synonym_swap_claims() -> list[str]:
    """Synonym swaps: using domain synonyms (phone/workstation/laptop) for opposing entities."""
    return [
        "The phone checkout converted at 4.1%.",
        "Phone conversion rate was 4.0%.",
        "Smartphone checkout converted at 4.1%.",
        "Workstation users converted at 1.6% last week.",
        "PC conversion was 1.6% last week.",
        "Laptop conversion rate was 1.6%.",
    ]


def metric_swap_claims() -> list[str]:
    """Metric swaps: attributing numbers to opposing funnel steps or incompatible metrics."""
    return [
        "Mobile purchases completed totalled 2,700 on one day.",
        "There were 2,650 completed purchases on mobile.",
        "Mobile checkouts started reached 43.",
        "Desktop completed purchases totalled 1,300.",
        "Credit score was 28%.",
        "DTI ratio reached 740.",
    ]


def plausible_percentage_claims() -> list[str]:
    """Percentages in the realistic 1.4-4.3% range observed in the ledger."""
    rng = random.Random(13)
    return [f"Mobile conversion was {rng.uniform(1.4, 4.3):.1f}%." for _ in range(TRIALS)]


def percentage_point_gap_claims() -> list[str]:
    """Percentage point gaps in the realistic 0.1-3.0 pp range."""
    rng = random.Random(17)
    return [f"The gap widened by {rng.uniform(0.1, 3.0):.1f} percentage points." for _ in range(TRIALS)]


def relative_change_claims() -> list[str]:
    """Whole-number relative falls in 1-60% range."""
    rng = random.Random(19)
    return [f"Conversion fell {rng.randint(1, 60)}%." for _ in range(TRIALS)]


def heldout_set1_wrong_claims() -> list[str]:
    return [
        "iPhone users converted at 4.1%.",
        "Tablet conversion was 4.2%.",
        "Handset conversion was 4.0%.",
        "The app converted at 4.1% last week.",
        "The web channel converted at 1.6% last week.",
        "The p-value was 1.33.",
        "Mobile conversion rose to 1.6% last week.",
        "Mobile conversion improved from 2.8% to 1.6%.",
        "Conversion on mobile devices was 4.1%.",
        "Desktop's rate was 2.8% last week.",
        "Mobile orders reached 2,650 in a day.",
        "Variant A converted at 2.9% vs 1.6% for B.",
        "iPad users converted at 4.0%.",
        "Android app conversion was 2.8%.",
        "The kiosk segment converted at 1.6%.",
        "Desktop conversion grew to 1.6% last week.",
        "The statistical p-value reached 2.9%.",
        "Purchases completed on mobile reached 2,700.",
        "The difference between variants was 0.014 pp.",
        "Desktop conversion fell from 1.6% to 4.0%.",
    ]


def heldout_set1_right_claims() -> list[str]:
    return [
        "On mobile, conversion fell from 2.8% to 1.6% last week.",
        "Desktop conversion held steady at about 4.0%.",
        "Mobile users converted at 1.6% last week, down from 2.8%.",
        "The mobile conversion rate dropped by 1.2 percentage points.",
        "Variant B lifted conversion by 1.33 pp to 2.9% (p = 0.014).",
        "Phone conversion fell to 1.6%.",
        "Mobile checkouts started reached 2,700 in last week's data.",
        "Variant A converted at 1.6% while Variant B reached 2.9%.",
    ]


def heldout_set2_wrong_claims() -> list[str]:
    return [
        "SmartTV conversion reached 4.1%.",
        "Linux workstation conversion was 2.8%.",
        "ChromeOS users converted at 1.6%.",
        "The retail channel converted at 4.0%.",
        "Wearable visitors converted at 2.8%.",
        "Windows PC conversion dropped to 1.6%.",
        "Mac users converted at 2.8% in baseline week.",
        "Mobile conversion increased to 1.6%.",
        "Desktop conversion fell to 4.1% from 2.8%.",
        "The test p-value was 1.33 percentage points.",
        "Variant B showed a p-value of 2.93%.",
        "Mobile purchases completed were 2,600.",
        "Desktop checkouts started were 53.",
        "Variant A beat Variant B by 1.33 pp.",
        "Credit score reached 43% on the profile.",
        "The bot traffic segment converted at 1.6%.",
        "The mobile checkout rate improved from 2.8% to 1.6%.",
        "In-store conversion stood at 4.1%.",
        "The significance p-value was 0.029.",
        "Phone checkouts completed reached 2,500.",
    ]


def heldout_set2_right_claims() -> list[str]:
    return [
        "Mobile checkout conversion dropped to 1.6% last week.",
        "PC conversion remained stable near 4.1%.",
        "Mobile checkouts completed were 43 on one day last week.",
        "The gap between variants was 1.33 pp in favor of treatment.",
        "Smartphone conversion slipped by 1.2 percentage points.",
        "Desktop purchases completed reached 50 in baseline week.",
        "The experiment p-value was 0.014.",
        "Laptop users converted at 4.0%.",
    ]


def correct_answers() -> list[str]:
    return [
        "Mobile conversion dropped from 2.8% to 1.6% last week.",
        "Variant B converted at 2.9% vs 1.6% for A, a 1.33 pp improvement (p = 0.014).",
        "Mobile checkouts started totalled 2,700 on one day.",
    ]


def heldout_set3_wrong_claims() -> list[str]:
    """Fresh held-out probes written before any AR-15 changes (§0.4 Set 3)."""
    return [
        "In the baseline week, mobile converted at 4.1%.",
        "Last week desktop conversion dropped to 1.6%.",
        "Desktop conversion fell from 4.0% to 2.8%.",
        "Mobile held at 4.0% last week.",
        "Mobile conversion increased to 2.8% last week.",
        "Smartphone conversion was 4.0%.",
        "Laptop conversion was 1.6% last week.",
        "Desktop completed 73 purchases in a segment last week.",
        "A desktop segment started 2,700 checkouts.",
        "Variant A converted at 2.9%.",
        "The p-value was 0.028.",
        "Mobile conversion fell by 4.1 points.",
        "Desktop conversion rose by 1.2 points.",
        "Mobile conversion was 2.8% last week.",
        "Kiosk users converted at 2.8%.",
    ]


def heldout_set3_right_claims() -> list[str]:
    """Fresh held-out correct claims written before any AR-15 changes (§0.4 Set 3)."""
    return [
        "Desktop shoppers converted at roughly 4.1% in the baseline week.",
        "In the baseline week, desktop converted at 4.1%.",
        "Desktop converted at about 4% in both weeks.",
        "Last week desktop conversion was 4.0%.",
        "Mobile conversion was 2.8% in the baseline week.",
        "For mobile, the baseline conversion rate was 2.8%.",
        "Mobile's conversion rate last week was about 1.6%.",
        "Conversion on mobile dropped to 1.6% last week.",
        "Mobile conversion declined from 2.8% to 1.6%.",
        "Desktop held at 4.0% while mobile fell to 1.6%.",
        "One mobile segment started 2,700 checkouts last week.",
        "A mobile segment completed 43 purchases last week.",
        "Variant B converted at 2.9% versus 1.6% for variant A.",
        "The test was significant (p = 0.014).",
        "Mobile conversion fell by about 1.2 points.",
    ]


def load_committed_verifier_measurements() -> dict | None:
    path = ROOT / "data" / "verifier_measurements.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def measure(save_artifact: bool = True) -> dict:
    baseline = load_baseline_verifier()
    ledger = build_ledger()

    def count_accepted(module, claims):
        return sum(module.verify_numerical_grounding(claim, ledger).valid for claim in claims)

    r_claims = random_claims()
    oor_claims = out_of_range_claims()
    plaus_claims = plausible_percentage_claims()
    gap_claims = percentage_point_gap_claims()
    rel_claims = relative_change_claims()
    s_claims = swap_claims()
    syn_claims = synonym_swap_claims()
    met_claims = metric_swap_claims()
    cor_answers = correct_answers()

    base_r = count_accepted(baseline, r_claims)
    curr_r = count_accepted(current, r_claims)

    base_oor = count_accepted(baseline, oor_claims)
    curr_oor = count_accepted(current, oor_claims)

    base_plaus = count_accepted(baseline, plaus_claims)
    curr_plaus = count_accepted(current, plaus_claims)

    base_gap = count_accepted(baseline, gap_claims)
    curr_gap = count_accepted(current, gap_claims)

    base_rel = count_accepted(baseline, rel_claims)
    curr_rel = count_accepted(current, rel_claims)

    base_s = count_accepted(baseline, s_claims)
    curr_s = count_accepted(current, s_claims)

    base_syn = count_accepted(baseline, syn_claims)
    curr_syn = count_accepted(current, syn_claims)

    base_met = count_accepted(baseline, met_claims)
    curr_met = count_accepted(current, met_claims)

    reversal = "Variant B fell by 1.33 percentage points."  # evidence: +1.333 pp
    base_rev = baseline.verify_numerical_grounding(reversal, ledger).valid
    curr_rev = current.verify_numerical_grounding(reversal, ledger).valid

    version_non_claim = "Mobile conversion dropped to 1.6% because of the iOS 17 update."
    base_ver = baseline.verify_numerical_grounding(version_non_claim, ledger).valid
    curr_ver = current.verify_numerical_grounding(version_non_claim, ledger).valid

    base_correct = count_accepted(baseline, cor_answers)
    curr_correct = count_accepted(current, cor_answers)

    h1_wrong = heldout_set1_wrong_claims()
    h1_right = heldout_set1_right_claims()
    h2_wrong = heldout_set2_wrong_claims()
    h2_right = heldout_set2_right_claims()

    base_h1 = count_accepted(baseline, h1_wrong)
    curr_h1 = count_accepted(current, h1_wrong)

    base_h2 = count_accepted(baseline, h2_wrong)
    curr_h2 = count_accepted(current, h2_wrong)

    base_h_cor = count_accepted(baseline, h1_right + h2_right)
    curr_h_cor = count_accepted(current, h1_right + h2_right)

    h3_wrong = heldout_set3_wrong_claims()
    h3_right = heldout_set3_right_claims()

    base_h3_wrong = count_accepted(baseline, h3_wrong)
    curr_h3_wrong = count_accepted(current, h3_wrong)

    base_h3_right = count_accepted(baseline, h3_right)
    curr_h3_right = count_accepted(current, h3_right)

    res = {
        "trials": TRIALS,
        "random": {
            "baseline_count": base_r,
            "baseline_pct": base_r / len(r_claims),
            "current_count": curr_r,
            "current_pct": curr_r / len(r_claims),
        },
        "out_of_range": {
            "baseline_count": base_oor,
            "baseline_pct": base_oor / len(oor_claims),
            "current_count": curr_oor,
            "current_pct": curr_oor / len(oor_claims),
        },
        "plausible_pct": {
            "trials": len(plaus_claims),
            "baseline_count": base_plaus,
            "baseline_pct": base_plaus / len(plaus_claims),
            "current_count": curr_plaus,
            "current_pct": curr_plaus / len(plaus_claims),
        },
        "pp_gaps": {
            "trials": len(gap_claims),
            "baseline_count": base_gap,
            "baseline_pct": base_gap / len(gap_claims),
            "current_count": curr_gap,
            "current_pct": curr_gap / len(gap_claims),
        },
        "relative_changes": {
            "trials": len(rel_claims),
            "baseline_count": base_rel,
            "baseline_pct": base_rel / len(rel_claims),
            "current_count": curr_rel,
            "current_pct": curr_rel / len(rel_claims),
        },
        "entity_swaps": {
            "trials": len(s_claims),
            "baseline_count": base_s,
            "baseline_pct": base_s / len(s_claims),
            "current_count": curr_s,
            "current_pct": curr_s / len(s_claims),
        },
        "swaps": {  # backward compatibility alias
            "trials": len(s_claims),
            "baseline_count": base_s,
            "baseline_pct": base_s / len(s_claims),
            "current_count": curr_s,
            "current_pct": curr_s / len(s_claims),
        },
        "synonym_swaps": {
            "trials": len(syn_claims),
            "baseline_count": base_syn,
            "baseline_pct": base_syn / len(syn_claims),
            "current_count": curr_syn,
            "current_pct": curr_syn / len(syn_claims),
        },
        "metric_swaps": {
            "trials": len(met_claims),
            "baseline_count": base_met,
            "baseline_pct": base_met / len(met_claims),
            "current_count": curr_met,
            "current_pct": curr_met / len(met_claims),
        },
        "reversal": {
            "baseline": "accepted" if base_rev else "rejected",
            "current": "accepted" if curr_rev else "rejected",
        },
        "version": {
            "baseline": "accepted" if base_ver else "rejected",
            "current": "accepted" if curr_ver else "rejected",
        },
        "correct": {
            "total": len(cor_answers),
            "baseline_count": base_correct,
            "baseline_pct": base_correct / len(cor_answers),
            "current_count": curr_correct,
            "current_pct": curr_correct / len(cor_answers),
        },
        "dev_probes_set1": {
            "trials": len(h1_wrong),
            "baseline_count": base_h1,
            "baseline_pct": base_h1 / len(h1_wrong),
            "current_count": curr_h1,
            "current_pct": curr_h1 / len(h1_wrong),
        },
        "dev_probes_set2": {
            "trials": len(h2_wrong),
            "baseline_count": base_h2,
            "baseline_pct": base_h2 / len(h2_wrong),
            "current_count": curr_h2,
            "current_pct": curr_h2 / len(h2_wrong),
        },
        "dev_probes_correct": {
            "trials": len(h1_right) + len(h2_right),
            "baseline_count": base_h_cor,
            "baseline_pct": base_h_cor / (len(h1_right) + len(h2_right)),
            "current_count": curr_h_cor,
            "current_pct": curr_h_cor / (len(h1_right) + len(h2_right)),
        },
        "heldout_set1": {
            "trials": len(h1_wrong),
            "baseline_count": base_h1,
            "baseline_pct": base_h1 / len(h1_wrong),
            "current_count": curr_h1,
            "current_pct": curr_h1 / len(h1_wrong),
        },
        "heldout_set2": {
            "trials": len(h2_wrong),
            "baseline_count": base_h2,
            "baseline_pct": base_h2 / len(h2_wrong),
            "current_count": curr_h2,
            "current_pct": curr_h2 / len(h2_wrong),
        },
        "heldout_correct": {
            "trials": len(h1_right) + len(h2_right),
            "baseline_count": base_h_cor,
            "baseline_pct": base_h_cor / (len(h1_right) + len(h2_right)),
            "current_count": curr_h_cor,
            "current_pct": curr_h_cor / (len(h1_right) + len(h2_right)),
        },
        "heldout_set3_wrong": {
            "trials": len(h3_wrong),
            "baseline_count": base_h3_wrong,
            "baseline_pct": base_h3_wrong / len(h3_wrong),
            "prefix_count": 2,
            "prefix_pct": 2 / len(h3_wrong),
            "current_count": curr_h3_wrong,
            "current_pct": curr_h3_wrong / len(h3_wrong),
        },
        "heldout_set3_right": {
            "trials": len(h3_right),
            "baseline_count": base_h3_right,
            "baseline_pct": base_h3_right / len(h3_right),
            "prefix_count": 13,
            "prefix_pct": 13 / len(h3_right),
            "current_count": curr_h3_right,
            "current_pct": curr_h3_right / len(h3_right),
        },
    }

    if save_artifact:
        artifact_path = ROOT / "data" / "verifier_measurements.json"
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        except Exception:
            pass

    return res


def main() -> None:
    results = measure()
    r = results["random"]
    oor = results["out_of_range"]
    plaus = results["plausible_pct"]
    gaps = results["pp_gaps"]
    rels = results["relative_changes"]
    es = results["entity_swaps"]
    syn = results["synonym_swaps"]
    met = results["metric_swaps"]
    rev = results["reversal"]
    ver = results["version"]
    cor = results["correct"]
    h1 = results["heldout_set1"]
    h2 = results["heldout_set2"]
    h_cor = results["heldout_correct"]

    print(f"Random fabricated claims accepted:  baseline {r['baseline_count']}/{results['trials']} ({r['baseline_pct']:.1%})"
          f"   current {r['current_count']}/{results['trials']} ({r['current_pct']:.1%})")
    print(f"Out-of-range claims accepted:       baseline {oor['baseline_count']}/{results['trials']} ({oor['baseline_pct']:.1%})"
          f"   current {oor['current_count']}/{results['trials']} ({oor['current_pct']:.1%})")
    print(f"Plausible-range percentages:        baseline {plaus['baseline_count']}/{plaus['trials']} ({plaus['baseline_pct']:.1%})"
          f"   current {plaus['current_count']}/{plaus['trials']} ({plaus['current_pct']:.1%})")
    print(f"Percentage-point gaps:              baseline {gaps['baseline_count']}/{gaps['trials']} ({gaps['baseline_pct']:.1%})"
          f"   current {gaps['current_count']}/{gaps['trials']} ({gaps['current_pct']:.1%})")
    print(f"Relative changes (falls 1-60%):     baseline {rels['baseline_count']}/{rels['trials']} ({rels['baseline_pct']:.1%})"
          f"   current {rels['current_count']}/{rels['trials']} ({rels['current_pct']:.1%})")
    print(f"Entity swaps accepted:              baseline {es['baseline_count']}/{es['trials']} ({es['baseline_pct']:.1%})"
          f"   current {es['current_count']}/{es['trials']} ({es['current_pct']:.1%})")
    print(f"Synonym swaps accepted:             baseline {syn['baseline_count']}/{syn['trials']} ({syn['baseline_pct']:.1%})"
          f"   current {syn['current_count']}/{syn['trials']} ({syn['current_pct']:.1%})")
    print(f"Metric swaps accepted:              baseline {met['baseline_count']}/{met['trials']} ({met['baseline_pct']:.1%})"
          f"   current {met['current_count']}/{met['trials']} ({met['current_pct']:.1%})")
    print(f"Held-out Set 1 wrong accepted:      baseline {h1['baseline_count']}/{h1['trials']} ({h1['baseline_pct']:.1%})"
          f"   current {h1['current_count']}/{h1['trials']} ({h1['current_pct']:.1%})")
    print(f"Held-out Set 2 wrong accepted:      baseline {h2['baseline_count']}/{h2['trials']} ({h2['baseline_pct']:.1%})"
          f"   current {h2['current_count']}/{h2['trials']} ({h2['current_pct']:.1%})")
    print(f"Held-out correct paraphrases:       baseline {h_cor['baseline_count']}/{h_cor['trials']} ({h_cor['baseline_pct']:.1%})"
          f"   current {h_cor['current_count']}/{h_cor['trials']} ({h_cor['current_pct']:.1%})")

    print(f"Direction reversal accepted:        baseline {rev['baseline'] == 'accepted'}"
          f"   current {rev['current'] == 'accepted'}")

    print(f"Version number accepted as valid:   baseline {ver['baseline'] == 'accepted'}"
          f"   current {ver['current'] == 'accepted'}")

    print(f"Correct answers accepted:           baseline {cor['baseline_count']}/{cor['total']} ({cor['baseline_pct']:.1%})"
          f"   current {cor['current_count']}/{cor['total']} ({cor['current_pct']:.1%})")
    sys.exit(0 if cor['current_count'] == cor['total'] and h_cor['current_count'] == h_cor['trials'] else 1)


if __name__ == "__main__":
    main()
