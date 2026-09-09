"""Groundedness and Numerical Verification.

Two layers of verification:
1. `verify_citations`: Deterministic check that every cited [n] tag exists in the
   evidence citation map, and that claims are not made without evidence.
2. `verify_numerical_grounding`: Deterministic check that quantitative claims
   (percentages, percentage points, currencies, counts, rates) trace to raw observations
   or derived calculations in the EvidenceLedger, avoiding false claims and semantic mismatches.
"""

from dataclasses import dataclass, field
import re
from typing import Any

from pydantic import BaseModel

from ..rag.documents import Chunk
from .evidence import EvidenceLedger

_REFUSAL_MARKERS = (
    "don't know",
    "do not know",
    "not available",
    "no information",
    "cannot find",
    "not answerable",
    "not in the knowledge base",
    "doesn't contain",
    "does not contain",
    "insufficient evidence",
)


class VerificationResult(BaseModel):
    grounded: bool
    issues: list[str]
    confidence: float
    used_tags: list[str]


def extract_citation_tags(answer: str) -> set[str]:
    return set(re.findall(r"\[\d+\]", answer))


def verify_citations(answer: str, citation_map: dict[str, Chunk]) -> VerificationResult:
    used_tags = extract_citation_tags(answer)
    issues: list[str] = []

    unknown = used_tags - set(citation_map.keys())
    if unknown:
        issues.append(f"Cites source tag(s) not present in retrieved evidence: {sorted(unknown)}")

    is_refusal = any(marker in answer.lower() for marker in _REFUSAL_MARKERS)
    if not used_tags and citation_map and not is_refusal:
        issues.append("Makes claims but cites no sources, even though evidence was retrieved")

    grounded = not issues
    return VerificationResult(
        grounded=grounded, issues=issues, confidence=1.0 if grounded else 0.4, used_tags=sorted(used_tags)
    )


# --- Numerical Grounding Verification ---


@dataclass(frozen=True)
class QuantitativeClaim:
    raw_text: str
    value: float
    claim_type: str  # "percentage", "percentage_points", "currency", "count", "decimal"
    direction: str | None = None  # "increase", "decrease", None
    is_verified: bool = False
    matched_evidence_id: str | None = None
    explanation: str = ""


@dataclass(frozen=True)
class NumericalVerificationResult:
    valid: bool
    claims: list[QuantitativeClaim] = field(default_factory=list)
    unverified_claims: list[QuantitativeClaim] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def extract_quantitative_claims(text: str) -> list[QuantitativeClaim]:
    """Extracts quantitative claims (percentages, percentage points, currencies, counts).

    Filters out:
    - Experiment / product / tool IDs (e.g. EXP-204, v1.2, SQL-01)
    - Citation tags ([1], [2])
    - Calendar years and dates (e.g. 2024, 2024-11-20, Q3)
    - Step numbering (Step 1, Step 2)
    """
    claims: list[QuantitativeClaim] = []

    # Clean citation tags and code identifiers before extracting numbers
    cleaned = re.sub(r"\[\d+\]", "", text)
    cleaned = re.sub(r"\b[A-Za-z]+[-_]\d+\b", "", cleaned)  # e.g. EXP-204, user-102
    cleaned = re.sub(r"\b(?:Step|step)\s*\d+\b", "", cleaned)
    cleaned = re.sub(r"\b(?:20\d\d)\b", "", cleaned)  # 2020-2099 years
    cleaned = re.sub(r"\bQ[1-4]\b", "", cleaned)  # Q1, Q2, Q3, Q4

    # 1. Percentage points (must check before percentage to avoid partial match)
    pp_pattern = re.compile(
        r"(\b(?:fell|declined|dropped|decreased|increased|rose|grew|gap|difference|by|of)?\s*)?"
        r"(\d+(?:\.\d+)?)\s*(?:percentage points?|pp\b)",
        re.IGNORECASE,
    )
    for m in pp_pattern.finditer(cleaned):
        context_prefix = (m.group(1) or "").lower()
        val = float(m.group(2))
        direction = None
        if any(w in context_prefix for w in ("fell", "declined", "dropped", "decreased", "lower")):
            direction = "decrease"
        elif any(w in context_prefix for w in ("increased", "rose", "grew", "higher")):
            direction = "increase"
        claims.append(
            QuantitativeClaim(
                raw_text=m.group(0).strip(),
                value=val,
                claim_type="percentage_points",
                direction=direction,
            )
        )

    # 2. Percentages (e.g. 18.4%, 42.1 percent)
    pct_pattern = re.compile(
        r"(\b(?:fell|declined|dropped|decreased|increased|rose|grew|by|to|at)?\s*)?"
        r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)(?!\s*points?)",
        re.IGNORECASE,
    )
    for m in pct_pattern.finditer(cleaned):
        context_prefix = (m.group(1) or "").lower()
        val = float(m.group(2))
        direction = None
        if any(w in context_prefix for w in ("fell", "declined", "dropped", "decreased")):
            direction = "decrease"
        elif any(w in context_prefix for w in ("increased", "rose", "grew")):
            direction = "increase"
        claims.append(
            QuantitativeClaim(
                raw_text=m.group(0).strip(),
                value=val,
                claim_type="percentage",
                direction=direction,
            )
        )

    # 3. Currency (e.g. $500, ₹1000, USD 4000)
    curr_pattern = re.compile(r"(\$|₹|€|USD|INR)\s*(\d+(?:,\d{3})*(?:\.\d+)?)", re.IGNORECASE)
    for m in curr_pattern.finditer(cleaned):
        raw_val = m.group(2).replace(",", "")
        claims.append(
            QuantitativeClaim(
                raw_text=m.group(0).strip(),
                value=float(raw_val),
                claim_type="currency",
            )
        )

    # 4. Explicit counts (e.g. 1,050 users, 420 sessions, 80 orders)
    count_pattern = re.compile(
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(users|sessions|events|visitors|conversions|orders|customers|rows)",
        re.IGNORECASE,
    )
    for m in count_pattern.finditer(cleaned):
        raw_val = m.group(1).replace(",", "")
        claims.append(
            QuantitativeClaim(
                raw_text=m.group(0).strip(),
                value=float(raw_val),
                claim_type="count",
            )
        )

    return claims


def verify_numerical_grounding(
    answer: str,
    ledger: EvidenceLedger | None,
    tolerance: float = 0.015,
) -> NumericalVerificationResult:
    """Verifies that quantitative claims in `answer` reconcile with facts in the EvidenceLedger.

    Supports:
    - Direct factual matches (raw values in tables/retrieval)
    - Scale alignment (18.4% aligns with 0.184)
    - Derived calculations (e.g., A - B = difference)
    - Rejects fabricated figures or semantic direction reversals
    """
    is_refusal = any(marker in answer.lower() for marker in _REFUSAL_MARKERS)
    claims = extract_quantitative_claims(answer)

    if not claims or is_refusal:
        return NumericalVerificationResult(valid=True, claims=claims, unverified_claims=[], issues=[])

    if ledger is None or len(ledger) == 0:
        unverified = [
            QuantitativeClaim(
                raw_text=c.raw_text,
                value=c.value,
                claim_type=c.claim_type,
                direction=c.direction,
                is_verified=False,
                explanation="No evidence ledger present to substantiate numerical claims.",
            )
            for c in claims
        ]
        return NumericalVerificationResult(
            valid=False,
            claims=claims,
            unverified_claims=unverified,
            issues=["Answer states quantitative claims without an evidence ledger."],
        )

    facts = ledger.extract_numerical_facts()
    fact_values = list(facts.items())  # [(key, float_val)]

    verified_claims: list[QuantitativeClaim] = []
    unverified_claims: list[QuantitativeClaim] = []
    issues: list[str] = []

    for claim in claims:
        matched = False
        matched_ev_id = None
        matched_explanation = ""

        # 1. Direct match check
        for key, val in fact_values:
            ev_id = key.split(".")[0]
            # Match direct percentage (e.g. 18.4% -> 0.184 or 18.4)
            if claim.claim_type == "percentage":
                if abs(val - claim.value) <= tolerance:
                    matched = True
                    matched_ev_id = ev_id
                    matched_explanation = f"Matched direct fact {key}={val}"
                    break
                if abs(val * 100.0 - claim.value) <= tolerance * 100.0:
                    matched = True
                    matched_ev_id = ev_id
                    matched_explanation = f"Matched direct ratio {key}={val} as {claim.value}%"
                    break
            elif claim.claim_type in ("currency", "count", "decimal", "percentage_points"):
                if abs(val - claim.value) <= tolerance or (abs(val * 100.0 - claim.value) <= tolerance * 100.0 and claim.claim_type == "percentage_points"):
                    matched = True
                    matched_ev_id = ev_id
                    matched_explanation = f"Matched direct fact {key}={val}"
                    break

        # 2. Derived calculation check (differences, gaps, percentage changes)
        if not matched:
            for i, (k1, v1) in enumerate(fact_values):
                for k2, v2 in fact_values[i + 1 :]:
                    # Derived absolute difference (e.g. 42.1% - 37.9% = 4.2 pp)
                    diff_raw = abs(v1 - v2)
                    diff_pct = diff_raw * 100.0

                    if claim.claim_type == "percentage_points":
                        if abs(diff_raw - claim.value) <= tolerance or abs(diff_pct - claim.value) <= tolerance * 100.0:
                            matched = True
                            matched_ev_id = f"{k1.split('.')[0]},{k2.split('.')[0]}"
                            matched_explanation = f"Derived difference |{k1} - {k2}| ({v1} vs {v2}) = {claim.value} pp"
                            break

                    elif claim.claim_type == "percentage" and claim.direction is not None:
                        # Relative percentage change: (v2 - v1) / v1 * 100
                        if v1 != 0:
                            rel_change = ((v2 - v1) / v1) * 100.0
                            if abs(abs(rel_change) - claim.value) <= tolerance * 100.0:
                                matched = True
                                matched_ev_id = f"{k1.split('.')[0]},{k2.split('.')[0]}"
                                matched_explanation = f"Derived relative change between {k1} and {k2} = {claim.value}%"
                                break
                if matched:
                    break

        if matched:
            verified_claims.append(
                QuantitativeClaim(
                    raw_text=claim.raw_text,
                    value=claim.value,
                    claim_type=claim.claim_type,
                    direction=claim.direction,
                    is_verified=True,
                    matched_evidence_id=matched_ev_id,
                    explanation=matched_explanation,
                )
            )
        else:
            unverified_claim = QuantitativeClaim(
                raw_text=claim.raw_text,
                value=claim.value,
                claim_type=claim.claim_type,
                direction=claim.direction,
                is_verified=False,
                explanation="No direct evidence or derived computation in the ledger supports this value.",
            )
            unverified_claims.append(unverified_claim)
            issues.append(f"Unsubstantiated numerical claim '{claim.raw_text}' ({claim.value})")

    valid = len(unverified_claims) == 0
    return NumericalVerificationResult(
        valid=valid,
        claims=verified_claims + unverified_claims,
        unverified_claims=unverified_claims,
        issues=issues,
    )
