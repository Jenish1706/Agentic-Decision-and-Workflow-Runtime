"""Deterministic answer verification: citations and numbers.

Two independent checks run on every final answer before the agent loop will
accept it (agent/loop.py):

1. `verify_citations` - every [n] tag in the answer must refer to a chunk that
   retrieval actually returned, and a purely retrieval-based answer must cite
   something.
2. `verify_numerical_grounding` - every quantitative claim in the answer must
   be supported by evidence: a number a tool returned, a number printed in
   retrieved/tool text, a number the user's own question stated, or a declared
   policy threshold.

WHAT "SUPPORTED" MEANS (and the bugs the current rules replace)
    A previous version accepted a claim if ANY ledger number came within a
    flat 1.5 percentage points of it, and additionally searched every pair of
    ledger numbers for a matching difference. With a few dozen numbers in
    evidence, almost any fabricated percentage found a coincidental match. It
    also discarded the sign of changes, so "fell 20%" matched a 20% rise, and
    it deleted every number between 2000 and 2099 as a "year" - which let a
    fabricated "$2000" skip verification entirely.

    The rules are now:

    - Rounding-aware tolerance. A claim is consistent with a value that rounds
      to it at the precision the claim was written with: "2.9%" accepts
      2.85-2.95, "2.93%" accepts 2.925-2.935, "$5,000" accepts 4999.5-5000.5.
      Nothing looser.
    - Scale alignment. "18.4%" matches a stored ratio 0.184 or a stored
      percentage 18.4.
    - Derived values are searched only between *comparable* numbers: values
      that share a field name (conversion_rate vs conversion_rate) or come
      from the same tool output (rate_a vs rate_b). Groups larger than
      `_MAX_DERIVATION_GROUP` are skipped - past that size coincidental
      matches become likely, and the agent should compute the difference with
      the calculator tool, which records it as evidence.
    - Direction. A "rose/fell" claim must agree in sign with a signed change
      found in evidence (fields such as difference_pp or relative_lift_pct),
      and a relative change is only derived in the direction the claim states.
      A percentage-point gap derived from two unordered values has no
      recoverable direction, so only its magnitude is verified; that
      limitation is reported in the claim's explanation rather than hidden.

KNOWN LIMITATIONS (worth stating in an interview)
    - This verifies that each number appears in evidence at a compatible
      scale. It does not verify that the number is attached to the right
      concept - an answer that swaps two real numbers ("DTI 740, score 28%")
      passes. Semantic attribution is handled where it matters for decisions:
      the policy engine reads facts per entity from structured evidence
      (agent/facts.py) and never from the answer text.
    - Calculator results count as evidence, but the calculator's *inputs* are
      not checked: a number the model invents and passes through
      `calculator("9999 + 0")` becomes grounded. Requiring every input to be
      grounded would also reject legitimate constants (x 100, / 12), so the
      audit trail - which records every expression - is the control here.
    - Small integers (0-10) written as plain numbers are not treated as
      claims, so "1 delinquency" is not checked. Observed in a real run: the
      model wrote that for an account whose evidence says 0; the delinquency
      rule, reading evidence, was unaffected.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ..rag.documents import Chunk
from .evidence import EvidenceKind, EvidenceLedger, EvidenceSource

_REFUSAL_MARKERS = (
    "don't know",
    "do not know",
    "not available",
    "no information",
    "cannot find",
    "cannot be determined",
    "not answerable",
    "not in the knowledge base",
    "doesn't contain",
    "does not contain",
    "insufficient evidence",
)

_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
})


def _stem(token: str) -> str:
    token = token.lower()
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            if suffix == "ies":
                return token[:-3] + "y"
            return token[:-len(suffix)]
    return token


def _extract_content_tokens(text: str) -> set[str]:
    cleaned = re.sub(r"\[\d+\]", " ", text.lower()).replace("_", " ").replace("-", " ")
    tokens = re.findall(r"\b[a-z0-9]+\b", cleaned)
    return {_stem(t) for t in tokens if t not in _STOPWORDS and len(t) > 1}


def chunk_supports_sentence(sentence: str, chunk: Chunk) -> tuple[bool, str]:
    """Check whether `chunk` contains lexical evidence supporting `sentence`."""
    sentence_tokens = _extract_content_tokens(sentence)
    if not sentence_tokens:
        return True, ""

    chunk_text = f"{chunk.doc_title} {chunk.text}"
    chunk_tokens = _extract_content_tokens(chunk_text)

    overlap = sentence_tokens & chunk_tokens
    overlap_ratio = len(overlap) / len(sentence_tokens)

    # Support requires either >= 2 shared content keywords, or >= 1 keyword with >= 40% overlap
    is_supported = len(overlap) >= 2 or (len(overlap) >= 1 and overlap_ratio >= 0.4)
    if is_supported:
        return True, f"supported by {len(overlap)} keywords ({overlap_ratio:.0%}): {sorted(overlap)[:5]}"
    return False, f"insufficient keyword support: only {len(overlap)} of {len(sentence_tokens)} keywords match ({overlap_ratio:.0%})"


def is_refusal(answer: str) -> bool:
    """True when the answer declines to state facts, so it needs no citation.

    A phrase match, and deliberately used only for the citation requirement:
    an answer that mixes a refusal phrase with factual claims would slip past
    "must cite", which is why numerical verification does NOT honour it (see
    verify_numerical_grounding). Non-numeric claims in such a mixed answer
    remain a known gap of this heuristic.
    """
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# 1. Citations
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    grounded: bool
    issues: list[str]
    confidence: float
    used_tags: list[str]
    """The [n] tags actually present in the answer text - not every tag that
    retrieval produced this run. AgentState.citations is set from this field
    so that "citations" means "what the answer cites"."""


def extract_citation_tags(answer: str) -> set[str]:
    return set(re.findall(r"\[\d+\]", answer))


def verify_citations(
    answer: str,
    citation_map: Mapping[str, Chunk],
    *,
    has_non_retrieval_evidence: bool = False,
) -> VerificationResult:
    """Check the answer's [n] tags against what retrieval returned.

    `has_non_retrieval_evidence` exists because citation tags are only issued
    for retrieved chunks. An answer built from SQL or tool output has nothing
    to put in brackets; requiring a document citation there made correct
    tool-grounded answers fail whenever an unrelated retrieval had also
    happened. Such answers are instead held to numerical verification.
    """
    used_tags = extract_citation_tags(answer)
    issues: list[str] = []

    unknown = used_tags - set(citation_map)
    if unknown:
        issues.append(f"Cites source tag(s) not present in retrieved evidence: {sorted(unknown)}")

    if not used_tags and citation_map and not has_non_retrieval_evidence and not is_refusal(answer):
        issues.append("Makes claims but cites no sources, even though evidence was retrieved")

    # Verify that each cited chunk actually supports the sentence that cites it
    for tag in sorted(used_tags):
        if tag not in citation_map:
            continue
        chunk = citation_map[tag]
        for match in re.finditer(re.escape(tag), answer):
            start = match.start()
            left = start
            while left > 0 and answer[left - 1] not in ".?!\n\r":
                left -= 1
            right = match.end()
            while right < len(answer) and answer[right] not in ".?!\n\r":
                right += 1
            sentence = answer[left:right].strip()
            supported, explanation = chunk_supports_sentence(sentence, chunk)
            if not supported:
                issues.append(f"Citation {tag} does not support claim '{sentence}' ({explanation})")

    grounded = not issues
    return VerificationResult(
        grounded=grounded, issues=issues, confidence=1.0 if grounded else 0.4, used_tags=sorted(used_tags)
    )


# ---------------------------------------------------------------------------
# 2. Numbers - claim extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantitativeClaim:
    raw_text: str
    value: float
    claim_type: str  # "percentage" | "percentage_points" | "currency" | "count" | "number"
    tolerance: float = 0.0
    direction: str | None = None  # "increase" | "decrease" | None
    is_verified: bool = False
    matched_evidence_id: str | None = None
    explanation: str = ""
    sentence: str = ""
    context_words: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class NumericalVerificationResult:
    valid: bool
    claims: list[QuantitativeClaim] = field(default_factory=list)
    unverified_claims: list[QuantitativeClaim] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


_MONTHS = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_COUNT_NOUNS = (
    r"users|sessions|events|visitors|conversions|orders|customers|rows|accounts|incidents|"
    r"requests|checkouts|purchases|minutes|seconds|hours|days|nodes|replicas|items|invoices"
)

# Spans that contain digits but are not quantitative claims. They are blanked
# out (replaced with spaces, so offsets and word boundaries survive) before
# any claim pattern runs.
_NON_METRIC_PATTERNS = [
    re.compile(r"\[\d+\]"),  # citation tags
    # identifiers: EXP-204, ACCT-101, INV-2026-001, CHART-SLA-0926, user_102
    re.compile(r"\b[A-Za-z]+(?:[-_][A-Za-z0-9]+)*[-_]\d[A-Za-z0-9-]*\b"),
    re.compile(r"\bv?\d+(?:\.\d+){2,}\b"),  # versions / builds: 4.12.0
    # major OS/software/platform/model versions: iOS 17, Android 14, Windows 11, Python 3, v2, version 3
    re.compile(r"\b(?:iOS|Android|Windows|macOS|OS|Python|Ubuntu|v|version|ver|model)\s*\d+(?:\.\d+)*\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-W\d{2}\b"),  # ISO weeks: 2026-W38
    re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b"),  # ISO dates: 2024-11-10
    re.compile(rf"\b{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b", re.IGNORECASE),
    re.compile(rf"\b{_MONTHS}\s+(?:19|20)\d{{2}}\b", re.IGNORECASE),
    # ordinals and labels: Step 1, Week 3, Q3, P0, Tier 1, top 5, variant 2
    re.compile(r"\b(?:step|week|phase|stage|tier|priority|rank|top|level|variant|p|q)\s*\d+\b", re.IGNORECASE),
    # hyphenated descriptors: 90-day, 3-step, 30-minute
    re.compile(r"\b\d+-(?:day|week|month|year|hour|minute|second|step|point|page|tier)s?\b", re.IGNORECASE),
    # A bare four-digit year - but only when it is NOT a currency amount and
    # NOT followed by a unit. This is deliberately narrower than "any 20xx":
    # "$2000" and "2050 users" are claims and must stay.
    re.compile(
        rf"(?<![$€₹\d.,])\b(?:19|20)\d{{2}}\b(?!\s*(?:%|percent|pp\b|percentage|{_COUNT_NOUNS}|[kKmMbB]\b))",
        re.IGNORECASE,
    ),
]

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_DECREASE_WORDS = ("fell", "declined", "dropped", "decreased", "lower", "down", "reduced", "drop")
_INCREASE_WORDS = ("rose", "increased", "grew", "higher", "up", "improved", "gained", "lift", "increase")

# The optional leading word (letters only, so it can never swallow a
# neighbouring number) is read for direction: "fell by 4.2 pp", "rose to 85%".
_PP_PATTERN = re.compile(
    rf"(?:([A-Za-z]+)\s+(?:by\s+)?)?({_NUMBER})\s*(?:percentage[ -]points?|pp\b)", re.IGNORECASE
)
_PCT_PATTERN = re.compile(
    rf"(?:([A-Za-z]+)\s+(?:by\s+|to\s+)?)?({_NUMBER})\s*(?:%|percent\b)(?!\s*points?)", re.IGNORECASE
)
_CURRENCY_PATTERN = re.compile(rf"(?:\$|€|₹|\bUSD\s*|\bINR\s*)({_NUMBER})\s*([kKmMbB]\b)?", re.IGNORECASE)
_PLAIN_PATTERN = re.compile(rf"(?<![\w.])({_NUMBER})\s*([kKmMbB]\b)?(?:\s*({_COUNT_NOUNS})\b)?", re.IGNORECASE)

_SUFFIX_MULTIPLIER = {"k": 1e3, "m": 1e6, "b": 1e9}


def _parse_number(text: str) -> float:
    return float(text.replace(",", ""))


def _rounding_tolerance(number_text: str, multiplier: float = 1.0) -> float:
    """Half of the smallest unit the claim was written in: '2.9' -> 0.05,
    '2.93' -> 0.005, '5,000' -> 0.5 (scaled by any k/M suffix)."""
    decimals = len(number_text.split(".", 1)[1]) if "." in number_text else 0
    return 0.5 * (10 ** -decimals) * multiplier


def _direction_from(word: str | None) -> str | None:
    if not word:
        return None
    word = word.lower()
    if word in _DECREASE_WORDS:
        return "decrease"
    if word in _INCREASE_WORDS:
        return "increase"
    return None


def extract_quantitative_claims(text: str) -> list[QuantitativeClaim]:
    """Find every quantitative claim in `text`.

    Each span of text is claimed by at most one pattern, checked from most to
    least specific (percentage points, percentages, currency, then plain
    numbers), so "4.2 percentage points" is never re-read as the number 4.2.
    Plain integers from 0 to 10 are skipped: in prose they are almost always
    small counts or ordinals ("2 approvals", "3 variants") and treating them
    as claims produced false rejections without catching real fabrications.
    """
    cleaned = text
    for pattern in _NON_METRIC_PATTERNS:
        cleaned = pattern.sub(lambda match: " " * len(match.group(0)), cleaned)

    claimed = [False] * len(cleaned)
    claims: list[QuantitativeClaim] = []

    def _sentence_and_words(start: int, end: int) -> tuple[str, frozenset[str]]:
        def _is_boundary(idx: int) -> bool:
            ch = text[idx]
            if ch in "\n\r?!":
                return True
            if ch == ".":
                # Decimal point inside numbers (e.g. 28.0% or 1.6) is not a sentence boundary
                if idx + 1 < len(text) and text[idx + 1].isdigit():
                    return False
                if idx > 0 and text[idx - 1].isdigit():
                    return False
                return True
            return False

        left = start
        while left > 0 and not _is_boundary(left - 1):
            left -= 1
        right = end
        while right < len(text) and not _is_boundary(right):
            right += 1
        s = text[left:right].strip()
        words = frozenset(re.findall(r"[a-z0-9_-]+", s.lower()))
        return s, words

    def _take(match: re.Match[str], number_group: int) -> bool:
        start, end = match.span(number_group)
        if any(claimed[start:end]):
            return False
        for index in range(match.start(), match.end()):
            claimed[index] = True
        return True

    for match in _PP_PATTERN.finditer(cleaned):
        if _take(match, 2):
            number = match.group(2)
            sent, words = _sentence_and_words(match.start(), match.end())
            claims.append(QuantitativeClaim(
                raw_text=match.group(0).strip(),
                value=_parse_number(number),
                claim_type="percentage_points",
                tolerance=_rounding_tolerance(number),
                direction=_direction_from(match.group(1)),
                sentence=sent,
                context_words=words,
            ))

    for match in _PCT_PATTERN.finditer(cleaned):
        if _take(match, 2):
            number = match.group(2)
            sent, words = _sentence_and_words(match.start(), match.end())
            claims.append(QuantitativeClaim(
                raw_text=match.group(0).strip(),
                value=_parse_number(number),
                claim_type="percentage",
                tolerance=_rounding_tolerance(number),
                direction=_direction_from(match.group(1)),
                sentence=sent,
                context_words=words,
            ))

    for match in _CURRENCY_PATTERN.finditer(cleaned):
        if _take(match, 1):
            number = match.group(1)
            multiplier = _SUFFIX_MULTIPLIER.get((match.group(2) or "").lower(), 1.0)
            sent, words = _sentence_and_words(match.start(), match.end())
            claims.append(QuantitativeClaim(
                raw_text=match.group(0).strip(),
                value=_parse_number(number) * multiplier,
                claim_type="currency",
                tolerance=_rounding_tolerance(number, multiplier),
                sentence=sent,
                context_words=words,
            ))

    for match in _PLAIN_PATTERN.finditer(cleaned):
        start, end = match.span(1)
        if any(claimed[start:end]):
            continue
        number = match.group(1)
        multiplier = _SUFFIX_MULTIPLIER.get((match.group(2) or "").lower(), 1.0)
        value = _parse_number(number) * multiplier
        if "." not in number and multiplier == 1.0 and value <= 10:
            continue
        _take(match, 1)
        sent, words = _sentence_and_words(match.start(), match.end())
        claims.append(QuantitativeClaim(
            raw_text=match.group(0).strip(),
            value=value,
            claim_type="count" if match.group(3) else "number",
            tolerance=_rounding_tolerance(number, multiplier),
            sentence=sent,
            context_words=words,
        ))

    return claims


# ---------------------------------------------------------------------------
# 2. Numbers - grounding
# ---------------------------------------------------------------------------

# Numeric fields that are bookkeeping rather than facts about the world. A
# cross-encoder relevance score of 6.82 is not something an answer can
# legitimately "cite".
_NOISE_KEYS = frozenset({"score", "timestamp", "max_rows"})
# Field names whose values are signed changes, which lets direction be checked.
_SIGNED_CHANGE_HINTS = ("difference", "diff", "delta", "change", "lift", "growth")
# Field names whose values are already expressed in percent units (94.2, not 0.942).
_PERCENT_UNIT_HINTS = ("pct", "percent", "_pp", "uptime")
_MAX_DERIVATION_GROUP = 30


@dataclass(frozen=True)
class _GroundingValue:
    value: float
    source_id: str  # evidence id, "user_query", or "policy:<name>"
    leaf: str  # field name, or "" for numbers taken from text
    # Unit information decides which claims a value may support. Without it,
    # "43%" was "grounded" by a raw count of 43 purchases.
    is_ratio: bool = False  # 0.184 meaning 18.4%
    is_percent: bool = False  # 18.4 already in percent units
    signed_change: bool = False
    entity_context: Mapping[str, str] = field(default_factory=dict)


def _leaf_name(path: str) -> str:
    return re.sub(r"\[\d+\]$", "", path.rsplit(".", 1)[-1])


def _extract_entity_context(item: Any, path: str) -> dict[str, str]:
    """Extract identifying attributes (e.g. device, account_id, variant) for this number."""
    entities: dict[str, str] = {}
    if getattr(item, "query_or_input", None):
        ident_match = re.search(r"\b(?:ACCT|EXP|INV)-\d+\b", str(item.query_or_input), re.IGNORECASE)
        if ident_match:
            entities["entity_id"] = ident_match.group(0).lower()

    # Walk the path in item.data to find parent mapping
    tokens = path.split(".")[1:]
    curr = getattr(item, "data", {})
    parent_dict = None
    for token in tokens:
        if "[" in token:
            name, idx_str = token.split("[", 1)
            idx = int(idx_str.rstrip("]"))
            if isinstance(curr, Mapping) and name in curr:
                curr = curr[name]
            if isinstance(curr, (list, tuple)) and 0 <= idx < len(curr):
                parent_dict = curr[idx] if isinstance(curr[idx], Mapping) else None
                curr = curr[idx]
        elif isinstance(curr, Mapping) and token in curr:
            parent_dict = curr
            curr = curr[token]
        else:
            break

    if isinstance(parent_dict, Mapping):
        for k, v in parent_dict.items():
            if isinstance(v, (str, int)) and not isinstance(v, bool):
                k_lower = str(k).lower()
                if k_lower in ("device", "period", "account_id", "user_id", "variant", "segment", "country", "status", "experiment_id", "date", "week"):
                    entities[k_lower] = str(v).lower()

    leaf = _leaf_name(path).lower()
    if leaf.endswith("_a") or leaf.endswith(" a"):
        entities["variant"] = "a"
    elif leaf.endswith("_b") or leaf.endswith(" b"):
        entities["variant"] = "b"

    return entities


def _classify_field(leaf: str, value: float) -> tuple[bool, bool]:
    """(is_ratio, is_percent) for a structured field, from its name and value."""
    name = leaf.lower()
    if any(hint in name for hint in _PERCENT_UNIT_HINTS):
        # e.g. on_time_payment_pct=1.0 is stored as a ratio despite its name,
        # so a named percent field in [0, 1] may be read either way.
        return (0.0 <= abs(value) <= 1.0), True
    rate_like = any(hint in name for hint in ("rate", "ratio", "share", "margin", "utilization", "p_value"))
    is_fraction = 0.0 < abs(value) < 1.0
    return (is_fraction or (rate_like and abs(value) <= 1.0)), False


def _numbers_in_text(text: str) -> list[tuple[float, bool]]:
    """(value, written_as_percent) for every number in free text."""
    return [
        (_parse_number(match.group(1)), bool(match.group(2)))
        for match in re.finditer(rf"({_NUMBER})\s*(%|percent\b)?", text, re.IGNORECASE)
    ]


# Text fields that echo the *request* rather than report a result. A SQL
# string's "LIMIT 50" must not make "50" a grounded fact.
_REQUEST_TEXT_KEYS = frozenset({"query", "sql"})


def _walk_strings(value: Any) -> Iterable[str]:
    """Every piece of text in an evidence payload: string values AND field
    names. Field names carry facts of their own - `delinquencies_90d` is why
    "no delinquencies in the last 90 days" is a grounded statement. (Observed:
    a real run was rejected for exactly that sentence while only values were
    walked.) Numbers found this way are plain numbers, never percentages."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if key not in _REQUEST_TEXT_KEYS:
                if isinstance(key, str):
                    yield key
                yield from _walk_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child)


def collect_grounding_values(
    ledger: EvidenceLedger | None,
    *,
    query: str = "",
    reference_values: Mapping[str, float] | None = None,
) -> list[_GroundingValue]:
    """Every number an answer may legitimately use, with where it came from."""
    values: list[_GroundingValue] = []
    if ledger is not None:
        # Calculator results carry no unit: "3000 / 5000 * 100" returns a bare
        # 60.0 that the model legitimately reports as "60%". The system prompt
        # tells the model to derive numbers with the calculator, so its results
        # may ground percent claims as well as plain ones. (Observed: a real
        # run was rejected for exactly this "60%" before the exception existed.)
        calculator_ids = {item.evidence_id for item in ledger.all() if item.source is EvidenceSource.CALCULATOR}
        for path, number in ledger.extract_numerical_facts().items():
            leaf = _leaf_name(path)
            if leaf in _NOISE_KEYS:
                continue
            source_id = path.split(".", 1)[0]
            is_ratio, is_percent = _classify_field(leaf, number)
            item = ledger.get_by_id(source_id)
            entity_context = _extract_entity_context(item, path) if item else {}
            values.append(_GroundingValue(
                value=number,
                source_id=source_id,
                leaf=leaf,
                is_ratio=is_ratio,
                is_percent=is_percent or source_id in calculator_ids,
                signed_change=any(hint in leaf.lower() for hint in _SIGNED_CHANGE_HINTS),
                entity_context=entity_context,
            ))
        for item in ledger.all():
            # Numbers inside text count too: a retrieved chunk saying "5,000
            # events per second" grounds the claim "5,000", even though no
            # structured field holds it. RAG answers were being rejected
            # without this.
            texts = [item.summary, *_walk_strings(item.data)]
            if item.kind is EvidenceKind.RETRIEVAL:
                texts.append(item.query_or_input)
            for text in texts:
                for number, as_percent in _numbers_in_text(text):
                    values.append(_GroundingValue(
                        value=number, source_id=item.evidence_id, leaf="", is_percent=as_percent
                    ))
    # Numbers the user stated are inputs, not claims the model invented.
    for number, as_percent in _numbers_in_text(query):
        values.append(_GroundingValue(value=number, source_id="user_query", leaf="", is_percent=as_percent))
    for name, number in (reference_values or {}).items():
        is_ratio, is_percent = _classify_field(name, float(number))
        values.append(_GroundingValue(
            value=float(number), source_id=f"policy:{name}", leaf=name, is_ratio=is_ratio, is_percent=is_percent
        ))
    return values


def _close(candidate: float, claimed: float, tolerance: float) -> bool:
    # A tiny relative epsilon absorbs float representation noise (0.1 + 0.2).
    return abs(candidate - claimed) <= tolerance + 1e-9 * max(1.0, abs(claimed))


def _unit_matches(candidate: float, grounding: _GroundingValue, claim: QuantitativeClaim) -> bool:
    """Does `candidate` (a grounding value, or a magnitude derived from one)
    support `claim` at a compatible unit?

    - Percent / pp claims accept ratio values scaled by 100, and values already
      in percent units. They do NOT accept plain numbers - a count of 43 is not
      "43%".
    - Every other claim compares values directly, with no scaling.
    """
    if claim.claim_type in ("percentage", "percentage_points"):
        if grounding.is_ratio and _close(candidate * 100.0, claim.value, claim.tolerance):
            return True
        return grounding.is_percent and _close(candidate, claim.value, claim.tolerance)
    return _close(candidate, claim.value, claim.tolerance)


def _direction_consistent(value: float, direction: str | None) -> bool:
    if direction == "increase":
        return value > 0
    if direction == "decrease":
        return value < 0
    return True


def _find_direction_contradiction(claim: QuantitativeClaim, values: list[_GroundingValue]) -> str | None:
    """A directional claim whose magnitude matches a signed change in evidence
    but whose sign is wrong is not merely unsupported - the evidence says the
    opposite. Detected first, so no weaker match can paper over it."""
    if claim.direction is None:
        return None
    for grounding in values:
        if (
            grounding.signed_change
            and _unit_matches(abs(grounding.value), grounding, claim)
            and not _direction_consistent(grounding.value, claim.direction)
        ):
            return f"evidence {grounding.leaf}={grounding.value:+g} contradicts the claimed {claim.direction}"
    return None


def _entity_and_metric_compatible(
    claim: QuantitativeClaim,
    grounding: _GroundingValue,
    category_values: Mapping[str, set[str]],
    all_values: list[_GroundingValue],
) -> tuple[bool, str]:
    # 1. Entity swap check: if an opposing entity in the same category is mentioned
    # in the sentence while the evidence's own entity is absent, this is an entity swap.
    for cat, val in grounding.entity_context.items():
        all_vals = category_values.get(cat, set())
        if len(all_vals) > 1:
            for opp_val in all_vals:
                if opp_val != val and opp_val in claim.context_words and val not in claim.context_words:
                    return False, f"entity swap: claimed for '{opp_val}' but evidence is for '{val}'"

    # 2. Metric mismatch check: a percentage claim attributing a DTI value to a credit score
    if claim.claim_type in ("percentage", "percentage_points"):
        if grounding.leaf:
            leaf_lower = grounding.leaf.lower()
            if "dti" in leaf_lower or "debt" in leaf_lower:
                if ("credit" in claim.context_words or "score" in claim.context_words) and "dti" not in claim.context_words and "debt" not in claim.context_words:
                    return False, f"metric mismatch: percentage claim mentions credit/score but evidence is '{grounding.leaf}'"

    return True, ""


def _match_direct(
    claim: QuantitativeClaim,
    values: list[_GroundingValue],
    category_values: Mapping[str, set[str]],
) -> tuple[str, str] | None:
    for grounding in values:
        magnitude = abs(grounding.value) if grounding.signed_change and claim.direction else grounding.value
        if not _unit_matches(magnitude, grounding, claim):
            continue
        compatible, _ = _entity_and_metric_compatible(claim, grounding, category_values, values)
        if not compatible:
            continue
        label = grounding.leaf or "a number in evidence text"
        return grounding.source_id, f"matched {label}={grounding.value:g}"
    return None


def _comparable_groups(values: list[_GroundingValue]) -> list[list[_GroundingValue]]:
    """Groups of values it is meaningful to subtract or divide: same field
    name across the ledger, or same tool output."""
    by_leaf: dict[str, list[_GroundingValue]] = {}
    by_source: dict[str, list[_GroundingValue]] = {}
    for grounding in values:
        if not grounding.leaf or grounding.source_id.startswith(("policy:", "user_query")):
            continue
        by_leaf.setdefault(grounding.leaf, []).append(grounding)
        if grounding.is_ratio or grounding.is_percent:
            by_source.setdefault(grounding.source_id, []).append(grounding)
    groups = [*by_leaf.values(), *[g for g in by_source.values() if len(g) <= 6]]
    return [group for group in groups if 2 <= len(group) <= _MAX_DERIVATION_GROUP]


def _match_derived(
    claim: QuantitativeClaim,
    values: list[_GroundingValue],
    category_values: Mapping[str, set[str]],
) -> tuple[str, str] | None:
    if claim.claim_type not in ("percentage_points", "percentage"):
        return None
    for group in _comparable_groups(values):
        for index, first in enumerate(group):
            for second in group[index + 1:]:
                # Entities must not conflict between first and second
                conflict = False
                for cat in ("device", "account_id", "user_id"):
                    if cat in first.entity_context and cat in second.entity_context:
                        if first.entity_context[cat] != second.entity_context[cat]:
                            conflict = True
                            break
                if conflict:
                    continue

                c1, _ = _entity_and_metric_compatible(claim, first, category_values, values)
                c2, _ = _entity_and_metric_compatible(claim, second, category_values, values)
                if not c1 or not c2:
                    continue

                if claim.claim_type == "percentage_points":
                    # A percentage-point gap is only meaningful between two
                    # rates expressed in the same unit - never between counts.
                    same_unit = (first.is_ratio and second.is_ratio) or (first.is_percent and second.is_percent)
                    if not same_unit or first.signed_change or second.signed_change:
                        continue
                    gap = abs(first.value - second.value)
                    if _unit_matches(gap, first, claim):
                        note = " (direction not recoverable from unordered values)" if claim.direction else ""
                        return (
                            f"{first.source_id},{second.source_id}",
                            f"derived |{first.leaf} - {second.leaf}| = {gap:g}{note}",
                        )
                elif claim.direction is not None:
                    # Relative change ("conversion fell 43%"). Only between two
                    # values of the same unit class, never involving a value that
                    # is itself a change, and tried in both orders with the sign
                    # required to match - so a fall is never accepted as a rise.
                    if first.signed_change or second.signed_change:
                        continue
                    if (first.is_percent, first.is_ratio) != (second.is_percent, second.is_ratio):
                        continue
                    for base, new in ((first, second), (second, first)):
                        if base.value == 0:
                            continue
                        relative = (new.value - base.value) / base.value * 100.0
                        if _direction_consistent(relative, claim.direction) and _close(
                            abs(relative), claim.value, claim.tolerance
                        ):
                            return (
                                f"{base.source_id},{new.source_id}",
                                f"derived relative change {base.leaf}->{new.leaf} = {relative:+.3g}%",
                            )
    return None


def verify_numerical_grounding(
    answer: str,
    ledger: EvidenceLedger | None,
    *,
    query: str = "",
    reference_values: Mapping[str, float] | None = None,
) -> NumericalVerificationResult:
    """Check every quantitative claim in `answer` against evidence.

    `query` lets numbers the user supplied count as given; `reference_values`
    are declared policy thresholds (PolicyEngine.thresholds) so an answer may
    explain a rule ("exceeds the 43% ceiling") without being rejected.
    """
    claims = extract_quantitative_claims(answer)
    # Refusals get no exemption here. This used to return valid for any answer
    # containing a refusal phrase, so "Revenue was $9,999,999; the OS breakdown
    # is not available." skipped the check entirely. A genuine refusal has no
    # claims (or only numbers the user supplied), so it passes on its own.
    if not claims:
        return NumericalVerificationResult(valid=True, claims=claims)

    values = collect_grounding_values(ledger, query=query, reference_values=reference_values)
    verified: list[QuantitativeClaim] = []
    unverified: list[QuantitativeClaim] = []
    issues: list[str] = []

    category_values: dict[str, set[str]] = {}
    for g in values:
        for cat, val in g.entity_context.items():
            category_values.setdefault(cat, set()).add(val)

    for claim in claims:
        contradiction = _find_direction_contradiction(claim, values)
        match = None if contradiction else (
            _match_direct(claim, values, category_values) or _match_derived(claim, values, category_values)
        )
        if match is not None:
            evidence_id, explanation = match
            verified.append(QuantitativeClaim(
                raw_text=claim.raw_text,
                value=claim.value,
                claim_type=claim.claim_type,
                tolerance=claim.tolerance,
                direction=claim.direction,
                is_verified=True,
                matched_evidence_id=evidence_id,
                explanation=explanation,
            ))
            continue
        if contradiction:
            explanation = contradiction
        elif not values:
            explanation = "No evidence ledger present to substantiate this number."
        else:
            explanation = "No evidence value, stated input, policy threshold, or comparable derivation supports this value."
        unverified.append(QuantitativeClaim(
            raw_text=claim.raw_text,
            value=claim.value,
            claim_type=claim.claim_type,
            tolerance=claim.tolerance,
            direction=claim.direction,
            is_verified=False,
            explanation=explanation,
        ))
        issues.append(f"Unsubstantiated numerical claim '{claim.raw_text}' ({claim.value:g})")

    return NumericalVerificationResult(
        valid=not unverified,
        claims=verified + unverified,
        unverified_claims=unverified,
        issues=issues,
    )
