"""Evidence-backed fact resolution for deterministic policy rules.

WHAT
    Helpers that answer "what does the evidence say about field X of entity
    Y?" - e.g. "what is the credit_score of account ACCT-102?" - and report
    honestly when the evidence does not settle the question.

WHY (the bug this replaces)
    The domain policy rules originally looked facts up like this:

        1. if the field is in proposal.parameters, use it;
        2. otherwise take the FIRST ledger number whose key contains the name.

    Both steps defeated the purpose of deterministic governance:

    - Step 1 let the LLM decide the facts its own proposal was judged on. A
      proposal carrying {"credit_score": 700} was approved even when the
      evidence said 600; {"override_authorized": true} let the model grant
      itself an emergency override.
    - Step 2 resolved facts for an arbitrary entity. After a query such as
      `SELECT * FROM accounts`, "the first credit_score" belonged to whichever
      account happened to be row 0, not the applicant.
    - A missing fact returned None, every rule treated None as "no
      violation", and so an agent that simply never looked anything up was
      approved.

HOW
    - Facts come from evidence only. Callers identify the *target* of a
      proposal (which account / invoice / service it acts on) from the
      proposal - that is legitimately the proposal's content - but every fact
      *about* that target is read from tool outputs.
    - Resolution is entity-scoped. While walking an evidence payload, any
      mapping that carries the entity key (e.g. "account_id") sets the scope
      for everything nested inside it, so a SQL row, an account profile, and a
      nested `credit_history` block are all attributed to the right entity.
    - Values that cannot be attributed to any entity (e.g. a filtered query
      that did not SELECT the id column) are used only when no attributed
      value exists, and the result is marked `attributed=False`.
    - The result is three-valued: FOUND, MISSING, or CONFLICT. Rules must
      decide what MISSING and CONFLICT mean for them; the convention in this
      project is to escalate (REQUIRES_APPROVAL), i.e. fail closed.

THE ASYMMETRY EVERY DOMAIN RULE FOLLOWS
    Evidence may RESTRICT a decision on weak grounds, but may PERMIT one only
    on strong grounds:

    - Any candidate value that violates a limit is enough to block or
      escalate - including one value out of a conflicting set, and including
      an unattributed value.
    - Allowing requires a `conclusive` fact: a single agreed value, attributed
      to the entity the proposal acts on.

    So a borrower's 600 credit score blocks an approval even if the query
    forgot to select account_id, but a 740 from such a query cannot approve
    anyone - it might belong to a different account.

TRADE-OFF / FAILURE MODE
    Scoping relies on tool outputs carrying entity identifiers. A tool that
    returns bare numbers with no id gives the resolver nothing to attribute,
    and rules then escalate rather than guess - safe, but it can mean more
    human approvals than strictly necessary.
"""

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .evidence import EvidenceLedger


class FactStatus(str, Enum):
    FOUND = "found"
    MISSING = "missing"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class ResolvedFact:
    """The outcome of looking one field up in evidence.

    `value` is set only when `status` is FOUND. `candidates` lists every
    distinct value that was considered (useful for audit messages when the
    status is CONFLICT), and `evidence_ids` records where they came from.
    `attributed` is True when those values were attributed to the requested
    entity (or when no entity was requested), False when they are the
    unattributed fallback.
    """

    field: str
    status: FactStatus
    value: Any = None
    candidates: tuple[Any, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    attributed: bool = False
    can_permit: bool = True

    @property
    def found(self) -> bool:
        return self.status is FactStatus.FOUND

    @property
    def conclusive(self) -> bool:
        """Strong enough to PERMIT a decision: one agreed value, attributed
        to the target entity, and backed by an authoritative source (not model-shaped
        SQL alone). See "THE ASYMMETRY" in the module docstring."""
        return self.found and self.attributed and self.can_permit

    def describe(self) -> str:
        """Short human-readable explanation for policy audit notes."""
        if self.status is FactStatus.FOUND:
            if not self.can_permit:
                return f"{self.field}={self.value!r} (free-form SQL, cannot autonomously permit)"
            return f"{self.field}={self.value!r}"
        if self.status is FactStatus.MISSING:
            return f"{self.field} is not established by any evidence"
        return f"{self.field} is contradictory in evidence ({list(self.candidates)!r})"


def normalize_identifier(value: Any) -> str:
    """Canonical form for comparing identifiers: 'ACCT-101 ' == 'acct-101'."""
    return str(value).strip().casefold()


def as_number(value: Any) -> float | None:
    """Strictly parse a number, returning None when the value is not one.

    Booleans are rejected (True must not become 1.0). Strings may carry
    thousands separators or a leading currency symbol / trailing percent
    sign, which is how tool outputs and LLM-written parameters commonly
    format them ("$5,000", "48%").
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").lstrip("$").rstrip("%").strip()
        try:
            number = float(cleaned)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


_TRUE_STRINGS = frozenset({"true", "yes", "y", "1"})
_FALSE_STRINGS = frozenset({"false", "no", "n", "0"})


def as_bool(value: Any) -> bool | None:
    """Strictly parse a boolean, returning None when the value is not one.

    `bool(value)` is the classic trap here: `bool("false")` is True, so an
    override flag serialized as the string "false" would have been read as an
    authorized override. Only explicit spellings are accepted.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().casefold()
        if token in _TRUE_STRINGS:
            return True
        if token in _FALSE_STRINGS:
            return False
    return None


def iter_scoped_mappings(
    ledger: EvidenceLedger | None, entity_key: str | None
) -> Iterator[tuple[str, str | None, Mapping[str, Any]]]:
    """Yield (evidence_id, scope, mapping) for every mapping in the ledger.

    `scope` is the normalized value of the nearest enclosing `entity_key`, or
    None when no enclosing mapping carries that key. A mapping that itself
    contains the key is scoped to its own value.
    """
    if ledger is None:
        return

    def _walk(node: Any, scope: str | None, evidence_id: str) -> Iterator[tuple[str, str | None, Mapping[str, Any]]]:
        if isinstance(node, Mapping):
            if entity_key is not None and node.get(entity_key) is not None:
                scope = normalize_identifier(node[entity_key])
            yield evidence_id, scope, node
            for child in node.values():
                yield from _walk(child, scope, evidence_id)
        elif isinstance(node, (list, tuple)):
            for child in node:
                yield from _walk(child, scope, evidence_id)

    for item in ledger.all():
        yield from _walk(item.data, None, item.evidence_id)


def _values_agree(first: Any, second: Any) -> bool:
    first_number, second_number = as_number(first), as_number(second)
    if first_number is not None and second_number is not None:
        return math.isclose(first_number, second_number, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(first, str) and isinstance(second, str):
        return normalize_identifier(first) == normalize_identifier(second)
    return first == second


def _distinct(values: list[Any]) -> list[Any]:
    distinct: list[Any] = []
    for value in values:
        if not any(_values_agree(value, seen) for seen in distinct):
            distinct.append(value)
    return distinct


def resolve_fact(
    ledger: EvidenceLedger | None,
    field: str,
    *,
    entity_key: str | None = None,
    entity_value: Any = None,
) -> ResolvedFact:
    """Resolve `field` for one entity from evidence, never from the proposal.

    With no entity given, every occurrence of the field counts. With an
    entity given, values attributed to that entity win; values attributed to
    *other* entities are ignored; unattributed values are a fallback used
    only if nothing is attributed to the target.
    """
    target = normalize_identifier(entity_value) if entity_value is not None else None
    attributed: list[tuple[Any, str]] = []
    unattributed: list[tuple[Any, str]] = []

    for evidence_id, scope, mapping in iter_scoped_mappings(ledger, entity_key):
        if field not in mapping or mapping[field] is None:
            continue
        pair = (mapping[field], evidence_id)
        if target is None:
            attributed.append(pair)
        elif scope == target:
            attributed.append(pair)
        elif scope is None:
            unattributed.append(pair)
        # else: the value belongs to a different entity - deliberately ignored.

    chosen = attributed or unattributed
    if not chosen:
        return ResolvedFact(field=field, status=FactStatus.MISSING)

    distinct = _distinct([value for value, _ in chosen])
    evidence_ids = tuple(dict.fromkeys(evidence_id for _, evidence_id in chosen))
    # Evidence can permit only when backed by permit-capable tools (pinned in client / domain manifest).
    # Ignore server-declared categories.
    items = [
        ledger.get_by_id(ev_id)
        for ev_id in evidence_ids
        if ledger is not None and ledger.get_by_id(ev_id) is not None
    ]
    can_permit = all(item.can_permit for item in items) if items else False

    return ResolvedFact(
        field=field,
        status=FactStatus.FOUND if len(distinct) == 1 else FactStatus.CONFLICT,
        value=distinct[0] if len(distinct) == 1 else None,
        candidates=tuple(distinct),
        evidence_ids=evidence_ids,
        attributed=bool(attributed),
        can_permit=can_permit,
    )


def numeric_candidates(fact: ResolvedFact) -> list[float]:
    """Every candidate value of `fact` that parses as a number. Rules check
    limits against ALL candidates, so one violating value among conflicting
    evidence still restricts the decision."""
    return [number for number in map(as_number, fact.candidates) if number is not None]


def find_entity_records(
    ledger: EvidenceLedger | None,
    *,
    entity_key: str,
    entity_value: Any,
    required_fields: tuple[str, ...],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Every evidence mapping about one entity that carries all
    `required_fields` - for rules that must reason about whole records, e.g.
    "each incident on this service", rather than a single field."""
    target = normalize_identifier(entity_value)
    records: list[tuple[str, Mapping[str, Any]]] = []
    for evidence_id, scope, mapping in iter_scoped_mappings(ledger, entity_key):
        if scope == target and all(name in mapping for name in required_fields):
            records.append((evidence_id, mapping))
    return records


def proposal_target(parameters: Mapping[str, Any], *keys: str) -> str | None:
    """The entity a proposal acts on, taken from its parameters.

    One of the two things rules legitimately read from the LLM's proposal:
    *which* account / invoice / service it wants to act on. Facts about that
    entity still come from evidence via `resolve_fact`.
    """
    for key in keys:
        value = parameters.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def proposal_number(parameters: Mapping[str, Any], *keys: str) -> float | None:
    """A quantity that defines the proposed action itself - the other thing a
    rule may read from the proposal (e.g. the traffic percentage of a
    rollout). Where evidence can also speak to the quantity, rules combine
    the two conservatively instead of trusting this value alone."""
    for key in keys:
        number = as_number(parameters.get(key))
        if number is not None:
            return number
    return None


def action_tokens(action: str) -> frozenset[str]:
    """Lower-case words of an action name: "Approve_Increase" -> {approve, increase}."""
    return frozenset(re.findall(r"[a-z]+", action.casefold()))


def is_governed_action(action: str, *, governed: frozenset[str], exempt: frozenset[str]) -> bool:
    """Whether a rule applies to `action`, deciding conservatively.

    An action is exempt only when it uses an exempt word and no governed
    word ("reject_increase" is still governed - it mentions an increase).
    Anything unrecognised is governed: rules used to match a short list of
    exact strings, so a new spelling such as "grant_limit_raise" skipped
    every rule and was allowed.
    """
    tokens = action_tokens(action)
    if tokens & governed:
        return True
    return not tokens & exempt
