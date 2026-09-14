"""Evidence model and append-only Evidence Ledger.

WHAT
    Every meaningful tool execution (SQL query, document retrieval, statistical
    test, record lookup, calculation) produces one `EvidenceItem`. The
    `EvidenceLedger` collects them in insertion order for a single agent run.

WHY
    Everything downstream that claims to be "deterministic" - numerical
    verification (verification.py) and policy rules (policy.py, facts.py) -
    reads facts from this ledger rather than from anything the LLM wrote. The
    ledger is therefore the trust boundary of the whole runtime: the model can
    *propose* anything, but only what a tool actually returned becomes
    evidence.

    That only holds if evidence cannot be edited after the fact, so this
    module enforces immutability for real, not just by convention:

    - `EvidenceItem` is a frozen dataclass, AND its `data` payload is
      deep-frozen on construction (dicts become read-only `MappingProxyType`,
      lists become tuples). An earlier version was `frozen=True` but kept a
      plain `dict` inside, which `frozen` does not protect - `item.data["p"]
      = 0.9` silently succeeded, and `to_dict()` handed callers the live dict.
    - `EvidenceLedger.add()` freezes a *copy* of the caller's payload, so a
      tool that mutates its own result afterwards cannot rewrite history.
    - `to_dict()` returns a fresh, thawed (plain JSON-compatible) copy for
      export, never the internal structure.

TRADE-OFF
    Freezing costs a recursive copy per item. Ledgers here hold tens of items,
    so that is negligible; a high-throughput system would instead serialize
    items to an append-only store.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class EvidenceSource(str, Enum):
    """Where a piece of evidence came from (its provenance category)."""

    KNOWLEDGE_RAG = "knowledge_rag"
    ANALYTICS_SQL = "analytics_sql"
    STATISTICS = "statistics"
    POLICY_RULE = "policy_rule"
    # Document / chart extraction tools. In a production system this would be
    # a vision-language model or OCR pipeline; in this project the extraction
    # step is simulated over pre-extracted JSON fixtures (see
    # domains/multimodal_investigation/server.py), and the name is kept for the
    # role the tool plays, not a claim that a VLM runs.
    VISION_VLM = "vision_vlm"
    RECORD_LOOKUP = "record_lookup"
    CALCULATOR = "calculator"


class EvidenceKind(str, Enum):
    """What sort of fact an item is: raw observation, derived calculation, or
    retrieved text."""

    OBSERVATION = "observation"
    CALCULATION = "calculation"
    RETRIEVAL = "retrieval"


# Short, stable prefixes used to build human-readable evidence ids such as
# "ev-sql-001". Kept at module level so the mapping is defined exactly once.
_ID_PREFIX: dict[EvidenceSource, str] = {
    EvidenceSource.ANALYTICS_SQL: "sql",
    EvidenceSource.KNOWLEDGE_RAG: "rag",
    EvidenceSource.STATISTICS: "stat",
    EvidenceSource.POLICY_RULE: "pol",
    EvidenceSource.VISION_VLM: "vis",
    EvidenceSource.RECORD_LOOKUP: "rec",
    EvidenceSource.CALCULATOR: "calc",
}


def freeze(value: Any) -> Any:
    """Recursively convert containers into read-only equivalents.

    Mappings -> `MappingProxyType` (read-only view over a private dict copy),
    lists/tuples -> tuples, sets -> frozensets. Scalars (str, int, float, bool,
    None, dates) are already immutable and pass through unchanged.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    """Inverse of `freeze` for export: produce plain dicts/lists that
    `json.dumps` accepts and that a caller may modify without affecting the
    ledger (every container is a new object)."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source: EvidenceSource
    kind: EvidenceKind
    query_or_input: str
    # Read-only after construction - see __post_init__.
    data: Mapping[str, Any]
    provenance: str
    summary: str
    parent_evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # A frozen dataclass forbids normal attribute assignment, so the one
        # legitimate write (replacing the payload with its frozen copy) has to
        # go through object.__setattr__. This runs for items built directly as
        # well as through EvidenceLedger.add(), so no construction path can
        # smuggle a mutable payload in.
        object.__setattr__(self, "data", freeze(self.data))
        object.__setattr__(self, "parent_evidence_ids", tuple(self.parent_evidence_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source.value,
            "kind": self.kind.value,
            "query_or_input": self.query_or_input,
            "data": thaw(self.data),
            "provenance": self.provenance,
            "summary": self.summary,
            "parent_evidence_ids": list(self.parent_evidence_ids),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceItem":
        return cls(
            evidence_id=payload["evidence_id"],
            source=EvidenceSource(payload["source"]),
            kind=EvidenceKind(payload.get("kind", EvidenceKind.OBSERVATION.value)),
            query_or_input=payload["query_or_input"],
            data=payload.get("data", {}),
            provenance=payload.get("provenance", ""),
            summary=payload.get("summary", ""),
            parent_evidence_ids=tuple(payload.get("parent_evidence_ids", [])),
            timestamp=payload.get("timestamp", time.time()),
        )


class EvidenceLedger:
    """Append-only collection of evidence for one agent run.

    Items can be added and read, never modified or removed. Because each item
    is deeply immutable (see module docstring), returning the item objects
    themselves from the read methods is safe - there is nothing to defend by
    copying.
    """

    def __init__(self, run_id: str | None = None):
        self._run_id = run_id or f"run-{int(time.time() * 1000)}"
        self._items: list[EvidenceItem] = []
        self._index_by_id: dict[str, EvidenceItem] = {}
        # Highest sequence number issued per source. Tracked explicitly rather
        # than recounted from the item list: recounting made auto-generated ids
        # collide with caller-supplied ones (add id "ev-sql-002" manually, and
        # the next automatic SQL id was also computed as "ev-sql-002", raising
        # "already exists" mid-run).
        self._last_sequence: dict[EvidenceSource, int] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    def __len__(self) -> int:
        return len(self._items)

    def _next_evidence_id(self, source: EvidenceSource) -> str:
        prefix = _ID_PREFIX.get(source, "ev")
        sequence = self._last_sequence.get(source, 0)
        while True:
            sequence += 1
            candidate = f"ev-{prefix}-{sequence:03d}"
            if candidate not in self._index_by_id:
                self._last_sequence[source] = sequence
                return candidate

    def add(
        self,
        source: EvidenceSource,
        query_or_input: str,
        data: Mapping[str, Any],
        provenance: str,
        summary: str,
        kind: EvidenceKind = EvidenceKind.OBSERVATION,
        parent_evidence_ids: list[str] | tuple[str, ...] | None = None,
        evidence_id: str | None = None,
    ) -> EvidenceItem:
        if evidence_id is None:
            evidence_id = self._next_evidence_id(source)
        if evidence_id in self._index_by_id:
            raise ValueError(f"Evidence with id '{evidence_id}' already exists in ledger.")

        parents = tuple(parent_evidence_ids or ())
        # A derived calculation may only cite evidence that actually exists;
        # otherwise "provenance" would be an unverifiable claim.
        for parent_id in parents:
            if parent_id not in self._index_by_id:
                raise ValueError(f"Parent evidence '{parent_id}' does not exist in ledger.")

        item = EvidenceItem(
            evidence_id=evidence_id,
            source=source,
            kind=kind,
            query_or_input=query_or_input,
            data=data,  # frozen (copied) inside EvidenceItem.__post_init__
            provenance=provenance,
            summary=summary,
            parent_evidence_ids=parents,
        )
        self._items.append(item)
        self._index_by_id[item.evidence_id] = item
        return item

    def get_by_id(self, evidence_id: str) -> EvidenceItem | None:
        return self._index_by_id.get(evidence_id)

    def all(self) -> list[EvidenceItem]:
        """Items in insertion order. The list is a new object; the items are
        immutable, so neither can be used to alter the ledger."""
        return list(self._items)

    def find_by_source(self, source: EvidenceSource) -> list[EvidenceItem]:
        return [item for item in self._items if item.source == source]

    def find_by_kind(self, kind: EvidenceKind) -> list[EvidenceItem]:
        return [item for item in self._items if item.kind == kind]

    def extract_numerical_facts(self) -> dict[str, float]:
        """Flatten every numeric scalar in every item's payload.

        Keys are dotted paths prefixed with the evidence id, for example::

            {
                "ev-sql-001.rows[0].conversion_rate": 0.0293,
                "ev-stat-001.p_value": 0.014,
            }

        Booleans are skipped on purpose: `True` is an `int` in Python, and a
        flag like `significant=True` must never be mistaken for the number 1.
        This is a raw, unfiltered view; verification.py decides which of these
        numbers are meaningful enough to ground a claim.
        """
        facts: dict[str, float] = {}

        def _walk(path: str, value: Any) -> None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                facts[path] = float(value)
            elif isinstance(value, Mapping):
                for key, child in value.items():
                    _walk(f"{path}.{key}", child)
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    _walk(f"{path}[{index}]", child)

        for item in self._items:
            for key, value in item.data.items():
                _walk(f"{item.evidence_id}.{key}", value)
        return facts

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]
