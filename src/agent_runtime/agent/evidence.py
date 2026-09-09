"""Evidence Model and Append-Only Evidence Ledger.

Every meaningful tool execution (SQL query, RAG retrieval, statistical test,
record lookup, policy evaluation) produces an immutable `EvidenceItem`.
The `EvidenceLedger` tracks raw observations, derived calculations, and retrieval
results in an append-only structure that cannot be mutated retroactively by an LLM.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class EvidenceSource(str, Enum):
    KNOWLEDGE_RAG = "knowledge_rag"
    ANALYTICS_SQL = "analytics_sql"
    STATISTICS = "statistics"
    POLICY_RULE = "policy_rule"
    VISION_VLM = "vision_vlm"
    RECORD_LOOKUP = "record_lookup"


class EvidenceKind(str, Enum):
    OBSERVATION = "observation"
    CALCULATION = "calculation"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source: EvidenceSource
    kind: EvidenceKind
    query_or_input: str
    data: dict[str, Any]
    provenance: str
    summary: str
    parent_evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source.value,
            "kind": self.kind.value,
            "query_or_input": self.query_or_input,
            "data": self.data,
            "provenance": self.provenance,
            "summary": self.summary,
            "parent_evidence_ids": list(self.parent_evidence_ids),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceItem":
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
    """Append-only ledger for evidence items.

    Public interface does not expose mutable internal storage.
    Once an item is added, it cannot be modified or deleted.
    """

    def __init__(self, run_id: str | None = None):
        self._run_id = run_id or f"run-{int(time.time() * 1000)}"
        self._items: list[EvidenceItem] = []
        self._index_by_id: dict[str, EvidenceItem] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        source: EvidenceSource,
        query_or_input: str,
        data: dict[str, Any],
        provenance: str,
        summary: str,
        kind: EvidenceKind = EvidenceKind.OBSERVATION,
        parent_evidence_ids: list[str] | tuple[str, ...] | None = None,
        evidence_id: str | None = None,
    ) -> EvidenceItem:
        if evidence_id is None:
            # Prefixed counter, e.g. ev-sql-001 or ev-rag-002
            prefix_map = {
                EvidenceSource.ANALYTICS_SQL: "sql",
                EvidenceSource.KNOWLEDGE_RAG: "rag",
                EvidenceSource.STATISTICS: "stat",
                EvidenceSource.POLICY_RULE: "pol",
                EvidenceSource.VISION_VLM: "vis",
                EvidenceSource.RECORD_LOOKUP: "rec",
            }
            prefix = prefix_map.get(source, "ev")
            count = len([it for it in self._items if it.source == source]) + 1
            evidence_id = f"ev-{prefix}-{count:03d}"

        if evidence_id in self._index_by_id:
            raise ValueError(f"Evidence with id '{evidence_id}' already exists in ledger.")

        parents = tuple(parent_evidence_ids) if parent_evidence_ids else tuple()
        # Verify parents exist if specified
        for pid in parents:
            if pid not in self._index_by_id:
                raise ValueError(f"Parent evidence '{pid}' does not exist in ledger.")

        item = EvidenceItem(
            evidence_id=evidence_id,
            source=source,
            kind=kind,
            query_or_input=query_or_input,
            data=dict(data),
            provenance=provenance,
            summary=summary,
            parent_evidence_ids=parents,
        )
        self._items.append(item)
        self._index_by_id[item.evidence_id] = item
        return item

    record = add

    def get_by_id(self, evidence_id: str) -> EvidenceItem | None:
        return self._index_by_id.get(evidence_id)

    def all(self) -> list[EvidenceItem]:
        """Returns a shallow copy of items in insertion order."""
        return list(self._items)

    def find_by_source(self, source: EvidenceSource) -> list[EvidenceItem]:
        return [it for it in self._items if it.source == source]

    def find_by_kind(self, kind: EvidenceKind) -> list[EvidenceItem]:
        return [it for it in self._items if it.kind == kind]

    def extract_numerical_facts(self) -> dict[str, float]:
        """Flattens all numeric scalars across evidence for verification.

        Returns keys mapped to floats. Example:
        {
            "ev-sql-001.mobile_conversion": 0.184,
            "ev-sql-001.desktop_conversion": 0.421,
            "ev-stat-001.p_value": 0.003
        }
        """
        facts: dict[str, float] = {}

        def _walk(prefix: str, val: Any) -> None:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                facts[prefix] = float(val)
            elif isinstance(val, dict):
                for k, v in val.items():
                    _walk(f"{prefix}.{k}", v)
            elif isinstance(val, (list, tuple)):
                for idx, v in enumerate(val):
                    _walk(f"{prefix}[{idx}]", v)

        for it in self._items:
            for k, v in it.data.items():
                _walk(f"{it.evidence_id}.{k}", v)

        return facts

    def to_list(self) -> list[dict[str, Any]]:
        return [it.to_dict() for it in self._items]
