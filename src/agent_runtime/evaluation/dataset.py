"""Loads the hand-authored evaluation benchmark (data/benchmark.json).
See docs/SPEC.md section 22 for how it was constructed and why it's
small (~20 queries) and hand-verified rather than large and synthetic."""

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalQuery(BaseModel):
    id: str
    question: str
    category: str
    difficulty: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_record_ids: list[str] = Field(default_factory=list)
    expected_answer: str
    expected_tool: str | None = None
    expected_numbers: list[float] = Field(default_factory=list)
    expected_policy_status: str | None = None


def load_benchmark(path: Path) -> list[EvalQuery]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuery(**row) for row in rows]
