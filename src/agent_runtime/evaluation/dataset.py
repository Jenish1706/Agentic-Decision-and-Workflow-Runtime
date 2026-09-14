"""Loads hand-authored evaluation benchmarks: data/benchmark.json for the
knowledge base and domains/<name>/benchmark.json for each business domain.

The benchmarks are small (a few dozen questions) and hand-verified against
the data rather than large and synthetic: every expected answer and expected
number can be checked by reading the corpus or querying the database, which
matters more here than statistical power. The optional fields are used only
where they apply - `expected_doc_ids` for retrieval scoring,
`expected_numbers` for expected-number recall, `expected_policy_status` for
policy alignment."""

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
