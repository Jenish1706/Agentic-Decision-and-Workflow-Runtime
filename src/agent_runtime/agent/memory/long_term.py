"""Long-term memory: persisted across process restarts, and retrieved by
similarity - deliberately reusing the exact same Embedder + VectorIndex
machinery as document RAG (rag/embeddings.py, rag/vector_index.py), just
over a different, much smaller collection. Memory and RAG are the same
retrieval mechanism applied to different content, not two different
systems; keeping that literally true in the code (not just true in
principle) is the point.

Write policy is the other half of what makes this "memory" rather than
"the whole conversation": most turns are not worth remembering. Each turn
is offered to an LLM call that must decide, explicitly, whether it contains
a durable, reusable fact (a stated preference or constraint) worth writing.
That call is the answer to "when should something be stored" - it is not
every message, and it is not a heuristic like "always store the last N
turns" (that's just short_term.py with extra steps).

Retrieval has a relevance floor (`min_score`). Without one, `retrieve`
returned the top-k memories however unrelated they were - with only a
couple of stored memories that meant ALL of them - and the loop injected
them into every prompt as "remembered facts about this user". The floor is
a cosine-similarity cut-off for the all-MiniLM-L6-v2 embeddings used here,
chosen from a small spot check rather than tuned: against two stored
memories, six unrelated questions scored 0.02-0.20, while two topical
questions scored 0.48-0.50 and a paraphrased preference ("keep your answers
short" vs "prefers concise answers") scored 0.27. Change it together with
the embedding model.

A known limit of similarity-retrieved memory: a general preference such as
"prefers concise answers" applies to every question but is only retrieved
by questions that resemble it.

There is no notion of *which* user a memory belongs to: the store is one
file for one local user. Anything multi-user (the FastAPI wrapper) must not
enable it - see api.py.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from ...llm.base import LLMClient, Message
from ...rag.embeddings import Embedder
from ...rag.vector_index import VectorIndex


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    text: str
    source_query: str


class _MemoryWriteDecision(BaseModel):
    worth_remembering: bool = Field(
        description="True only if this exchange states a durable fact, preference, or constraint "
        "worth recalling in a future, unrelated conversation."
    )
    fact: str | None = Field(default=None, description="The fact to store, written as a standalone sentence.")


_JUDGE_SYSTEM_PROMPT = (
    "You decide what belongs in long-term memory for an assistant. Only durable facts, "
    "preferences, or constraints stated by the user are worth remembering - not routine "
    "questions, not information that was merely looked up and reported back. "
    "When in doubt, prefer NOT storing it."
)


class LongTermMemory:
    def __init__(self, embedder: Embedder, path: Path, min_score: float = 0.25):
        self._embedder = embedder
        self._path = path
        self._min_score = min_score
        self._index = VectorIndex(dim=embedder.dim)
        self._items: dict[str, MemoryItem] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        items = [MemoryItem(**row) for row in raw]
        if not items:
            return
        vectors = self._embedder.embed([item.text for item in items])
        self._index.add([item.memory_id for item in items], vectors)
        self._items = {item.memory_id: item for item in items}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps([asdict(i) for i in self._items.values()], indent=2), encoding="utf-8")

    async def maybe_store(self, llm: LLMClient, query: str, answer: str) -> MemoryItem | None:
        messages = [
            Message(role="system", content=_JUDGE_SYSTEM_PROMPT),
            Message(role="user", content=f"User said: {query}\nAssistant answered: {answer}\n\nStore anything?"),
        ]
        decision = await llm.decide(messages, _MemoryWriteDecision)
        if not decision.worth_remembering or not decision.fact:
            return None

        memory_id = f"mem-{len(self._items)}"
        item = MemoryItem(memory_id=memory_id, text=decision.fact, source_query=query)
        vector = self._embedder.embed([item.text])
        self._index.add([item.memory_id], vector)
        self._items[item.memory_id] = item
        self._save()
        return item

    def retrieve(self, query: str, top_k: int = 2) -> list[MemoryItem]:
        if not self._items:
            return []
        query_vector = self._embedder.embed([query])[0]
        hits = self._index.search(query_vector, top_k)
        return [self._items[hit.chunk_id] for hit in hits if hit.score >= self._min_score]
