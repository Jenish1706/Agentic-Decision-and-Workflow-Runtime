import zlib

import numpy as np
import pytest

from agent_runtime.agent.memory.long_term import LongTermMemory, _MemoryWriteDecision
from agent_runtime.agent.memory.short_term import ShortTermMemory

from .fakes import FakeLLMClient


class _FakeEmbedder:
    """Deterministic bag-of-words embedder - just different enough for
    distinct texts to rank sensibly, without loading a real model.

    Buckets come from crc32, not `hash()`: Python randomizes string hashes
    per process, so hash-based buckets made similarity scores (and any test
    that depends on a score threshold) change from run to run."""

    dim = 64

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for word in text.lower().split():
                vec[zlib.crc32(word.encode()) % self.dim] += 1.0
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm > 0 else vec)
        return np.array(vectors, dtype=np.float32)


def test_short_term_memory_keeps_only_recent_turns():
    memory = ShortTermMemory(max_turns=2)
    memory.record("q1", "a1")
    memory.record("q2", "a2")
    memory.record("q3", "a3")
    turns = memory.recent_turns()
    assert len(turns) == 2
    assert [t.query for t in turns] == ["q2", "q3"]


def test_short_term_memory_clear():
    memory = ShortTermMemory()
    memory.record("q1", "a1")
    memory.clear()
    assert memory.recent_turns() == []


def test_short_term_memory_does_not_grow_without_bound():
    # Regression: turns were appended to a list forever and only the tail was read.
    memory = ShortTermMemory(max_turns=3)
    for index in range(1000):
        memory.record(f"q{index}", "a")
    assert len(memory._turns) == 3


@pytest.mark.asyncio
async def test_long_term_memory_ignores_unrelated_memories(tmp_path):
    # Regression: retrieve() returned the top-k memories however unrelated,
    # so every stored memory was injected into every prompt.
    llm = FakeLLMClient(decisions=[_MemoryWriteDecision(worth_remembering=True, fact="alpha beta gamma delta")])
    memory = LongTermMemory(embedder=_FakeEmbedder(), path=tmp_path / "memory.json", min_score=0.5)
    await memory.maybe_store(llm, "q", "a")
    assert memory.retrieve("alpha beta gamma delta")
    assert memory.retrieve("completely different words here") == []


@pytest.mark.asyncio
async def test_long_term_memory_skips_writes_when_not_worth_remembering(tmp_path):
    llm = FakeLLMClient(decisions=[_MemoryWriteDecision(worth_remembering=False, fact=None)])
    memory = LongTermMemory(embedder=_FakeEmbedder(), path=tmp_path / "memory.json")
    item = await memory.maybe_store(llm, "what's the weather", "It's sunny.")
    assert item is None
    assert memory.retrieve("weather") == []


@pytest.mark.asyncio
async def test_long_term_memory_stores_and_retrieves_durable_facts(tmp_path):
    llm = FakeLLMClient(
        decisions=[_MemoryWriteDecision(worth_remembering=True, fact="The user prefers concise answers.")]
    )
    path = tmp_path / "memory.json"
    memory = LongTermMemory(embedder=_FakeEmbedder(), path=path)
    item = await memory.maybe_store(llm, "please keep answers short", "Understood.")
    assert item is not None
    assert path.exists()

    hits = memory.retrieve("the user prefers concise answers")
    assert any("concise" in hit.text for hit in hits)


@pytest.mark.asyncio
async def test_long_term_memory_persists_across_instances(tmp_path):
    llm = FakeLLMClient(decisions=[_MemoryWriteDecision(worth_remembering=True, fact="User's team is Nimbus-Data.")])
    path = tmp_path / "memory.json"
    first = LongTermMemory(embedder=_FakeEmbedder(), path=path)
    await first.maybe_store(llm, "I'm on the data team", "Noted.")

    second = LongTermMemory(embedder=_FakeEmbedder(), path=path)
    hits = second.retrieve("what team is the user on")
    assert len(hits) == 1
    assert "Nimbus-Data" in hits[0].text
