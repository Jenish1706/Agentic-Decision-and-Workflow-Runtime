"""Short-term memory: recent conversation turns for THIS session, kept
in-process only. This is not persisted to disk and not retrieved by
similarity - it's just the last N (query, answer) pairs, injected into the
prompt verbatim so the agent can resolve things like "it" or "that service"
across turns. Contrast with long_term.py, which is retrieval-backed and
persisted."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    query: str
    answer: str


class ShortTermMemory:
    def __init__(self, max_turns: int = 6):
        self._turns: list[Turn] = []
        self._max_turns = max_turns

    def recent_turns(self) -> list[Turn]:
        return list(self._turns[-self._max_turns :])

    def record(self, query: str, answer: str) -> None:
        self._turns.append(Turn(query=query, answer=answer))

    def clear(self) -> None:
        self._turns.clear()
