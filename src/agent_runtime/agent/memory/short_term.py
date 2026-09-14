"""Short-term memory: recent conversation turns for THIS session, kept
in-process only. This is not persisted to disk and not retrieved by
similarity - it's just the last N (query, answer) pairs, injected into the
prompt verbatim so the agent can resolve things like "it" or "that service"
across turns. Contrast with long_term.py, which is retrieval-backed and
persisted.

Turns are held in a bounded deque. The previous version appended to a plain
list and only *read* the last N, so a long-lived process (the FastAPI
wrapper) grew its memory without limit.

Callers answering unrelated questions must `clear()` between them (see
AgentRuntime.reset_session): recent turns are injected into every prompt, so
leftover turns from another question are contamination, not context.
"""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    query: str
    answer: str


class ShortTermMemory:
    def __init__(self, max_turns: int = 6):
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def recent_turns(self) -> list[Turn]:
        return list(self._turns)

    def record(self, query: str, answer: str) -> None:
        self._turns.append(Turn(query=query, answer=answer))

    def clear(self) -> None:
        self._turns.clear()
