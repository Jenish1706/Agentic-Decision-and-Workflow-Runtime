"""Deterministic test doubles - a scripted LLMClient so agent-loop tests
exercise real control flow (retries, termination, tool dispatch) without
needing Ollama running."""

from typing import TypeVar

from pydantic import BaseModel

from agent_runtime.llm.base import LLMClient, LLMError, Message

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient(LLMClient):
    def __init__(self, decisions: list | None = None, generations: list[str] | None = None):
        self._decisions = list(decisions or [])
        self._generations = list(generations or [])
        self.decide_calls: list[list[Message]] = []
        self.generate_calls: list[list[Message]] = []

    async def generate(self, messages: list[Message]) -> str:
        self.generate_calls.append(messages)
        if self._generations:
            return self._generations.pop(0)
        return "generated text"

    async def decide(self, messages: list[Message], schema: type[T], *, max_retries: int = 2) -> T:
        self.decide_calls.append(messages)
        if not self._decisions:
            raise AssertionError("FakeLLMClient ran out of scripted decisions")
        item = self._decisions.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def raises_llm_error() -> LLMError:
    return LLMError("scripted failure: model never produced valid JSON")
