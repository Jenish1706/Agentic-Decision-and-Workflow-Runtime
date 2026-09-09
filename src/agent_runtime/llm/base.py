"""Provider-agnostic LLM interface.

Two methods only: `generate` (free-form text, used for final synthesis) and
`decide` (schema-constrained structured output, used everywhere the agent
loop needs a machine-parseable decision). Keeping the surface this small is
deliberate - it's the seam that would let a second backend (an API model)
be dropped in later without touching agent/rag code, without building
adapter machinery this project doesn't yet need with one backend.
"""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Message(BaseModel):
    role: str
    content: str


class LLMError(RuntimeError):
    """Raised when the backend cannot produce usable output after retries.

    This is a real, expected failure mode for small local models asked for
    structured output - not an edge case to paper over. Callers are expected
    to catch it and degrade gracefully (see agent/loop.py).
    """


class LLMClient(ABC):
    @abstractmethod
    async def generate(self, messages: list[Message]) -> str:
        """Free-form text completion."""

    @abstractmethod
    async def decide(self, messages: list[Message], schema: type[T], *, max_retries: int = 2) -> T:
        """Return output parsed and validated against `schema`.

        Implementations should use the backend's constrained-decoding
        capability where available (grammar/JSON-schema constrained
        sampling), then still validate: constrained decoding guarantees
        syntactic JSON matching the schema's shape, not that the values are
        semantically sensible (e.g. a tool name that doesn't exist). Retries
        feed the validation error back to the model as corrective feedback.
        """
