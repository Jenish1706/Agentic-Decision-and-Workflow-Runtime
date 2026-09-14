"""Ollama-backed LLMClient.

Structured decisions use Ollama's `format` parameter with a JSON Schema
(schema-constrained decoding), not Ollama's separate native tool-calling
API. This is a deliberate choice: the agent loop is hand-built (this
project's whole point), so the LLM is used purely as a "produce text
constrained to this shape" primitive. Native tool-calling would hand
decision-making back to a provider-specific mechanism this project is
supposed to demonstrate building from scratch.

Only schema failures become `LLMError`. Transport errors (Ollama not
running, connection dropped) propagate unchanged on purpose: the agent loop
treats LLMError as "the model produced unusable output" and ends the run
gracefully with an apology, so converting a dead server into LLMError would
turn an infrastructure outage into a benchmark full of plausible-looking
agent failures instead of a visible crash.
"""

import ollama
from pydantic import ValidationError

from .base import LLMClient, LLMError, Message, T


class OllamaClient(LLMClient):
    def __init__(self, model: str, host: str):
        self._model = model
        self._client = ollama.AsyncClient(host=host)

    async def generate(self, messages: list[Message]) -> str:
        response = await self._client.chat(
            model=self._model,
            messages=[m.model_dump() for m in messages],
        )
        return response["message"]["content"]

    async def decide(self, messages: list[Message], schema: type[T], *, max_retries: int = 2) -> T:
        schema_json = schema.model_json_schema()
        working = list(messages)
        last_error: Exception | None = None

        for _ in range(max_retries + 1):
            response = await self._client.chat(
                model=self._model,
                messages=[m.model_dump() for m in working],
                format=schema_json,
            )
            raw = response["message"]["content"]
            try:
                return schema.model_validate_json(raw)
            except ValidationError as exc:  # also raised for malformed JSON
                last_error = exc
                working = working + [
                    Message(role="assistant", content=raw),
                    Message(
                        role="user",
                        content=(
                            "That response did not satisfy the required schema "
                            f"({exc}). Reply again with ONLY a JSON object matching the schema, "
                            "no extra text."
                        ),
                    ),
                ]

        raise LLMError(
            f"Model failed to produce schema-valid output for {schema.__name__} "
            f"after {max_retries + 1} attempt(s): {last_error}"
        )
