"""Synthesizer: one free-form generation call combining whatever the
Retriever and Researcher workers gathered into a single cited answer. Not
an autonomous agent - it never decides to gather more evidence itself, it
only writes given what it's handed. If the Verifier rejects its draft, the
Supervisor calls it again with feedback (see supervisor.py); that retry
loop is what makes this an evaluator-optimizer pattern rather than a single
best-effort generation."""

from ..llm.base import LLMClient, Message

_SYSTEM_PROMPT = (
    "You combine retrieved evidence and tool results into one grounded, cited answer. "
    "Cite evidence using the exact [n] tags shown below. If nothing given to you answers "
    "the question, say so plainly instead of guessing."
)


class Synthesizer:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def synthesize(
        self,
        query: str,
        evidence_text: str,
        tool_observations: list[str],
        feedback: str | None = None,
    ) -> str:
        parts = [f"Question: {query}"]
        if evidence_text:
            parts.append(f"Retrieved evidence:\n{evidence_text}")
        if tool_observations:
            parts.append("Tool observations:\n" + "\n".join(tool_observations))
        if not evidence_text and not tool_observations:
            parts.append("(No evidence was retrieved and no tools were called.)")
        if feedback:
            parts.append(f"A previous draft had these problems - fix them: {feedback}")
        parts.append("Write the final answer now.")

        messages = [Message(role="system", content=_SYSTEM_PROMPT), Message(role="user", content="\n\n".join(parts))]
        return await self._llm.generate(messages)
