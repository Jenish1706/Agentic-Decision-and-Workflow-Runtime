"""Assembles retrieved chunks into a citation-tagged context block.

The citation tag ([1], [2], ...) is the load-bearing contract between
retrieval and generation: the system prompt instructs the model to cite
only tags that appear here, and agent/verification.py checks that promise
was kept. Tag numbering continues across multiple retrieval calls within
one run (via `start_index`) so earlier and later evidence never collide on
the same tag.
"""

from .documents import Chunk


def build_context(retrieved: list[tuple[Chunk, float]], start_index: int = 1) -> tuple[str, dict[str, Chunk]]:
    """retrieved: (chunk, score) pairs, already ranked best-first.
    Returns (context_text, {tag: chunk})."""
    parts: list[str] = []
    citation_map: dict[str, Chunk] = {}
    for offset, (chunk, _score) in enumerate(retrieved):
        tag = f"[{start_index + offset}]"
        citation_map[tag] = chunk
        parts.append(f"{tag} (source: {chunk.doc_title})\n{chunk.text}")
    return "\n\n".join(parts), citation_map
