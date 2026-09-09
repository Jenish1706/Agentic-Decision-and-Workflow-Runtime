"""Paragraph-aware chunking.

Why chunk at all: embedding models and the LLM's context window both work
best over short, topically coherent spans - a whole multi-topic document
embeds into one vector that represents none of its parts well, and dumping
whole documents into the prompt wastes context on irrelevant sections.

Why paragraph-aware rather than fixed-width: splitting mid-sentence at a
fixed character offset regularly cuts a fact in half across two chunks,
which then both retrieve poorly and cite awkwardly. Packing whole paragraphs
into a chunk_size budget keeps each chunk's text coherent.

Why overlap: without it, a fact stated right at a chunk boundary can end up
with its first half in one chunk and second half in the next, and neither
chunk alone retrieves well for a query about that fact. Repeating the tail
of chunk N at the start of chunk N+1 gives boundary-spanning content a
chance to be whole in at least one chunk.
"""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buffer = ""

    def flush(next_para: str | None) -> str:
        """Emit `buffer` as a chunk and return the seed for the next buffer."""
        chunks.append(buffer)
        tail = buffer[-overlap:] if overlap > 0 else ""
        if next_para is None:
            return ""
        return f"{tail}\n\n{next_para}" if tail else next_para

    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        if buffer:
            buffer = flush(para if len(para) <= chunk_size else None)
            if len(para) <= chunk_size:
                continue
        # Paragraph alone exceeds chunk_size (or we just flushed with no
        # buffer to seed): hard-split it by characters as a fallback.
        if len(para) > chunk_size:
            step = chunk_size - overlap
            for i in range(0, len(para), step):
                chunks.append(para[i : i + chunk_size])
            buffer = ""

    if buffer:
        chunks.append(buffer)

    return chunks
