import pytest

from agent_runtime.rag.chunking import chunk_text


def test_empty_text_produces_no_chunks():
    assert chunk_text("", chunk_size=100, overlap=10) == []


def test_short_text_stays_one_chunk():
    text = "This is a short paragraph that fits easily in one chunk."
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert chunks == [text]


def test_multiple_paragraphs_split_by_size():
    paragraphs = [f"Paragraph {i} " + "word " * 15 for i in range(6)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 120 + 20  # a little slack: overlap tail + one paragraph


def test_overlap_repeats_tail_of_previous_chunk():
    paragraphs = ["A" * 80, "B" * 80, "C" * 80]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, chunk_size=100, overlap=30)
    assert len(chunks) >= 2
    # the tail of chunk N should reappear at the start of chunk N+1
    tail_of_first = chunks[0][-30:]
    assert tail_of_first in chunks[1]


def test_single_paragraph_longer_than_chunk_size_is_hard_split():
    long_paragraph = "word " * 200  # ~1000 chars, no paragraph breaks
    chunks = chunk_text(long_paragraph, chunk_size=150, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 150 for c in chunks)
    # no content lost: concatenating without overlap-duplication should recover the length order of magnitude
    assert sum(len(c) for c in chunks) >= len(long_paragraph)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=100, overlap=-1)
