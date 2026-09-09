"""Document loading. Documents are plain markdown files with a tiny flat
frontmatter block (doc_id/title/category/tags) - deliberately not YAML: the
frontmatter here is four flat key:value lines, so a real parser would be
one more dependency for zero benefit."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    category: str
    tags: list[str]
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    position: int
    text: str


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Document is missing the '---' frontmatter block")
    fields: dict[str, str] = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        key, _, value = lines[i].partition(":")
        fields[key.strip()] = value.strip()
        i += 1
    body = "\n".join(lines[i + 1 :]).strip()
    return fields, body


def load_documents(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        fields, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        tags = [t.strip() for t in fields.get("tags", "").split(",") if t.strip()]
        documents.append(
            Document(
                doc_id=fields["doc_id"],
                title=fields["title"],
                category=fields.get("category", "uncategorized"),
                tags=tags,
                text=body,
            )
        )
    return documents
