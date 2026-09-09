from pathlib import Path

from agent_runtime.rag.documents import load_documents


def test_load_documents_parses_frontmatter_and_body(tmp_path: Path):
    (tmp_path / "sample.md").write_text(
        "---\n"
        "doc_id: sample-doc\n"
        "title: Sample Document\n"
        "category: test\n"
        "tags: alpha, beta\n"
        "---\n"
        "This is the body text.\n",
        encoding="utf-8",
    )
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "sample-doc"
    assert doc.title == "Sample Document"
    assert doc.category == "test"
    assert doc.tags == ["alpha", "beta"]
    assert doc.text == "This is the body text."


def test_load_documents_is_sorted_by_filename(tmp_path: Path):
    for name, doc_id in [("b.md", "second"), ("a.md", "first")]:
        (tmp_path / name).write_text(f"---\ndoc_id: {doc_id}\ntitle: T\ncategory: c\ntags:\n---\nbody\n", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.doc_id for d in docs] == ["first", "second"]
