from langchain_core.documents import Document

from app.rag.chunker import split_into_chunks


def page(number: int, text: str) -> Document:
    return Document(page_content=text, metadata={"page": number})


def test_chunks_carry_section_and_page() -> None:
    pages = [page(6, "# 1.1 Garantie\n\n3 Jahre keine Lackmängel.")]

    [chunk] = split_into_chunks(pages, chunk_size=800, chunk_overlap=0)

    assert chunk.metadata == {"page": 6, "section": "1.1 Garantie"}
    assert chunk.page_content == "1.1 Garantie\n\n3 Jahre keine Lackmängel."


def test_section_continues_on_next_page() -> None:
    pages = [
        page(6, "# 1.1 Garantie\n\n## Lack\n\n3 Jahre keine Lackmängel."),
        page(7, "Gilt für alle Volkswagen PKW."),
    ]

    chunks = split_into_chunks(pages, chunk_size=800, chunk_overlap=0)

    assert chunks[-1].metadata == {"page": 7, "section": "1.1 Garantie > Lack"}


def test_text_without_headings_has_empty_section() -> None:
    [chunk] = split_into_chunks([page(1, "Nur Fließtext.")], chunk_size=800, chunk_overlap=0)

    assert chunk.metadata["section"] == ""
    assert chunk.page_content == "Nur Fließtext."
