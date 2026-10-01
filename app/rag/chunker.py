import logging
import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

HEADINGS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
MARKUP = re.compile(r"[*_]|</?\w+>")


def split_into_chunks(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split pages into size-limited chunks, each prefixed with its section title."""
    sections = split_into_sections(pages)

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = size_splitter.split_documents(sections)

    for chunk in chunks:
        if chunk.metadata["section"]:
            chunk.page_content = f"{chunk.metadata['section']}\n\n{chunk.page_content}"

    logger.info(
        "Split %d pages into %d sections and %d chunks",
        len(pages),
        len(sections),
        len(chunks),
    )
    return chunks


def split_into_sections(pages: list[Document]) -> list[Document]:
    """Split each page by markdown headings, carrying open headings over to the next page."""
    heading_splitter = MarkdownHeaderTextSplitter(HEADINGS)
    open_headings: dict[str, str] = {}
    sections = []

    for page in pages:
        text = headings_as_markdown(open_headings) + page.page_content
        parts = heading_splitter.split_text(text)

        for part in parts:
            section = Document(
                page_content=part.page_content,
                metadata={"page": page.metadata["page"], "section": section_title(part.metadata)},
            )
            sections.append(section)

        if parts:
            open_headings = parts[-1].metadata

    return sections


def headings_as_markdown(headings: dict[str, str]) -> str:
    lines = [f"{marker} {headings[level]}\n" for marker, level in HEADINGS if level in headings]
    return "".join(lines)


def section_title(headings: dict[str, str]) -> str:
    titles = []
    for _, level in HEADINGS:
        if level in headings:
            title = MARKUP.sub("", headings[level])
            titles.append(" ".join(title.split()))
    return " > ".join(titles)
