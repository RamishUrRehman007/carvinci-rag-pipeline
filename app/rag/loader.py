import logging
import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

logger = logging.getLogger(__name__)

BOILERPLATE_PAGE_RATIO = 0.5
TOC_DOTS = re.compile(r"\.{5,}")
PICTURE_MARKER = re.compile(r"<!-- (Start|End) of picture text -->")
MARKUP = re.compile(r"[*_#>`]|</?\w+>")
DIGITS = re.compile(r"\d+")
EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


def load_pdf(path: Path) -> list[Document]:
    pages = PyMuPDF4LLMLoader(path, mode="page").load()
    boilerplate = find_boilerplate_lines(pages)

    cleaned_pages = []
    for page in pages:
        text = clean_text(page.page_content, boilerplate)
        if text:
            page_number = page.metadata["page"] + 1
            cleaned_pages.append(Document(page_content=text, metadata={"page": page_number}))

    logger.info(
        "Loaded %d pages, removed %d boilerplate lines, kept %d pages",
        len(pages),
        len(boilerplate),
        len(cleaned_pages),
    )
    return cleaned_pages


def find_boilerplate_lines(pages: list[Document]) -> set[str]:
    pages_per_line = Counter()
    for page in pages:
        unique_lines = {normalize(line) for line in page.page_content.splitlines()}
        pages_per_line.update(unique_lines)

    min_pages = len(pages) * BOILERPLATE_PAGE_RATIO
    return {line for line, count in pages_per_line.items() if line and count > min_pages}


def clean_text(text: str, boilerplate: set[str]) -> str:
    text = PICTURE_MARKER.sub("", text)

    kept_lines = []
    for line in text.splitlines():
        is_boilerplate = normalize(line) in boilerplate
        is_table_of_contents = TOC_DOTS.search(line) is not None
        if not is_boilerplate and not is_table_of_contents:
            kept_lines.append(line)

    return EXTRA_BLANK_LINES.sub("\n\n", "\n".join(kept_lines)).strip()


def normalize(line: str) -> str:
    """Strip markdown and mask digits so lines that differ only by page number match."""
    without_markup = MARKUP.sub("", line)
    return DIGITS.sub("#", without_markup).strip()
