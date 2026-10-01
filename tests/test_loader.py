from langchain_core.documents import Document

from app.rag.loader import clean_text, find_boilerplate_lines

FOOTER = "30.04.2025 Volkswagen AG   Nachdruck nicht gestattet   intern **{page}**"


def make_pages(bodies: list[str]) -> list[Document]:
    return [
        Document(page_content=f"{body}\n{FOOTER.format(page=number)}\nINTERNAL")
        for number, body in enumerate(bodies, start=1)
    ]


def test_footer_with_changing_page_number_is_boilerplate() -> None:
    pages = make_pages(["Lack", "Rost", "Batterie", "Reifen"])

    boilerplate = find_boilerplate_lines(pages)

    cleaned = [clean_text(page.page_content, boilerplate) for page in pages]
    assert cleaned == ["Lack", "Rost", "Batterie", "Reifen"]


def test_short_documents_have_no_boilerplate() -> None:
    pages = make_pages(["Lack", "Rost"])

    assert find_boilerplate_lines(pages) == set()


def test_clean_text_drops_table_of_contents_lines() -> None:
    text = "1.1 Garantiepaket ........................ 6\nEchter Inhalt"

    assert clean_text(text, boilerplate=set()) == "Echter Inhalt"
