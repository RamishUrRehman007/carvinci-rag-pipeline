from langchain_core.documents import Document

from app.rag.generator import PROMPT, format_sources


def test_prompt_contains_numbered_sources_and_question() -> None:
    chunks = [
        Document(page_content="3 Jahre keine Lackmängel", metadata={"page": 6}),
        Document(page_content="Pannenhilfe C901", metadata={"page": 81}),
    ]

    system, human = PROMPT.invoke(
        {"question": "Wie lange gilt die Lackgarantie?", "sources": format_sources(chunks)}
    ).to_messages()

    assert "[1] (page 6)\n3 Jahre keine Lackmängel" in system.content
    assert "[2] (page 81)\nPannenhilfe C901" in system.content
    assert human.content == "Wie lange gilt die Lackgarantie?"
