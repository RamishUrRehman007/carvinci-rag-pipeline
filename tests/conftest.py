import zlib
from collections.abc import Iterator
from itertools import repeat
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.api.dependencies import get_document_store, get_ingestion_service, get_rag_service
from app.main import app
from app.rag.generator import AnswerGenerator
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService
from app.stores.chunk_store import ChunkStore, tokenize
from app.stores.document_store import DocumentStore

FAKE_ANSWER = "Lackmängel sind 3 Jahre abgedeckt [1]."


class WordCountEmbeddings(Embeddings):
    """Bag-of-words vectors: texts sharing words are similar. Meaningful ranking without a model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * 256
        for word in tokenize(text):
            vector[zlib.crc32(word.encode()) % 256] += 1
        return vector


def make_pdf(pages: list[tuple[str, str]]) -> bytes:
    """Build a PDF in memory; each page is (heading, body)."""
    document = pymupdf.open()
    for heading, body in pages:
        page = document.new_page()
        page.insert_text((72, 72), heading, fontsize=20)
        page.insert_text((72, 110), body, fontsize=11)
    return document.tobytes()


@pytest.fixture
def warranty_pdf() -> bytes:
    return make_pdf(
        [
            ("1.1 Garantie", "3 Jahre keine Lackmaengel an der Karosserie."),
            ("1.2 Batterie", "8 Jahre Garantie auf die Hochvoltbatterie."),
            ("1.3 Mobilitaet", "Pannenhilfe wird mit KD-Nr. C901 abgerechnet."),
        ]
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """API with the real pipeline and stores; only the embedding model and the LLM are replaced."""
    documents = DocumentStore(tmp_path / "documents.json")
    chunks = ChunkStore(tmp_path / "chunks.json", WordCountEmbeddings())
    ingestion = IngestionService(documents, chunks, tmp_path / "uploads", 800, 150)
    llm = GenericFakeChatModel(messages=repeat(AIMessage(content=FAKE_ANSWER)))
    rag = RAGService(chunks, AnswerGenerator(llm))

    app.dependency_overrides = {
        get_document_store: lambda: documents,
        get_ingestion_service: lambda: ingestion,
        get_rag_service: lambda: rag,
    }
    yield TestClient(app)
    app.dependency_overrides.clear()
