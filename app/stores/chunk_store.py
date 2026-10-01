import logging
import re
from pathlib import Path
from threading import Lock
from uuid import UUID

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

logger = logging.getLogger(__name__)

CANDIDATES_PER_RETRIEVER = 20


class ChunkStore:
    """Embedded chunks in an in-memory vector store, persisted as a JSON file."""

    def __init__(self, path: Path, embeddings: Embeddings) -> None:
        self._path = path
        self._embeddings = embeddings
        self._lock = Lock()
        self._vector_store = self._read_file()

    def add(self, document_id: UUID, chunks: list[Document]) -> None:
        """Embed outside the lock, so searches are not blocked while a document is indexed."""
        for chunk in chunks:
            chunk.metadata["document_id"] = str(document_id)
        ids = [f"{document_id}:{index}" for index in range(len(chunks))]

        new_chunks = InMemoryVectorStore(self._embeddings)
        new_chunks.add_documents(chunks, ids=ids)

        with self._lock:
            self._vector_store.store.update(new_chunks.store)
            self._write_file()
        logger.info("Stored %d chunks for document %s", len(chunks), document_id)

    def search(self, query: str, k: int, document_id: UUID | None = None) -> list[Document]:
        """Hybrid search: semantic and BM25 keyword results merged with Reciprocal Rank Fusion."""
        with self._lock:
            chunks = self._chunks_of(document_id)
            if not chunks:
                return []

            semantic = self._vector_store.as_retriever(
                search_kwargs={
                    "k": CANDIDATES_PER_RETRIEVER,
                    "filter": lambda chunk: belongs_to(chunk, document_id),
                },
            )
            keyword = BM25Retriever.from_documents(
                chunks,
                k=CANDIDATES_PER_RETRIEVER,
                preprocess_func=tokenize,
            )
            hybrid = EnsembleRetriever(retrievers=[semantic, keyword], weights=[0.5, 0.5])
            results = hybrid.invoke(query)[:k]

        logger.info("Search over %d chunks returned %d results", len(chunks), len(results))
        return results

    def _chunks_of(self, document_id: UUID | None) -> list[Document]:
        all_chunks = self._vector_store.get_by_ids(list(self._vector_store.store))
        return [chunk for chunk in all_chunks if belongs_to(chunk, document_id)]

    def _read_file(self) -> InMemoryVectorStore:
        if not self._path.exists():
            return InMemoryVectorStore(self._embeddings)
        return InMemoryVectorStore.load(str(self._path), self._embeddings)

    def _write_file(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._vector_store.dump(str(self._path))


def belongs_to(chunk: Document, document_id: UUID | None) -> bool:
    return document_id is None or chunk.metadata["document_id"] == str(document_id)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
