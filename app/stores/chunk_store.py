import logging
from pathlib import Path
from threading import Lock
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

logger = logging.getLogger(__name__)


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

    def count(self) -> int:
        with self._lock:
            return len(self._vector_store.store)

    def _read_file(self) -> InMemoryVectorStore:
        if not self._path.exists():
            return InMemoryVectorStore(self._embeddings)
        return InMemoryVectorStore.load(str(self._path), self._embeddings)

    def _write_file(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._vector_store.dump(str(self._path))
