import hashlib
import logging
import time
from pathlib import Path
from uuid import UUID

from app.rag.chunker import split_into_chunks
from app.rag.loader import load_pdf
from app.schemas.document import DocumentRecord, DocumentStatus
from app.stores.chunk_store import ChunkStore
from app.stores.document_store import DocumentStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        documents: DocumentStore,
        chunks: ChunkStore,
        uploads_dir: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._documents = documents
        self._chunks = chunks
        self._uploads_dir = uploads_dir
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def register(self, filename: str, content: bytes) -> tuple[DocumentRecord, bool]:
        """Save the upload and return (document, needs_processing). Identical files are reused."""
        sha256 = hashlib.sha256(content).hexdigest()
        existing = self._documents.find_by_hash(sha256)

        if existing and existing.status != DocumentStatus.FAILED:
            logger.info("Document %s already uploaded, skipping processing", existing.id)
            return existing, False

        if existing:
            document = existing.model_copy(
                update={"status": DocumentStatus.PROCESSING, "error": None}
            )
        else:
            document = DocumentRecord(filename=filename, sha256=sha256)

        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._file_path(document.id).write_bytes(content)
        self._documents.save(document)
        logger.info("Registered document %s (%d bytes)", document.id, len(content))
        return document, True

    def process(self, document_id: UUID) -> None:
        """Load, chunk and index a registered document. Runs as a background task."""
        document = self._documents.get(document_id)
        started = time.perf_counter()
        logger.info("Processing document %s", document_id)

        try:
            pages = load_pdf(self._file_path(document_id))
            chunks = split_into_chunks(pages, self._chunk_size, self._chunk_overlap)
            if not chunks:
                raise ValueError("No extractable text found in the PDF")
            self._chunks.add(document_id, chunks)
        except Exception as error:
            logger.exception("Processing document %s failed", document_id)
            failed = document.model_copy(
                update={"status": DocumentStatus.FAILED, "error": str(error)}
            )
            self._documents.save(failed)
            return

        ready = document.model_copy(
            update={
                "status": DocumentStatus.READY,
                "page_count": len(pages),
                "chunk_count": len(chunks),
            }
        )
        self._documents.save(ready)
        elapsed_s = time.perf_counter() - started
        logger.info("Document %s ready in %.1f s", document_id, elapsed_s)

    def fail_interrupted(self) -> None:
        """Mark documents left processing by a server restart as failed, so they can be retried."""
        for document in self._documents.list_all():
            if document.status == DocumentStatus.PROCESSING:
                interrupted = document.model_copy(
                    update={"status": DocumentStatus.FAILED, "error": "Interrupted by restart"}
                )
                self._documents.save(interrupted)
                logger.warning("Marked interrupted document %s as failed", document.id)

    def _file_path(self, document_id: UUID) -> Path:
        return self._uploads_dir / f"{document_id}.pdf"
