from pathlib import Path
from threading import Lock
from uuid import UUID

from pydantic import TypeAdapter

from app.schemas.document import DocumentRecord

DocumentList = TypeAdapter(list[DocumentRecord])


class DocumentStore:
    """Document metadata and processing status, persisted as a JSON file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()
        self._documents = {document.id: document for document in self._read_file()}

    def save(self, document: DocumentRecord) -> None:
        with self._lock:
            self._documents[document.id] = document
            self._write_file()

    def get(self, document_id: UUID) -> DocumentRecord | None:
        return self._documents.get(document_id)

    def find_by_hash(self, sha256: str) -> DocumentRecord | None:
        for document in self.list_all():
            if document.sha256 == sha256:
                return document
        return None

    def list_all(self) -> list[DocumentRecord]:
        with self._lock:
            documents = list(self._documents.values())
        return sorted(documents, key=lambda document: document.created_at, reverse=True)

    def _read_file(self) -> list[DocumentRecord]:
        if not self._path.exists():
            return []
        return DocumentList.validate_json(self._path.read_bytes())

    def _write_file(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(DocumentList.dump_json(list(self._documents.values()), indent=2))
