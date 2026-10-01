from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str
    sha256: str
    status: DocumentStatus = DocumentStatus.PROCESSING
    page_count: int | None = None
    chunk_count: int | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
