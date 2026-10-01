from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, UploadFile, status

from app.api.dependencies import DocumentStoreDep, IngestionServiceDep, SettingsDep
from app.schemas.document import DocumentRecord

router = APIRouter(prefix="/documents", tags=["documents"])

PDF_SIGNATURE = b"%PDF-"


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    responses={status.HTTP_200_OK: {"model": DocumentRecord, "description": "Already uploaded"}},
)
def upload_document(
    file: UploadFile,
    response: Response,
    background_tasks: BackgroundTasks,
    service: IngestionServiceDep,
    settings: SettingsDep,
) -> DocumentRecord:
    """Upload a PDF. It is processed in the background; poll GET /documents/{id} for status."""
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = file.file.read(max_bytes + 1)

    if file.content_type != "application/pdf" or not content.startswith(PDF_SIGNATURE):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are supported")
    if len(content) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File exceeds {settings.max_upload_size_mb} MB",
        )

    document, needs_processing = service.register(file.filename or "document.pdf", content)
    if needs_processing:
        background_tasks.add_task(service.process, document.id)
    else:
        response.status_code = status.HTTP_200_OK
    return document


@router.get("")
def list_documents(documents: DocumentStoreDep) -> list[DocumentRecord]:
    return documents.list_all()


@router.get("/{document_id}")
def get_document(document_id: UUID, documents: DocumentStoreDep) -> DocumentRecord:
    document = documents.get(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return document
