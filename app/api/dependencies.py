from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService
from app.stores.document_store import DocumentStore


def get_document_store(request: Request) -> DocumentStore:
    return request.app.state.document_store


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


SettingsDep = Annotated[Settings, Depends(get_settings)]
DocumentStoreDep = Annotated[DocumentStore, Depends(get_document_store)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
