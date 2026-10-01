import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import documents, query
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.embeddings import create_embeddings
from app.rag.generator import create_generator
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService
from app.stores.chunk_store import ChunkStore
from app.stores.document_store import DocumentStore

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model and stores once at startup and share them across requests."""
    logger.info("Loading embedding model %s", settings.embedding_model)
    embeddings = create_embeddings(settings.embedding_model)

    document_store = DocumentStore(settings.data_dir / "documents.json")
    chunk_store = ChunkStore(settings.data_dir / "chunks.json", embeddings)

    app.state.document_store = document_store
    app.state.ingestion_service = IngestionService(
        documents=document_store,
        chunks=chunk_store,
        uploads_dir=settings.data_dir / "uploads",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    app.state.rag_service = RAGService(chunk_store, create_generator(settings))
    logger.info("Ready")
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(documents.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
