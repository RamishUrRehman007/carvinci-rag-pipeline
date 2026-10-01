import logging
import time

from app.rag.generator import AnswerGenerator
from app.schemas.query import QueryRequest, QueryResponse, Source
from app.stores.chunk_store import ChunkStore

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, chunks: ChunkStore, generator: AnswerGenerator | None) -> None:
        self._chunks = chunks
        self._generator = generator

    def query(self, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        chunks = self._chunks.search(request.question, request.top_k, request.document_id)

        answer = None
        if self._generator and chunks:
            try:
                answer = self._generator.generate(request.question, chunks)
            except Exception as error:
                reason = getattr(error, "status", None) or type(error).__name__
                logger.warning("Skipped answer generation (%s), returning sources only", reason)

        sources = [
            Source(
                document_id=chunk.metadata["document_id"],
                page=chunk.metadata["page"],
                section=chunk.metadata["section"],
                content=chunk.page_content,
            )
            for chunk in chunks
        ]

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Query handled in %.0f ms (%d sources, answer generated: %s)",
            elapsed_ms,
            len(sources),
            answer is not None,
        )
        return QueryResponse(answer=answer, sources=sources)
