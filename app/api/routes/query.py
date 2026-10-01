from fastapi import APIRouter

from app.api.dependencies import RAGServiceDep
from app.schemas.query import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("")
def query_documents(request: QueryRequest, service: RAGServiceDep) -> QueryResponse:
    """Semantic search over uploaded documents, plus a cited answer when an LLM is configured."""
    return service.query(request)
