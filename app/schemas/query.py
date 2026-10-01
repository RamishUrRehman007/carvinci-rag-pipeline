from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: UUID | None = None


class Source(BaseModel):
    document_id: UUID
    page: int
    section: str
    content: str


class QueryResponse(BaseModel):
    answer: str | None = Field(description="Null when no LLM is configured or nothing was found")
    sources: list[Source] = Field(description="Ranked chunks; [n] in the answer refers to n-th")
