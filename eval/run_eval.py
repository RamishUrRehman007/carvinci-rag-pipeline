"""Measure retrieval quality and latency on the sample handbook.

Indexes the PDF into a temporary directory, runs every question in questions.json through the
same RAGService the API uses and reports hit@k, MRR and latency. With GOOGLE_API_KEY set, the
generated answers and their token usage are shown too.

Usage: uv run python -m eval.run_eval [path/to/document.pdf]
"""

import json
import logging
import statistics
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.embeddings import create_embeddings
from app.rag.generator import create_generator
from app.schemas.query import QueryRequest
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService
from app.stores.chunk_store import ChunkStore
from app.stores.document_store import DocumentStore

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
DEFAULT_PDF = Path("Gewährleistungshandbuch.pdf")
TOP_K = 5


def main() -> None:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    questions = json.loads(QUESTIONS_FILE.read_text())
    settings = get_settings()

    setup_logging("WARNING")
    logging.getLogger("app.rag.generator").setLevel(logging.INFO)

    with TemporaryDirectory() as data_dir:
        print(f"Indexing {pdf_path.name} ...")
        embeddings = create_embeddings(settings.embedding_model)
        documents = DocumentStore(Path(data_dir) / "documents.json")
        chunks = ChunkStore(Path(data_dir) / "chunks.json", embeddings)
        ingestion = IngestionService(
            documents,
            chunks,
            Path(data_dir) / "uploads",
            settings.chunk_size,
            settings.chunk_overlap,
        )
        document, _ = ingestion.register(pdf_path.name, pdf_path.read_bytes())
        ingestion.process(document.id)

        rag = RAGService(chunks, create_generator(settings))
        rag.query(QueryRequest(question="warm up the embedding model"))
        results = [evaluate(rag, question) for question in questions]

    print_report(results)


def evaluate(rag: RAGService, question: dict) -> dict:
    started = time.perf_counter()
    response = rag.query(QueryRequest(question=question["question"], top_k=TOP_K))
    latency_ms = (time.perf_counter() - started) * 1000

    pages = [source.page for source in response.sources]
    rank = next(
        (index for index, page in enumerate(pages, start=1) if page in question["expected_pages"]),
        None,
    )
    return {
        **question,
        "rank": rank,
        "pages": pages,
        "latency_ms": latency_ms,
        "answer": response.answer,
    }


def print_report(results: list[dict]) -> None:
    print(f"\n{'type':<13}{'rank':>5}{'ms':>7}  question -> retrieved pages")
    for result in results:
        rank = "-" if result["rank"] is None else result["rank"]
        print(
            f"{result['type']:<13}{rank:>5}{result['latency_ms']:>7.0f}  "
            f"{result['question'][:60]} -> {result['pages']}"
        )
        if result["answer"]:
            print(f"{'':<25}answer: {' '.join(result['answer'].split())[:150]}")

    scored = [result for result in results if result["expected_pages"]]
    hits = [result for result in scored if result["rank"] is not None]
    mrr = sum(1 / result["rank"] for result in hits) / len(scored)
    latencies = [result["latency_ms"] for result in results]

    print(f"\nhit@{TOP_K}: {len(hits)}/{len(scored)} ({len(hits) / len(scored):.0%})")
    print(f"MRR:    {mrr:.2f}")
    print(f"latency: median {statistics.median(latencies):.0f} ms, max {max(latencies):.0f} ms")


if __name__ == "__main__":
    main()
