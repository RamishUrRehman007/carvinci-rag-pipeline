# PDF RAG Pipeline

Upload PDF documents and ask questions about them in plain language.
You get the most relevant passages back, plus a short answer with citations if a Gemini API key is set.

Built with FastAPI, Pydantic and LangChain. Everything runs locally.

## Quickstart

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.13 for you.

```bash
uv sync
cp .env.example .env      # optional: add GOOGLE_API_KEY for generated answers
uv run fastapi dev
```

Open http://127.0.0.1:8000/docs to try it in the browser.

> The first start downloads the embedding model (about 1.1 GB), so it takes a few minutes.
> After that it starts in seconds.

## Demo

Uploading the handbook. It was already indexed, so the existing document comes back right away:

![Upload](docs/screenshots/2-upload.png)

A German question. The answer cites source `[4]`, page 6 of the handbook:

![German question](docs/screenshots/3-query-german.png)

An English question about the German document:

![English question](docs/screenshots/4-query-english.png)

## Usage

Upload a PDF. It is processed in the background, so the response comes back right away:

```bash
curl -F "file=@Gewährleistungshandbuch.pdf;type=application/pdf" \
  http://127.0.0.1:8000/api/v1/documents
```

Check the status until it says `ready` (about 80 seconds for the 155-page handbook):

```bash
curl http://127.0.0.1:8000/api/v1/documents/<id>
```

Ask a question, in German or English:

```bash
curl -H "Content-Type: application/json" \
  -d '{"question": "Wie lange gilt die Garantie auf Lackmängel?"}' \
  http://127.0.0.1:8000/api/v1/query
```

The response looks like this (shortened):

```json
{
  "answer": "Lackmängel an der Karosserie sind 3 Jahre ohne Kilometerbegrenzung abgedeckt [1].",
  "sources": [
    {"document_id": "…", "page": 6, "section": "1.1 Volkswagen Garantie- und Gewährleistungspaket > …", "content": "…"}
  ]
}
```

`[1]` in the answer points to the first source. Without an API key, `answer` is `null` and you still get the sources.

| Endpoint | What it does |
|---|---|
| `POST /api/v1/documents` | Upload a PDF (`202`, or `200` if the same file is already indexed) |
| `GET /api/v1/documents` | List documents and their status |
| `GET /api/v1/documents/{id}` | Status of one document: `processing`, `ready` or `failed` |
| `POST /api/v1/query` | Ask a question. Optional: `top_k` (default 5) and `document_id` |

## How it works

```
PDF ─► Markdown per page ─► remove headers/footers ─► split by headings ─► embed ─► store
                                                                                     │
question ─► semantic search + keyword search ─► merge results ─► Gemini answer ◄─────┘
```

A few choices worth explaining:

- **Chunks follow the document's headings.** The handbook is well structured, so I split by section first and only then by size. Every result knows its section and page.
- **Local embeddings** (`multilingual-e5-base`). Good with German, free, and the documents never leave your machine.
- **Hybrid search.** Semantic search understands meaning ("Lackschäden" finds "Lackmängel"). Keyword search (BM25) finds exact codes like `C901`. Both lists are merged with Reciprocal Rank Fusion.
- **Grounded answers.** The prompt only allows facts from the sources, asks for citations, and says "not in the documents" instead of guessing.
- **I didn't use semantic chunking.** The handbook already marks its own sections, and embedding every sentence first would make indexing much slower.

## Evaluation

`eval/questions.json` has 13 questions about the handbook: facts, processes, exact codes, English questions and one question the handbook can't answer.

```bash
uv run python -m eval.run_eval      # expects Gewährleistungshandbuch.pdf in the project root
```

| Metric | Result |
|---|---|
| Correct page in the top 5 | 12 of 12 |
| MRR | 0.78 |
| Search latency | ~50 ms median |

With an API key, the script also prints each answer and its token usage.

## Tests

```bash
uv run pytest
```

The tests run the real pipeline on small generated PDFs. Only the embedding model and the LLM are replaced, so they finish in about 2 seconds.

## Project layout

```
app/
  api/         routes and dependencies
  schemas/     Pydantic models
  services/    ingestion and query logic
  stores/      document and chunk storage (JSON files in data/)
  rag/         PDF loading, chunking, embeddings, answer generation
  core/        settings and logging
eval/          retrieval evaluation
tests/
```

## Limitations

This was built for a 60-minute task, so I kept it simple on purpose:

- Data lives in memory and in JSON files under `data/`. Run a single server process.
- If the server restarts while a PDF is processing, upload it again.
- No OCR, so text inside images is not searchable.
- If Gemini fails or hits its daily limit, the API skips the answer, logs it, and still returns the sources.

## Making it production-ready

1. **Event-driven ingestion.** An upload publishes an event, and workers parse, chunk and embed the document. This scales on its own and survives restarts.
2. **A real vector store.** OpenSearch at scale, or PostgreSQL with pgvector while the volume is small. Both also give proper full-text search.
3. **Guardrails.** Check questions and answers, refuse anything outside the documents' scope with a denial gate, and scrub personal data (PII) before it reaches the LLM or the logs.
4. **Docker.** Package the API and workers as containers, so they are easy to run and deploy.
