from fastapi import status
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from tests.conftest import FAKE_ANSWER

DOCUMENTS = "/api/v1/documents"
QUERY = "/api/v1/query"


def upload(client: TestClient, content: bytes, content_type: str = "application/pdf"):
    return client.post(DOCUMENTS, files={"file": ("handbuch.pdf", content, content_type)})


def test_upload_processes_document_in_background(client: TestClient, warranty_pdf: bytes) -> None:
    response = upload(client, warranty_pdf)

    assert response.status_code == status.HTTP_202_ACCEPTED
    assert response.json()["status"] == "processing"

    document = client.get(f"{DOCUMENTS}/{response.json()['id']}").json()
    assert document["status"] == "ready"
    assert document["page_count"] == 3
    assert document["chunk_count"] == 3


def test_duplicate_upload_returns_existing_document(
    client: TestClient, warranty_pdf: bytes
) -> None:
    first = upload(client, warranty_pdf)
    second = upload(client, warranty_pdf)

    assert second.status_code == status.HTTP_200_OK
    assert second.json()["id"] == first.json()["id"]
    assert len(client.get(DOCUMENTS).json()) == 1


def test_failed_document_is_reprocessed_on_reupload(client: TestClient) -> None:
    broken_pdf = b"%PDF-1.7 this is not a real pdf"

    first = upload(client, broken_pdf)
    document = client.get(f"{DOCUMENTS}/{first.json()['id']}").json()
    assert document["status"] == "failed"
    assert document["error"]

    second = upload(client, broken_pdf)
    assert second.status_code == status.HTTP_202_ACCEPTED
    assert second.json()["id"] == first.json()["id"]


def test_rejects_non_pdf(client: TestClient) -> None:
    response = upload(client, b"hello", content_type="text/plain")

    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_rejects_too_large_file(client: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(max_upload_size_mb=1)

    response = upload(client, b"%PDF-" + b"0" * 1024 * 1024)

    assert response.status_code == status.HTTP_413_CONTENT_TOO_LARGE


def test_unknown_document_returns_404(client: TestClient) -> None:
    response = client.get(f"{DOCUMENTS}/00000000-0000-0000-0000-000000000000")

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_query_finds_page_with_exact_code(client: TestClient, warranty_pdf: bytes) -> None:
    document_id = upload(client, warranty_pdf).json()["id"]

    response = client.post(QUERY, json={"question": "Pannenhilfe KD-Nr. C901", "top_k": 1})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["answer"] == FAKE_ANSWER
    assert body["sources"] == [
        {
            "document_id": document_id,
            "page": 3,
            "section": "1.3 Mobilitaet",
            "content": "1.3 Mobilitaet\n\nPannenhilfe wird mit KD-Nr. C901 abgerechnet.",
        }
    ]
