from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Resolve namespace collision by renaming imports
from app.main import app as fastapi_app
import app.main as app_main

# Override parser service
app_main.parser_service = MagicMock()

client = TestClient(fastapi_app)


def test_file_size_validation():
    # Construct a payload larger than 25MB (e.g. 26MB)
    huge_payload = b"B" * (26 * 1024 * 1024)

    response = client.post(
        "/upload",
        files={"file": ("manual.pdf", huge_payload, "application/pdf")},
    )

    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_invalid_extensions():
    # Unsupported file extensions must be blocked
    response = client.post(
        "/upload",
        files={
            "file": (
                "malicious.exe",
                b"fake_executable_bytes",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_invalid_mime_types():
    # Allowed extension with unallowed mime-type should be rejected
    response = client.post(
        "/upload",
        files={
            "file": (
                "manual.pdf",
                b"fake_pdf_bytes",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert "mime type" in response.json()["detail"].lower()


def test_valid_upload():
    # Valid file structure should be processed
    app_main.parser_service.parse_file.return_value = {
        "markdown_file": "manual.md",
        "chunks_ingested": 5,
    }

    response = client.post(
        "/upload",
        files={"file": ("manual.pdf", b"fake_pdf_bytes", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "manual.pdf"
    assert response.json()["status"] == "processed"


def test_valid_ppt_xls_uploads():
    app_main.parser_service.parse_file.return_value = {
        "markdown_file": "manual.md",
        "chunks_ingested": 3,
    }

    # PPT upload
    response = client.post(
        "/upload",
        files={"file": ("manual.ppt", b"fake_ppt_bytes", "application/vnd.ms-powerpoint")},
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "manual.ppt"

    # XLS upload
    response = client.post(
        "/upload",
        files={"file": ("data.xls", b"fake_xls_bytes", "application/vnd.ms-excel")},
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "data.xls"