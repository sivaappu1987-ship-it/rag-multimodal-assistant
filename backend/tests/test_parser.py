# ruff: noqa: E402

import sys
import os
from unittest.mock import MagicMock, patch

# Mock markitdown first
sys.modules["markitdown"] = MagicMock()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.parser import ParserService


@patch("app.services.parser.MarkItDown")
@patch("app.services.parser.EmbedderService")
@patch("app.services.parser.VectorStoreService")
@patch("app.services.product_identifier.identify_product")
def test_parser_service(mock_identify, mock_vs, mock_emb, mock_mid):
    # Setup mocks
    mock_mid_instance = MagicMock()
    mock_mid_instance.convert.return_value.text_content = (
        "Manual page details for printer device X100 showing Error E105."
    )
    mock_mid.return_value = mock_mid_instance

    mock_emb_instance = MagicMock()
    mock_emb_instance.embed_batch.return_value = [[0.15] * 384]
    mock_emb.return_value = mock_emb_instance

    mock_vs_instance = MagicMock()
    mock_vs.return_value = mock_vs_instance

    mock_identify.return_value = {
        "product": "X100",
        "model": "X100",
        "category": "Printer",
        "error_code": "E105",
        "product_family": "X-Series",
    }

    # Execute service
    service = ParserService()
    result = service.parse_file("test_manual.pdf", b"fakecontent")

    # Assertions
    assert "markdown_file" in result
    assert result["chunks_ingested"] == 1
    mock_vs_instance.ingest_chunks.assert_called_once()