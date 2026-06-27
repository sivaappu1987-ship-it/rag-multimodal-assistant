
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.retriever import retrieve_context


@patch("app.services.retriever.EmbedderService")
@patch("app.services.retriever.VectorStoreService")
@patch("app.services.retriever.identify_product")
def test_retriever_prioritized_hierarchy(mock_identify, mock_vs, mock_emb):
    # Setup mocks
    mock_emb_instance = MagicMock()
    mock_emb_instance.embed_text.return_value = [0.1] * 384
    mock_emb.return_value = mock_emb_instance

    mock_vs_instance = MagicMock()
    mock_vs.return_value = mock_vs_instance

    # Level 1 Match mock point
    mock_hit = MagicMock()
    mock_hit.score = 0.95
    mock_hit.payload = {
        "chunk_id": "x100::c0",
        "content": "Cooling fan connector fix manual description.",
        "source_file": "x100_manual.pdf",
    }
    
    # Level 1 returns a hit, Level 2 / 3 would mock different outputs
    mock_vs_instance.search.return_value = [mock_hit]
    mock_vs_instance.get_all_chunks.return_value = [
        {
            "chunk_id": "x100::c0",
            "content": "Cooling fan connector fix manual description.",
            "source_file": "x100_manual.pdf",
        }
    ]

    # Pre-configure product query entities
    mock_identify.return_value = {
        "product": "X100",
        "model": "X100",
        "category": "Printer",
        "error_code": "E105",
        "product_family": "X-Series",
    }

    # Execute
    context = retrieve_context("Printer X100 displays Error E105")
    
    assert len(context) == 1
    assert context[0]["chunk_id"] == "x100::c0"
    assert "x100_manual.pdf" in context[0]["source"]
    
    # Verify Level 1 exact search executed
    mock_vs_instance.search.assert_called()
