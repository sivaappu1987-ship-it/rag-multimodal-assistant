# ruff: noqa: E402

from unittest.mock import MagicMock
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Retrieve globally mocked modules
mock_st = sys.modules["sentence_transformers"]

from app.services.embedder import EmbedderService


class MockNumpyArray:
    """Mock object implementing .tolist() to represent a numpy array output from encoder."""

    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data


def test_embedder():
    # Force reset the singleton instance to ensure clean instantiation
    EmbedderService._instance = None

    mock_inst = MagicMock()

    # Use side_effect to dynamically return single vector or batch list
    def mock_encode(text, **kwargs):
        if isinstance(text, str):
            return MockNumpyArray([0.2] * 384)
        return [MockNumpyArray([0.2] * 384) for _ in text]

    mock_inst.encode.side_effect = mock_encode
    mock_st.SentenceTransformer.return_value = mock_inst

    service = EmbedderService()
    service.model = mock_inst

    # Verify single text embedding
    vec = service.embed_text("sample user query")
    assert len(vec) == 384
    assert vec[0] == 0.2

    # Verify batch embedding
    vecs = service.embed_batch(["first line", "second line"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384