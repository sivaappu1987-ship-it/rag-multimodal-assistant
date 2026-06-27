
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.hybrid_search import BM25, rrf_merge


def test_bm25_search_indexing():
    chunks = [
        {"content": "printer power cable is loose error code e101", "chunk_id": "c1", "source_file": "m1.pdf"},
        {"content": "cooling fan connection inspection repair instructions", "chunk_id": "c2", "source_file": "m1.pdf"},
    ]

    bm25 = BM25(chunks)
    
    # "fan" query should score c2 highly and c1 zero
    scores = bm25.get_scores("fan")
    assert scores[1] > 0.0
    assert scores[0] == 0.0

    # "error" query should score c1 highly and c2 zero
    scores_err = bm25.get_scores("error")
    assert scores_err[0] > 0.0
    assert scores_err[1] == 0.0


def test_rrf_combination():
    dense_results = [
        {"chunk_id": "c1", "content": "c1 text", "source": "m1.pdf", "score": 0.85},
        {"chunk_id": "c2", "content": "c2 text", "source": "m1.pdf", "score": 0.80},
    ]

    sparse_results = [
        {"chunk_id": "c2", "content": "c2 text", "source": "m1.pdf", "score": 4.5},
        {"chunk_id": "c3", "content": "c3 text", "source": "m1.pdf", "score": 3.0},
    ]

    # Merge results
    merged = rrf_merge(dense_results, sparse_results, top_k=2)
    
    assert len(merged) == 2
    # c2 is present in both dense and sparse, so it should rank highly
    assert merged[0]["chunk_id"] in ["c1", "c2"]
