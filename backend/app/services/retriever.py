from typing import Optional, Tuple, List, Dict, Any
from app.services.embedder import EmbedderService
from app.services.vector_store import VectorStoreService
from app.services.hybrid_search import BM25, rrf_merge
from app.services.metadata_resolver import resolve_metadata_filter
from app.services.product_identifier import identify_product
from app.config import TOP_K, SCORE_THRESHOLD, RRF_HIGH_THRESHOLD, RRF_LOW_THRESHOLD


def retrieve_context(
    query: str, 
    source_file: Optional[str] = None,
    query_entities: Optional[dict] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Retrieve relevant chunks from Qdrant using hierarchical hybrid search:
      - Level 1: Exact product/model match filter
      - Level 2: Product family match filter
      - Level 3: Global manuals (no metadata filter)
    At each active level, merges dense similarity and sparse keyword search (BM25) using RRF.
    """
    embedder = EmbedderService()
    vector_store = VectorStoreService()

    # Identify query entities if not pre-extracted
    if not query_entities:
        query_entities = identify_product(query)

    query_vector = embedder.embed_text(query)

    # Hierarchical search loop (Levels 1, 2, 3)
    for level in [1, 2, 3]:
        q_filter = resolve_metadata_filter(query_entities, filter_level=level) if query_entities else None

        # Skip level if filter is expected but not resolved (e.g. no product detected)
        if level < 3 and not q_filter:
            continue

        # 1. Dense Vector Search: Fetch top 50 candidates
        dense_hits = vector_store.search(
            query_vector, 
            top_k=50, 
            source_file=source_file, 
            query_filter=q_filter
        )

        dense_candidates = []
        for hit in dense_hits:
            if hit.score >= SCORE_THRESHOLD:
                dense_candidates.append({
                    "content":  hit.payload.get("content", ""),
                    "score":    round(hit.score, 4),
                    "source":   hit.payload.get("source_file", "unknown"),
                    "chunk_id": hit.payload.get("chunk_id", ""),
                })

        # 2. Sparse BM25 Search: Fetch matching candidate chunks
        all_chunks = vector_store.get_all_chunks(source_file=source_file, scroll_filter=q_filter)
        
        sparse_candidates = []
        if all_chunks:
            bm25 = BM25(all_chunks)
            scores = bm25.get_scores(query)

            # Pair chunks with their scores
            chunk_scores = []
            for chunk, score in zip(all_chunks, scores):
                if score > 0:  # Only keep chunks with some keyword matching
                    chunk_scores.append((chunk, score))

            # Sort descending by score and pick top 50 candidates
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            top_sparse = chunk_scores[:50]

            for chunk, score in top_sparse:
                sparse_candidates.append({
                    "content":  chunk["content"],
                    "score":    round(score, 4),
                    "source":   chunk["source_file"],
                    "chunk_id": chunk["chunk_id"],
                })

        # 3. Reciprocal Rank Fusion (RRF): Merge dense and sparse candidate lists
        merged_results = rrf_merge(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            top_k=TOP_K,
        )

        # Return results if any are found at this hierarchy level
        if merged_results:
            print(f"[Retriever] Found {len(merged_results)} chunks at retrieval Level {level} (filter: {q_filter is not None}).")
            top_rrf_score = merged_results[0].get("rrf_score", 0.0)
            print(f"[Retriever] Top RRF score: {top_rrf_score}")
            if top_rrf_score >= RRF_HIGH_THRESHOLD:
                retrieval_confidence = "HIGH"
            elif top_rrf_score >= RRF_LOW_THRESHOLD:
                retrieval_confidence = "MEDIUM"
            else:
                retrieval_confidence = "LOW"
            return merged_results, retrieval_confidence

    # Return empty if nothing passes score threshold across all levels
    return [], "LOW"


