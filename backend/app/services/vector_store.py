"""
vector_store.py — In-memory Qdrant vector store service.
Runs entirely as a Python library — no external process or Docker required.
Uses the qdrant-client 1.x API (query_points / create_collection).
"""
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.config import QDRANT_COLLECTION, TOP_K


class VectorStoreService:
    _instance = None

    def __new__(cls):
        # Singleton: share one client across the app
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            from app.config import settings, BASE_DIR
            import os
            
            db_path = settings.QDRANT_PATH
            if not os.path.isabs(db_path):
                db_path = str(BASE_DIR / db_path)
            
            cls._instance.client = QdrantClient(path=db_path)
            cls._instance._collection_ready = False
            cls._instance._vector_size = None
            print(f"[VectorStore] Persistent Qdrant client initialized at: {db_path}")
        return cls._instance

    def _ensure_collection(self, vector_size: int):
        """Create the collection on first use if it does not already exist."""
        if not self._collection_ready:
            existing = [c.name for c in self.client.get_collections().collections]
            if QDRANT_COLLECTION not in existing:
                self.client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"[VectorStore] Collection '{QDRANT_COLLECTION}' created (dim={vector_size}).")
            else:
                print(f"[VectorStore] Collection '{QDRANT_COLLECTION}' already exists. Reusing it.")
            self._collection_ready = True
            self._vector_size = vector_size

    def ingest_chunks(self, chunks: list[dict]):
        """
        Upsert a list of embedded chunks into Qdrant.
        Each chunk must have: { chunk_id, content, source_file, embedding }
        May optionally contain metadata: product, model, category, version, section, page, product_family
        """
        if not chunks:
            return

        vector_size = len(chunks[0]["embedding"])
        self._ensure_collection(vector_size)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk["embedding"],
                payload={
                    "chunk_id":       chunk["chunk_id"],
                    "content":        chunk["content"],
                    "source_file":    chunk["source_file"],
                    "product":        chunk.get("product"),
                    "model":          chunk.get("model"),
                    "category":       chunk.get("category"),
                    "version":        chunk.get("version"),
                    "section":        chunk.get("section"),
                    "page":           chunk.get("page"),
                    "product_family": chunk.get("product_family"),
                },
            )
            for chunk in chunks
        ]

        self.client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        print(f"[VectorStore] Ingested {len(points)} chunks from '{chunks[0]['source_file']}' with metadata.")

    def search(
        self, 
        query_vector: list[float], 
        top_k: int = TOP_K, 
        source_file: str = None,
        query_filter: Filter = None
    ) -> list:
        """
        Search the collection and return top-K results, optionally filtered by source_file and query_filter.
        """
        if not self._collection_ready:
            return []

        # Merge filters if both source_file and query_filter are active
        if source_file:
            sf_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file),
                    )
                ]
            )
            if query_filter:
                if not query_filter.must:
                    query_filter.must = []
                query_filter.must.extend(sf_filter.must)
            else:
                query_filter = sf_filter

        response = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return response.points

    def get_all_chunks(self, source_file: str = None, scroll_filter: Filter = None) -> list[dict]:
        """
        Retrieve all chunks from Qdrant, optionally filtered.
        """
        if not self._collection_ready:
            return []

        if source_file:
            sf_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file),
                    )
                ]
            )
            if scroll_filter:
                if not scroll_filter.must:
                    scroll_filter.must = []
                scroll_filter.must.extend(sf_filter.must)
            else:
                scroll_filter = sf_filter

        # Scroll to get all matching points (limit set high to fetch all chunks in memory)
        response = self.client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        
        points, _ = response
        
        chunks = []
        for point in points:
            chunks.append({
                "chunk_id":       point.payload.get("chunk_id", ""),
                "content":        point.payload.get("content", ""),
                "source_file":    point.payload.get("source_file", "unknown"),
                "product":        point.payload.get("product"),
                "model":          point.payload.get("model"),
                "category":       point.payload.get("category"),
                "version":        point.payload.get("version"),
                "section":        point.payload.get("section"),
                "page":           point.payload.get("page"),
                "product_family": point.payload.get("product_family"),
            })
        return chunks

    def get_unique_sources(self) -> list[str]:
        """Retrieve all unique source filenames from the Qdrant database."""
        chunks = self.get_all_chunks()
        sources = list(set(c["source_file"] for c in chunks if c.get("source_file")))
        return sorted(sources)

    def get_unique_products(self) -> list[str]:
        """Retrieve all unique product names from the Qdrant database."""
        chunks = self.get_all_chunks()
        products = list(set(c["product"] for c in chunks if c.get("product")))
        return sorted(products)

    def count(self) -> int:
        """Return total number of vectors stored."""
        if not self._collection_ready:
            return 0
        return self.client.count(collection_name=QDRANT_COLLECTION).count
