"""QdrantVectorStore — Qdrant implementation of VectorStoreAdapter.

Uses the ``qdrant-client`` library to store document chunks, perform
cosine similarity search, and delete documents by ID filter.
Supports optional hybrid search (dense + sparse) and reranking.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from src.chunking.engine import DocumentChunk
from src.config.models import EmbeddingConfig, VectorStoreConfig
from src.vectorstore.adapter import SearchResult, VectorStoreAdapter


class QdrantVectorStore(VectorStoreAdapter):
    """Qdrant-backed vector store."""

    def __init__(
        self,
        config: VectorStoreConfig,
        embedding_config: EmbeddingConfig,
        *,
        client: QdrantClient | None = None,
    ) -> None:
        self._config = config
        self._embedding_config = embedding_config
        self._collection = config.collection_name
        self._dimension = embedding_config.dimension
        self._hybrid = config.hybrid_search

        # Allow injecting a client (useful for testing)
        self._client = client or QdrantClient(
            host=config.host,
            port=config.port,
        )

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection in existing:
            return

        vectors_config: dict[str, models.VectorParams] = {
            "dense": models.VectorParams(
                size=self._dimension,
                distance=models.Distance.COSINE,
            ),
        }

        sparse_vectors_config: dict[str, models.SparseVectorParams] | None = None
        if self._hybrid:
            sparse_vectors_config = {
                "sparse": models.SparseVectorParams(),
            }

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

    # ------------------------------------------------------------------
    # Embedding helper (placeholder — delegates to real model later)
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Return a dense embedding vector for *text*.

        This is a placeholder that returns a zero-vector of the configured
        dimension.  In production this would call the configured embedding
        model (e.g. bge-m3 via sentence-transformers or an API).
        """
        return [0.0] * self._dimension

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    async def store_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Store document chunks with embeddings and metadata."""
        if not chunks:
            return

        points: list[models.PointStruct] = []
        for chunk in chunks:
            vector = self._embed(chunk.text)
            point_id = str(uuid.uuid4())

            payload: dict[str, Any] = {
                "text": chunk.text,
                "source": chunk.metadata.get("source", ""),
                "page": chunk.metadata.get("page"),
                "section": chunk.metadata.get("section", ""),
                "index": chunk.index,
                **{
                    k: v
                    for k, v in chunk.metadata.items()
                    if k not in ("source", "page", "section")
                },
            }

            vectors: dict[str, Any] = {"dense": vector}
            # Hybrid: add a trivial sparse vector placeholder
            if self._hybrid:
                vectors["sparse"] = models.SparseVector(
                    indices=[0],
                    values=[1.0],
                )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Cosine similarity search, with optional reranking."""
        vector = self._embed(query)

        hits = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using="dense",
            limit=top_k if not self._should_rerank() else top_k * 3,
            with_payload=True,
        ).points

        results = [self._hit_to_result(h) for h in hits]

        if self._should_rerank():
            results = self._rerank(query, results, top_k)

        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_document(self, doc_id: str) -> None:
        """Delete all points whose ``source`` payload matches *doc_id*."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )

    # ------------------------------------------------------------------
    # Reranker helpers
    # ------------------------------------------------------------------

    def _should_rerank(self) -> bool:
        return (
            self._config.reranker is not None
            and self._config.reranker.enabled
        )

    def _rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Placeholder reranker — simply returns the first *top_k* results.

        A real implementation would load a cross-encoder model (e.g.
        ``cross-encoder/ms-marco-MiniLM-L-6-v2``) and re-score each result.
        """
        return results[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hit_to_result(hit: Any) -> SearchResult:
        payload = hit.payload or {}
        return SearchResult(
            text=payload.get("text", ""),
            source=payload.get("source", ""),
            page=payload.get("page"),
            score=hit.score if hasattr(hit, "score") and hit.score is not None else 0.0,
            metadata={
                k: v
                for k, v in payload.items()
                if k not in ("text",)
            },
        )
