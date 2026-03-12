"""VectorStoreAdapter — abstract base class for vector store backends.

Provides a unified interface for storing document chunks, performing
similarity search, and deleting documents.  Concrete implementations
(Qdrant, Milvus, Chroma, pgvector) inherit from this class.

A factory function ``create_vector_store`` instantiates the correct
adapter based on the application configuration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.chunking.engine import DocumentChunk
from src.config.models import EmbeddingConfig, VectorStoreConfig


@dataclass
class SearchResult:
    """A single search result returned by similarity search."""

    text: str
    source: str
    page: int | None
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStoreAdapter(ABC):
    """Abstract interface for vector store backends."""

    @abstractmethod
    async def store_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Store document chunks with their embeddings."""

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform similarity search and return results with relevance scores."""

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to *doc_id*."""


def create_vector_store(
    config: VectorStoreConfig,
    embedding_config: EmbeddingConfig,
) -> VectorStoreAdapter:
    """Factory: create the appropriate vector store adapter from config."""

    provider = config.provider.lower()

    if provider == "qdrant":
        from src.vectorstore.qdrant import QdrantVectorStore

        return QdrantVectorStore(config=config, embedding_config=embedding_config)

    raise ValueError(
        f"Unsupported vector store provider: '{config.provider}'. "
        "Supported providers: qdrant"
    )
