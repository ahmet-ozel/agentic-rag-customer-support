"""Unit tests for VectorStoreAdapter, QdrantVectorStore, and factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.chunking.engine import DocumentChunk
from src.config.models import (
    EmbeddingConfig,
    RerankerConfig,
    VectorStoreConfig,
)
from src.vectorstore.adapter import SearchResult, VectorStoreAdapter, create_vector_store
from src.vectorstore.qdrant import QdrantVectorStore


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _vs_config(**overrides) -> VectorStoreConfig:
    defaults = dict(provider="qdrant", host="localhost", port=6333, collection_name="test_docs")
    defaults.update(overrides)
    return VectorStoreConfig(**defaults)


def _emb_config(**overrides) -> EmbeddingConfig:
    defaults = dict(model="bge-m3", provider="local", dimension=128)
    defaults.update(overrides)
    return EmbeddingConfig(**defaults)


def _mock_qdrant_client(collection_exists: bool = False) -> MagicMock:
    """Return a mock QdrantClient with sensible defaults."""
    client = MagicMock()

    # get_collections
    col = MagicMock()
    col.name = "test_docs" if collection_exists else "__other__"
    collections_resp = MagicMock()
    collections_resp.collections = [col]
    client.get_collections.return_value = collections_resp

    # query_points — return empty by default
    query_resp = MagicMock()
    query_resp.points = []
    client.query_points.return_value = query_resp

    return client


def _make_store(
    collection_exists: bool = False,
    hybrid: bool = False,
    reranker: RerankerConfig | None = None,
    client: MagicMock | None = None,
) -> tuple[QdrantVectorStore, MagicMock]:
    mock_client = client or _mock_qdrant_client(collection_exists)
    config = _vs_config(hybrid_search=hybrid, reranker=reranker)
    emb = _emb_config()
    store = QdrantVectorStore(config=config, embedding_config=emb, client=mock_client)
    return store, mock_client


# ------------------------------------------------------------------
# SearchResult dataclass
# ------------------------------------------------------------------


class TestSearchResult:
    def test_fields(self) -> None:
        r = SearchResult(text="hello", source="doc.pdf", page=1, score=0.95, metadata={"k": "v"})
        assert r.text == "hello"
        assert r.source == "doc.pdf"
        assert r.page == 1
        assert r.score == 0.95
        assert r.metadata == {"k": "v"}

    def test_default_metadata(self) -> None:
        r = SearchResult(text="t", source="s", page=None, score=0.5)
        assert r.metadata == {}


# ------------------------------------------------------------------
# VectorStoreAdapter is abstract
# ------------------------------------------------------------------


class TestAbstractAdapter:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            VectorStoreAdapter()  # type: ignore[abstract]


# ------------------------------------------------------------------
# Factory function
# ------------------------------------------------------------------


class TestCreateVectorStore:
    def test_qdrant_provider(self) -> None:
        mock_client = _mock_qdrant_client()
        with patch("src.vectorstore.qdrant.QdrantClient", return_value=mock_client):
            store = create_vector_store(_vs_config(), _emb_config())
        assert isinstance(store, QdrantVectorStore)

    def test_unsupported_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported vector store provider"):
            create_vector_store(_vs_config(provider="redis"), _emb_config())

    def test_case_insensitive_provider(self) -> None:
        mock_client = _mock_qdrant_client()
        with patch("src.vectorstore.qdrant.QdrantClient", return_value=mock_client):
            store = create_vector_store(_vs_config(provider="Qdrant"), _emb_config())
        assert isinstance(store, QdrantVectorStore)


# ------------------------------------------------------------------
# QdrantVectorStore — collection management
# ------------------------------------------------------------------


class TestCollectionManagement:
    def test_creates_collection_when_missing(self) -> None:
        _, client = _make_store(collection_exists=False)
        client.create_collection.assert_called_once()

    def test_skips_creation_when_exists(self) -> None:
        _, client = _make_store(collection_exists=True)
        client.create_collection.assert_not_called()

    def test_hybrid_creates_sparse_config(self) -> None:
        _, client = _make_store(collection_exists=False, hybrid=True)
        call_kwargs = client.create_collection.call_args
        assert call_kwargs.kwargs.get("sparse_vectors_config") is not None

    def test_non_hybrid_no_sparse_config(self) -> None:
        _, client = _make_store(collection_exists=False, hybrid=False)
        call_kwargs = client.create_collection.call_args
        assert call_kwargs.kwargs.get("sparse_vectors_config") is None


# ------------------------------------------------------------------
# store_chunks
# ------------------------------------------------------------------


class TestStoreChunks:
    @pytest.mark.asyncio
    async def test_empty_chunks_no_upsert(self) -> None:
        store, client = _make_store(collection_exists=True)
        await store.store_chunks([])
        client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_stores_single_chunk(self) -> None:
        store, client = _make_store(collection_exists=True)
        chunk = DocumentChunk(text="hello world", index=0, metadata={"source": "a.pdf", "page": 1})
        await store.store_chunks([chunk])
        client.upsert.assert_called_once()
        points = client.upsert.call_args.kwargs["points"]
        assert len(points) == 1
        assert points[0].payload["text"] == "hello world"
        assert points[0].payload["source"] == "a.pdf"
        assert points[0].payload["page"] == 1

    @pytest.mark.asyncio
    async def test_stores_multiple_chunks(self) -> None:
        store, client = _make_store(collection_exists=True)
        chunks = [
            DocumentChunk(text=f"chunk {i}", index=i, metadata={"source": "b.pdf"})
            for i in range(3)
        ]
        await store.store_chunks(chunks)
        points = client.upsert.call_args.kwargs["points"]
        assert len(points) == 3

    @pytest.mark.asyncio
    async def test_metadata_preserved(self) -> None:
        store, client = _make_store(collection_exists=True)
        chunk = DocumentChunk(
            text="t", index=0,
            metadata={"source": "x.pdf", "page": 5, "section": "intro", "custom_key": "val"},
        )
        await store.store_chunks([chunk])
        payload = client.upsert.call_args.kwargs["points"][0].payload
        assert payload["source"] == "x.pdf"
        assert payload["page"] == 5
        assert payload["section"] == "intro"
        assert payload["custom_key"] == "val"

    @pytest.mark.asyncio
    async def test_hybrid_includes_sparse_vector(self) -> None:
        store, client = _make_store(collection_exists=True, hybrid=True)
        chunk = DocumentChunk(text="t", index=0, metadata={"source": "s"})
        await store.store_chunks([chunk])
        point = client.upsert.call_args.kwargs["points"][0]
        assert "sparse" in point.vector


# ------------------------------------------------------------------
# search
# ------------------------------------------------------------------


class TestSearch:
    @pytest.mark.asyncio
    async def test_returns_empty_on_no_hits(self) -> None:
        store, _ = _make_store(collection_exists=True)
        results = await store.search("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_search_results(self) -> None:
        store, client = _make_store(collection_exists=True)
        hit = MagicMock()
        hit.payload = {"text": "found", "source": "d.pdf", "page": 2}
        hit.score = 0.88
        resp = MagicMock()
        resp.points = [hit]
        client.query_points.return_value = resp

        results = await store.search("test query", top_k=3)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].text == "found"
        assert results[0].source == "d.pdf"
        assert results[0].page == 2
        assert results[0].score == 0.88

    @pytest.mark.asyncio
    async def test_search_passes_top_k(self) -> None:
        store, client = _make_store(collection_exists=True)
        await store.search("q", top_k=7)
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 7

    @pytest.mark.asyncio
    async def test_reranker_fetches_more_then_trims(self) -> None:
        reranker = RerankerConfig(enabled=True, model="test-model", top_k=5)
        store, client = _make_store(collection_exists=True, reranker=reranker)
        # Provide enough hits
        hits = []
        for i in range(15):
            h = MagicMock()
            h.payload = {"text": f"t{i}", "source": "s"}
            h.score = 1.0 - i * 0.05
            hits.append(h)
        resp = MagicMock()
        resp.points = hits
        client.query_points.return_value = resp

        results = await store.search("q", top_k=5)
        # Reranker placeholder trims to top_k
        assert len(results) == 5
        # Should have fetched top_k * 3 = 15
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 15


# ------------------------------------------------------------------
# delete_document
# ------------------------------------------------------------------


class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete_calls_client(self) -> None:
        store, client = _make_store(collection_exists=True)
        await store.delete_document("report.pdf")
        client.delete.assert_called_once()
        call_kwargs = client.delete.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_docs"
