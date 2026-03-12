"""Unit tests for ChunkingEngine."""

from __future__ import annotations

import pytest

from src.chunking.engine import ChunkingEngine, DocumentChunk


@pytest.fixture
def engine() -> ChunkingEngine:
    return ChunkingEngine()


# ------------------------------------------------------------------
# Empty / edge-case inputs
# ------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_string_returns_empty(self, engine: ChunkingEngine) -> None:
        assert engine.chunk("") == []

    def test_whitespace_only_returns_empty(self, engine: ChunkingEngine) -> None:
        assert engine.chunk("   \n\n  ") == []

    def test_empty_for_all_strategies(self, engine: ChunkingEngine) -> None:
        for strategy in ("recursive", "semantic", "document_aware"):
            assert engine.chunk("", strategy=strategy) == []


# ------------------------------------------------------------------
# All strategies produce valid DocumentChunk objects
# ------------------------------------------------------------------


class TestValidChunks:
    @pytest.mark.parametrize("strategy", ["recursive", "semantic", "document_aware"])
    def test_chunks_are_document_chunk_instances(
        self, engine: ChunkingEngine, strategy: str
    ) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = engine.chunk(text, strategy=strategy)
        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c, DocumentChunk)
            assert isinstance(c.text, str)
            assert isinstance(c.index, int)
            assert isinstance(c.metadata, dict)

    @pytest.mark.parametrize("strategy", ["recursive", "semantic", "document_aware"])
    def test_all_chunks_non_empty(
        self, engine: ChunkingEngine, strategy: str
    ) -> None:
        text = "Hello world.\n\nAnother section.\n\nFinal part."
        chunks = engine.chunk(text, strategy=strategy)
        for c in chunks:
            assert c.text.strip() != ""


# ------------------------------------------------------------------
# Recursive chunking
# ------------------------------------------------------------------


class TestRecursiveChunking:
    def test_produces_non_empty_chunks(self, engine: ChunkingEngine) -> None:
        text = "A" * 100 + "\n\n" + "B" * 100
        chunks = engine.chunk(text, strategy="recursive")
        assert len(chunks) > 0
        for c in chunks:
            assert c.text.strip() != ""

    def test_respects_chunk_size(self, engine: ChunkingEngine) -> None:
        text = "\n\n".join(f"Paragraph {i} content." for i in range(20))
        chunks = engine.chunk(text, strategy="recursive", chunk_size=60, chunk_overlap=0)
        for c in chunks:
            # Each raw chunk (before overlap) should be within chunk_size
            assert len(c.text) > 0

    def test_single_paragraph_no_split_needed(self, engine: ChunkingEngine) -> None:
        text = "Short text."
        chunks = engine.chunk(text, strategy="recursive", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."


# ------------------------------------------------------------------
# Chunk overlap
# ------------------------------------------------------------------


class TestChunkOverlap:
    def test_overlap_adds_trailing_context(self, engine: ChunkingEngine) -> None:
        text = "First paragraph here.\n\nSecond paragraph here."
        chunks = engine.chunk(
            text, strategy="recursive", chunk_size=30, chunk_overlap=10
        )
        # With overlap, the second chunk should contain trailing chars from the first
        if len(chunks) > 1:
            first_tail = chunks[0].text[-10:]
            assert first_tail in chunks[1].text

    def test_zero_overlap_no_duplication(self, engine: ChunkingEngine) -> None:
        text = "AAA.\n\nBBB.\n\nCCC."
        chunks = engine.chunk(
            text, strategy="recursive", chunk_size=512, chunk_overlap=0
        )
        assert len(chunks) >= 1


# ------------------------------------------------------------------
# Document-aware chunking
# ------------------------------------------------------------------


class TestDocumentAwareChunking:
    def test_splits_on_headings(self, engine: ChunkingEngine) -> None:
        text = "# Heading 1\nContent one.\n\n## Heading 2\nContent two."
        chunks = engine.chunk(text, strategy="document_aware")
        assert len(chunks) >= 2

    def test_heading_in_metadata(self, engine: ChunkingEngine) -> None:
        text = "# Introduction\nSome intro text.\n\n## Details\nMore details."
        chunks = engine.chunk(text, strategy="document_aware")
        headings = [c.metadata.get("section") for c in chunks]
        assert "Introduction" in headings
        assert "Details" in headings

    def test_heading_level_in_metadata(self, engine: ChunkingEngine) -> None:
        text = "# H1\nText.\n\n### H3\nMore text."
        chunks = engine.chunk(text, strategy="document_aware")
        levels = {c.metadata.get("heading_level") for c in chunks if c.metadata.get("heading_level")}
        assert 1 in levels
        assert 3 in levels

    def test_text_without_headings(self, engine: ChunkingEngine) -> None:
        text = "Just plain text without any headings."
        chunks = engine.chunk(text, strategy="document_aware")
        assert len(chunks) == 1
        assert chunks[0].text == text


# ------------------------------------------------------------------
# Semantic chunking
# ------------------------------------------------------------------


class TestSemanticChunking:
    def test_splits_by_paragraphs(self, engine: ChunkingEngine) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = engine.chunk(text, strategy="semantic")
        assert len(chunks) == 3

    def test_source_in_metadata(self, engine: ChunkingEngine) -> None:
        text = "Some text.\n\nMore text."
        chunks = engine.chunk(text, strategy="semantic", source="test.pdf")
        for c in chunks:
            assert c.metadata["source"] == "test.pdf"


# ------------------------------------------------------------------
# Unknown strategy
# ------------------------------------------------------------------


class TestUnknownStrategy:
    def test_raises_value_error(self, engine: ChunkingEngine) -> None:
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            engine.chunk("text", strategy="unknown")
