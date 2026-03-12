"""ChunkingEngine — splits documents into chunks using configurable strategies.

Strategies:
- recursive: paragraph → sentence → character splitting with overlap
- semantic: paragraph-based splitting (simple grouping by double newlines)
- document_aware: splits on markdown headings, preserving heading hierarchy in metadata
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """A single chunk produced by the ChunkingEngine."""

    text: str
    index: int
    metadata: dict = field(default_factory=dict)


class ChunkingEngine:
    """Splits text into DocumentChunk objects using the chosen strategy."""

    def chunk(
        self,
        text: str,
        strategy: str = "recursive",
        source: str = "",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[DocumentChunk]:
        if not text or not text.strip():
            return []

        dispatch = {
            "recursive": self._recursive,
            "semantic": self._semantic,
            "document_aware": self._document_aware,
        }
        fn = dispatch.get(strategy)
        if fn is None:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        return fn(
            text=text,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Recursive strategy
    # ------------------------------------------------------------------

    def _recursive(
        self,
        text: str,
        source: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[DocumentChunk]:
        """Split by paragraphs first, then sentences, then characters."""
        raw_chunks = self._recursive_split(text, chunk_size)
        # Apply overlap between consecutive chunks
        chunks_with_overlap = self._apply_overlap(raw_chunks, chunk_overlap)
        return self._to_document_chunks(chunks_with_overlap, source)

    def _recursive_split(self, text: str, chunk_size: int) -> list[str]:
        """Recursively split text respecting chunk_size."""
        # Try paragraph split first
        paragraphs = re.split(r"\n\n+", text)
        if len(paragraphs) > 1:
            return self._merge_splits(paragraphs, chunk_size)

        # Try sentence split
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            return self._merge_splits(sentences, chunk_size)

        # Fall back to character split
        return self._char_split(text, chunk_size)

    def _merge_splits(self, parts: list[str], chunk_size: int) -> list[str]:
        """Merge small parts into chunks that respect chunk_size."""
        chunks: list[str] = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            candidate = f"{current}\n\n{part}".strip() if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If the single part exceeds chunk_size, split it further
                if len(part) > chunk_size:
                    chunks.extend(self._recursive_split(part, chunk_size))
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    def _char_split(self, text: str, chunk_size: int) -> list[str]:
        """Split text into character-level chunks."""
        chunks: list[str] = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _apply_overlap(self, chunks: list[str], overlap: int) -> list[str]:
        """Prepend trailing characters from the previous chunk as overlap."""
        if overlap <= 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-overlap:] if len(prev) >= overlap else prev
            merged = overlap_text + " " + chunks[i]
            result.append(merged.strip())
        return result

    # ------------------------------------------------------------------
    # Semantic strategy (simple paragraph grouping for now)
    # ------------------------------------------------------------------

    def _semantic(
        self,
        text: str,
        source: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[DocumentChunk]:
        """Split by double newlines (paragraph boundaries)."""
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        return self._to_document_chunks(paragraphs, source)

    # ------------------------------------------------------------------
    # Document-aware strategy
    # ------------------------------------------------------------------

    def _document_aware(
        self,
        text: str,
        source: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[DocumentChunk]:
        """Split on markdown headings, keeping heading hierarchy in metadata."""
        sections = self._split_by_headings(text)
        chunks: list[DocumentChunk] = []
        for idx, section in enumerate(sections):
            if not section["text"].strip():
                continue
            meta: dict = {"source": source}
            if section.get("heading"):
                meta["section"] = section["heading"]
            if section.get("level"):
                meta["heading_level"] = section["level"]
            chunks.append(
                DocumentChunk(text=section["text"].strip(), index=idx, metadata=meta)
            )
        # Re-index to be contiguous
        for i, c in enumerate(chunks):
            c.index = i
        return chunks

    def _split_by_headings(self, text: str) -> list[dict]:
        """Split text on markdown heading lines (# ## ### etc.)."""
        heading_re = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
        sections: list[dict] = []
        last_end = 0

        for match in heading_re.finditer(text):
            # Capture any text before this heading
            before = text[last_end : match.start()]
            if before.strip():
                sections.append({"text": before.strip(), "heading": None, "level": None})

            last_end = match.end()
            level = len(match.group(1))
            heading = match.group(2).strip()

            # Collect text until the next heading (or end)
            next_match = heading_re.search(text, match.end())
            end = next_match.start() if next_match else len(text)
            body = text[match.end() : end].strip()
            full_text = f"{match.group(0).strip()}\n{body}".strip() if body else match.group(0).strip()
            sections.append({"text": full_text, "heading": heading, "level": level})
            last_end = end

        # Remaining text after last heading
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                sections.append({"text": remaining, "heading": None, "level": None})

        # If no headings found, return the whole text as one section
        if not sections:
            sections.append({"text": text.strip(), "heading": None, "level": None})

        return sections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_document_chunks(
        self, texts: list[str], source: str
    ) -> list[DocumentChunk]:
        return [
            DocumentChunk(text=t, index=i, metadata={"source": source})
            for i, t in enumerate(texts)
            if t.strip()
        ]
