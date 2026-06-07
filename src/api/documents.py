"""Document endpoints.

POST   /api/v1/documents      - Upload a document (parser → Reference Store → chunking → embedding → Vector Store)
GET    /api/v1/documents      - List all documents
DELETE /api/v1/documents/{id}  - Delete a document
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from src.api.dependencies import (
    get_chunking_engine,
    get_reference_store,
    get_vector_store,
)
from src.chunking.engine import ChunkingEngine
from src.models.schemas import DocumentUploadResponse
from src.store.reference import ReferenceStore
from src.vectorstore.adapter import VectorStoreAdapter

router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------------
# In-memory document tracking
# ---------------------------------------------------------------------------

VALID_TRANSITIONS = {
    "uploading": {"parsing", "error"},
    "parsing": {"chunking", "error"},
    "chunking": {"embedding", "error"},
    "embedding": {"completed", "error"},
    "completed": {"error"},
    "error": set(),
}

_documents: dict[str, dict] = {}


def _set_document_status(doc_id: str, new_status: str) -> None:
    """Transition document status, enforcing valid transitions."""
    doc = _documents.get(doc_id)
    if doc is None:
        return
    current = doc["status"]
    if new_status in VALID_TRANSITIONS.get(current, set()):
        doc["status"] = new_status
        doc["updated_at"] = datetime.now(UTC).isoformat()


def get_documents_store() -> dict[str, dict]:
    """Return the in-memory documents dict (for testing)."""
    return _documents


# ---------------------------------------------------------------------------
# Optional dependency helpers (allow None when not configured)
# ---------------------------------------------------------------------------


def _optional_reference_store() -> Optional[ReferenceStore]:
    from src.api import dependencies as deps
    return deps._reference_store


def _optional_chunking_engine() -> Optional[ChunkingEngine]:
    from src.api import dependencies as deps
    return deps._chunking_engine


def _optional_vector_store() -> Optional[VectorStoreAdapter]:
    from src.api import dependencies as deps
    return deps._vector_store


# ---------------------------------------------------------------------------
# POST /api/v1/documents
# ---------------------------------------------------------------------------


@router.post("/documents", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    reference_store: Optional[ReferenceStore] = Depends(_optional_reference_store),
    chunking_engine: Optional[ChunkingEngine] = Depends(_optional_chunking_engine),
    vector_store: Optional[VectorStoreAdapter] = Depends(_optional_vector_store),
) -> DocumentUploadResponse:
    """Upload a document and run the processing pipeline.

    Status transitions: uploading → parsing → chunking → embedding → completed (or → error)
    """
    doc_id = str(uuid.uuid4())
    filename = file.filename or "unknown"

    _documents[doc_id] = {
        "document_id": doc_id,
        "filename": filename,
        "status": "uploading",
        "message": "Document received",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
    }

    try:
        content = await file.read()
        text = content.decode("utf-8", errors="replace")

        # --- parsing ---
        _set_document_status(doc_id, "parsing")
        if reference_store is not None:
            reference_store.store(
                text, metadata={"filename": filename, "doc_id": doc_id}
            )

        # --- chunking ---
        _set_document_status(doc_id, "chunking")
        chunks = []
        if chunking_engine is not None:
            chunks = chunking_engine.chunk(text, source=filename)

        # --- embedding + vector store ---
        _set_document_status(doc_id, "embedding")
        if vector_store is not None and chunks:
            for chunk in chunks:
                chunk.metadata["doc_id"] = doc_id
            await vector_store.store_chunks(chunks)

        # --- completed ---
        _set_document_status(doc_id, "completed")
        _documents[doc_id]["message"] = (
            f"Document processed successfully ({len(chunks)} chunks)"
        )

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=filename,
            status="completed",
            message=_documents[doc_id]["message"],
        )

    except Exception as exc:
        _set_document_status(doc_id, "error")
        _documents[doc_id]["message"] = f"Processing failed: {exc}"
        raise HTTPException(
            status_code=500,
            detail={
                "error": "document_processing_error",
                "message": str(exc),
                "document_id": doc_id,
            },
        ) from exc


# ---------------------------------------------------------------------------
# GET /api/v1/documents
# ---------------------------------------------------------------------------


@router.get("/documents")
async def list_documents() -> list[dict]:
    """List all tracked documents."""
    return list(_documents.values())


# ---------------------------------------------------------------------------
# DELETE /api/v1/documents/{id}
# ---------------------------------------------------------------------------


@router.delete("/documents/{doc_id}", status_code=200)
async def delete_document(
    doc_id: str,
    vector_store: Optional[VectorStoreAdapter] = Depends(_optional_vector_store),
) -> dict:
    """Delete a document by ID."""
    if doc_id not in _documents:
        raise HTTPException(status_code=404, detail="Document not found")

    if vector_store is not None:
        try:
            await vector_store.delete_document(doc_id)
        except Exception:
            pass

    del _documents[doc_id]
    return {"message": "Document deleted", "document_id": doc_id}
