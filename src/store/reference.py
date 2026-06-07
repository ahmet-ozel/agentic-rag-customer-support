"""Reference Store - temporary in-memory storage for large data with TTL."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


@dataclass
class ReferenceEntry:
    """A single stored reference."""

    data: str
    metadata: dict | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ReferenceStore:
    """In-memory store that maps ``ref_xxx`` codes to data strings.

    Parameters
    ----------
    ttl_minutes:
        Time-to-live for entries in minutes.  Defaults to 30.
    """

    def __init__(self, ttl_minutes: int = 30) -> None:
        self._entries: dict[str, ReferenceEntry] = {}
        self._ttl = timedelta(minutes=ttl_minutes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, data: str, metadata: dict | None = None) -> str:
        """Store *data* and return a ``ref_<uuid4>`` reference code."""
        ref_id = f"ref_{uuid.uuid4().hex}"
        self._entries[ref_id] = ReferenceEntry(data=data, metadata=metadata)
        return ref_id

    def retrieve(self, ref_id: str) -> str | None:
        """Return the stored data for *ref_id*, or ``None`` if missing."""
        entry = self._entries.get(ref_id)
        return entry.data if entry is not None else None

    def delete(self, ref_id: str) -> bool:
        """Delete a reference.  Returns ``True`` if it existed."""
        return self._entries.pop(ref_id, None) is not None

    def cleanup_expired(self) -> int:
        """Remove entries older than the configured TTL.

        Returns the number of entries removed.
        """
        now = datetime.now(UTC)
        expired = [
            rid for rid, entry in self._entries.items()
            if now - entry.created_at >= self._ttl
        ]
        for rid in expired:
            del self._entries[rid]
        return len(expired)
