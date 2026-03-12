"""Unit tests for ReferenceStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.store.reference import ReferenceEntry, ReferenceStore


class TestStoreAndRetrieve:
    def test_round_trip(self) -> None:
        store = ReferenceStore()
        ref_id = store.store("hello world")
        assert store.retrieve(ref_id) == "hello world"

    def test_store_with_metadata(self) -> None:
        store = ReferenceStore()
        ref_id = store.store("data", metadata={"source": "test.pdf"})
        assert store.retrieve(ref_id) == "data"

    def test_retrieve_missing_returns_none(self) -> None:
        store = ReferenceStore()
        assert store.retrieve("ref_nonexistent") is None

    def test_ref_id_format(self) -> None:
        store = ReferenceStore()
        ref_id = store.store("x")
        assert ref_id.startswith("ref_")
        assert len(ref_id) > len("ref_")

    def test_unique_ids(self) -> None:
        store = ReferenceStore()
        ids = {store.store(f"data-{i}") for i in range(50)}
        assert len(ids) == 50


class TestDelete:
    def test_delete_existing(self) -> None:
        store = ReferenceStore()
        ref_id = store.store("to delete")
        assert store.delete(ref_id) is True
        assert store.retrieve(ref_id) is None

    def test_delete_nonexistent(self) -> None:
        store = ReferenceStore()
        assert store.delete("ref_nope") is False


class TestCleanupExpired:
    def test_cleanup_removes_old_entries(self) -> None:
        store = ReferenceStore(ttl_minutes=10)
        ref_id = store.store("old data")

        # Manually age the entry
        past = datetime.now(UTC) - timedelta(minutes=15)
        store._entries[ref_id].created_at = past

        removed = store.cleanup_expired()
        assert removed == 1
        assert store.retrieve(ref_id) is None

    def test_cleanup_keeps_fresh_entries(self) -> None:
        store = ReferenceStore(ttl_minutes=10)
        ref_id = store.store("fresh data")

        removed = store.cleanup_expired()
        assert removed == 0
        assert store.retrieve(ref_id) == "fresh data"

    def test_cleanup_mixed(self) -> None:
        store = ReferenceStore(ttl_minutes=5)
        old_ref = store.store("old")
        fresh_ref = store.store("fresh")

        store._entries[old_ref].created_at = datetime.now(UTC) - timedelta(minutes=10)

        removed = store.cleanup_expired()
        assert removed == 1
        assert store.retrieve(old_ref) is None
        assert store.retrieve(fresh_ref) == "fresh"
