"""Unit tests for SessionManager."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.session.manager import Session, SessionManager


class TestCreateAndGetSession:
    def test_create_session_returns_session(self) -> None:
        mgr = SessionManager()
        session = mgr.create_session()
        assert isinstance(session, Session)
        assert isinstance(session.session_id, str)
        assert session.messages == []

    def test_get_session_returns_created_session(self) -> None:
        mgr = SessionManager()
        session = mgr.create_session()
        retrieved = mgr.get_session(session.session_id)
        assert retrieved is session

    def test_get_nonexistent_session_returns_none(self) -> None:
        mgr = SessionManager()
        assert mgr.get_session("does-not-exist") is None

    def test_unique_session_ids(self) -> None:
        mgr = SessionManager()
        ids = {mgr.create_session().session_id for _ in range(100)}
        assert len(ids) == 100


class TestAddMessages:
    def test_add_message_appends(self) -> None:
        mgr = SessionManager()
        session = mgr.create_session()
        msg = {"role": "user", "content": "hello"}
        mgr.add_message(session.session_id, msg)
        assert session.messages == [msg]

    def test_add_message_updates_last_activity(self) -> None:
        mgr = SessionManager()
        session = mgr.create_session()
        before = session.last_activity
        mgr.add_message(session.session_id, {"role": "user", "content": "hi"})
        assert session.last_activity >= before

    def test_add_message_to_nonexistent_raises(self) -> None:
        mgr = SessionManager()
        with pytest.raises(KeyError):
            mgr.add_message("no-such-id", {"role": "user", "content": "x"})


class TestMaxMessageLimit:
    def test_oldest_messages_trimmed(self) -> None:
        mgr = SessionManager(max_messages=3)
        session = mgr.create_session()
        for i in range(5):
            mgr.add_message(session.session_id, {"index": i})

        assert len(session.messages) == 3
        # Only the last 3 messages should remain
        assert session.messages == [{"index": 2}, {"index": 3}, {"index": 4}]

    def test_at_limit_no_trim(self) -> None:
        mgr = SessionManager(max_messages=3)
        session = mgr.create_session()
        for i in range(3):
            mgr.add_message(session.session_id, {"index": i})
        assert len(session.messages) == 3


class TestCleanupExpired:
    def test_cleanup_removes_expired_sessions(self) -> None:
        mgr = SessionManager(timeout_minutes=10)
        session = mgr.create_session()

        # Manually age the session
        session.last_activity = datetime.now(UTC) - timedelta(minutes=15)

        removed = mgr.cleanup_expired()
        assert removed == 1
        assert mgr.get_session(session.session_id) is None

    def test_cleanup_keeps_active_sessions(self) -> None:
        mgr = SessionManager(timeout_minutes=10)
        session = mgr.create_session()

        removed = mgr.cleanup_expired()
        assert removed == 0
        assert mgr.get_session(session.session_id) is session

    def test_cleanup_mixed(self) -> None:
        mgr = SessionManager(timeout_minutes=5)
        old = mgr.create_session()
        fresh = mgr.create_session()

        old.last_activity = datetime.now(UTC) - timedelta(minutes=10)

        removed = mgr.cleanup_expired()
        assert removed == 1
        assert mgr.get_session(old.session_id) is None
        assert mgr.get_session(fresh.session_id) is fresh
