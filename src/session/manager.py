"""Session Manager - in-memory conversation session management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


@dataclass
class Session:
    """A single conversation session."""

    session_id: str
    messages: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))


class SessionManager:
    """Manages conversation sessions in memory.

    Parameters
    ----------
    max_messages:
        Maximum number of messages kept per session.  Oldest are trimmed.
    timeout_minutes:
        Inactivity timeout after which sessions are considered expired.
    """

    def __init__(
        self,
        max_messages: int = 50,
        timeout_minutes: int = 30,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._max_messages = max_messages
        self._timeout = timedelta(minutes=timeout_minutes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self) -> Session:
        """Create a new session with a unique UUID-based session_id."""
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Return the session for *session_id*, or ``None`` if not found."""
        return self._sessions.get(session_id)

    def add_message(self, session_id: str, message: dict) -> None:
        """Append *message* to the session and update last_activity.

        If the session's message count exceeds *max_messages*, the oldest
        messages are trimmed to stay within the limit.

        Raises
        ------
        KeyError
            If *session_id* does not exist.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        session.messages.append(message)
        session.last_activity = datetime.now(UTC)

        if len(session.messages) > self._max_messages:
            session.messages = session.messages[-self._max_messages:]

    def cleanup_expired(self) -> int:
        """Remove sessions that have been inactive past the timeout.

        Returns the number of sessions removed.
        """
        now = datetime.now(UTC)
        expired = [
            sid
            for sid, session in self._sessions.items()
            if now - session.last_activity >= self._timeout
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)
