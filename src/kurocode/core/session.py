"""
Session model for KuroCode.

A *session* is a pure in-memory snapshot of one conversation: the chosen
model, an ordered list of messages, and helper methods for token estimation.

This module has **zero I/O** — it never reads or writes files, sockets, or
environment variables.  Persistence is the caller's responsibility.

Usage::

    from kurocode.session import Session

    session = Session(model_id="openai/gpt-4o-mini")
    session.add_user_message("Hello!")
    session.add_assistant_message("Hi there!")
    print(session.token_estimate())      # rough char/4 heuristic
    msgs = session.to_openrouter_messages()
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single chat message."""

    role: Role
    content: str
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """
    An in-memory conversation session.

    Parameters
    ----------
    model_id:
        OpenRouter model identifier (e.g. ``"openai/gpt-4o-mini"``).
    id:
        Session UUID.  Auto-generated when not supplied.
    messages:
        Initial message list; defaults to an empty list.
    created_at:
        Unix timestamp of session creation.  Defaults to *now*.
    """

    model_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> Message:
        """Append a ``user`` message and return the new :class:`Message`."""
        msg = Message(role="user", content=content)
        self.messages.append(msg)
        return msg

    def add_assistant_message(self, content: str) -> Message:
        """Append an ``assistant`` message and return the new :class:`Message`."""
        msg = Message(role="assistant", content=content)
        self.messages.append(msg)
        return msg

    def add_system_message(self, content: str) -> Message:
        """Append a ``system`` message and return the new :class:`Message`."""
        msg = Message(role="system", content=content)
        self.messages.append(msg)
        return msg

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------

    def token_estimate(self) -> int:
        """
        Rough token count using the ``char / 4`` heuristic.

        This is intentionally approximate — it avoids a tokeniser dependency
        and is fast enough for real-time UI feedback.  Returns ``0`` for an
        empty session.
        """
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4 if total_chars else 0

    def to_openrouter_messages(self) -> list[dict[str, str]]:
        """Return messages in the ``{role, content}`` format expected by the API."""
        return [{"role": m.role, "content": m.content} for m in self.messages]
