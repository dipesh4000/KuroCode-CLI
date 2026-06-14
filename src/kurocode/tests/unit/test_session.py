"""
Unit tests for kurocode.session.

All tests are pure — Session is a plain dataclass with no I/O.
No fixtures, no mocking, no tmp_path required.
"""

from __future__ import annotations

import time
import uuid

import pytest

from kurocode.core.session import Message, Session


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_stores_role_and_content(self) -> None:
        msg = Message(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_created_at_defaults_to_now(self) -> None:
        before = time.time()
        msg = Message(role="assistant", content="Hi")
        after = time.time()
        assert before <= msg.created_at <= after

    def test_created_at_can_be_set_explicitly(self) -> None:
        ts = 1_700_000_000.0
        msg = Message(role="system", content="ctx", created_at=ts)
        assert msg.created_at == ts

    def test_all_valid_roles(self) -> None:
        for role in ("system", "user", "assistant"):
            msg = Message(role=role, content="x")  # type: ignore[arg-type]
            assert msg.role == role


# ---------------------------------------------------------------------------
# Session — construction
# ---------------------------------------------------------------------------


class TestSessionConstruction:
    def test_requires_model_id(self) -> None:
        s = Session(model_id="openai/gpt-4o-mini")
        assert s.model_id == "openai/gpt-4o-mini"

    def test_id_is_valid_uuid_by_default(self) -> None:
        s = Session(model_id="x")
        parsed = uuid.UUID(s.id)  # raises ValueError if not a valid UUID
        assert str(parsed) == s.id

    def test_two_sessions_have_different_ids(self) -> None:
        a = Session(model_id="x")
        b = Session(model_id="x")
        assert a.id != b.id

    def test_id_can_be_supplied(self) -> None:
        fixed = "00000000-0000-0000-0000-000000000001"
        s = Session(model_id="x", id=fixed)
        assert s.id == fixed

    def test_messages_default_to_empty_list(self) -> None:
        s = Session(model_id="x")
        assert s.messages == []

    def test_messages_not_shared_between_instances(self) -> None:
        """Each Session must own its own list — no mutable default sharing."""
        a = Session(model_id="x")
        b = Session(model_id="x")
        a.add_user_message("hello")
        assert b.messages == []

    def test_created_at_defaults_to_now(self) -> None:
        before = time.time()
        s = Session(model_id="x")
        after = time.time()
        assert before <= s.created_at <= after


# ---------------------------------------------------------------------------
# Session — add_*_message helpers
# ---------------------------------------------------------------------------


class TestSessionAddMessage:
    def test_add_user_message_appends(self) -> None:
        s = Session(model_id="x")
        msg = s.add_user_message("Hello")
        assert len(s.messages) == 1
        assert s.messages[0] is msg
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_add_assistant_message_appends(self) -> None:
        s = Session(model_id="x")
        msg = s.add_assistant_message("Hi there")
        assert len(s.messages) == 1
        assert msg.role == "assistant"
        assert msg.content == "Hi there"

    def test_add_system_message_appends(self) -> None:
        s = Session(model_id="x")
        msg = s.add_system_message("You are a helpful assistant.")
        assert len(s.messages) == 1
        assert msg.role == "system"

    def test_messages_preserved_in_order(self) -> None:
        s = Session(model_id="x")
        s.add_system_message("ctx")
        s.add_user_message("question")
        s.add_assistant_message("answer")
        roles = [m.role for m in s.messages]
        assert roles == ["system", "user", "assistant"]

    def test_returned_message_is_in_list(self) -> None:
        s = Session(model_id="x")
        returned = s.add_user_message("ping")
        assert returned in s.messages

    def test_add_multiple_user_messages(self) -> None:
        s = Session(model_id="x")
        s.add_user_message("first")
        s.add_user_message("second")
        assert len(s.messages) == 2
        assert s.messages[1].content == "second"


# ---------------------------------------------------------------------------
# Session — token_estimate
# ---------------------------------------------------------------------------


class TestTokenEstimate:
    def test_empty_session_returns_zero(self) -> None:
        s = Session(model_id="x")
        assert s.token_estimate() == 0

    def test_single_message_char_over_four(self) -> None:
        s = Session(model_id="x")
        s.add_user_message("abcd")  # 4 chars → 1 token
        assert s.token_estimate() == 1

    def test_accumulates_across_messages(self) -> None:
        s = Session(model_id="x")
        s.add_user_message("a" * 40)   # 40 chars
        s.add_assistant_message("b" * 40)  # 40 chars → total 80 → 20 tokens
        assert s.token_estimate() == 20

    def test_rounds_down(self) -> None:
        s = Session(model_id="x")
        s.add_user_message("abc")  # 3 chars → 0 tokens (floor division)
        assert s.token_estimate() == 0

    def test_long_content_scales_linearly(self) -> None:
        s = Session(model_id="x")
        s.add_user_message("x" * 400)
        assert s.token_estimate() == 100

    def test_does_not_count_role_in_estimate(self) -> None:
        """Only message content should contribute to the estimate."""
        s = Session(model_id="x")
        s.add_user_message("hello")  # "user" role NOT counted
        expected = len("hello") // 4
        assert s.token_estimate() == expected


# ---------------------------------------------------------------------------
# Session — to_openrouter_messages
# ---------------------------------------------------------------------------


class TestToOpenRouterMessages:
    def test_empty_session_returns_empty_list(self) -> None:
        s = Session(model_id="x")
        assert s.to_openrouter_messages() == []

    def test_single_message_format(self) -> None:
        s = Session(model_id="x")
        s.add_user_message("Hello!")
        result = s.to_openrouter_messages()
        assert result == [{"role": "user", "content": "Hello!"}]

    def test_preserves_order_and_all_roles(self) -> None:
        s = Session(model_id="x")
        s.add_system_message("sys")
        s.add_user_message("usr")
        s.add_assistant_message("ast")
        result = s.to_openrouter_messages()
        assert result == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "ast"},
        ]

    def test_keys_are_exactly_role_and_content(self) -> None:
        s = Session(model_id="x")
        s.add_user_message("test")
        msg = s.to_openrouter_messages()[0]
        assert set(msg.keys()) == {"role", "content"}

    def test_original_messages_not_mutated(self) -> None:
        s = Session(model_id="x")
        s.add_user_message("hi")
        wire = s.to_openrouter_messages()
        wire[0]["role"] = "HACKED"
        # Session's own messages must be unchanged
        assert s.messages[0].role == "user"
