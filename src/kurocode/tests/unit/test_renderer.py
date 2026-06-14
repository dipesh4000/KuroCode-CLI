"""
Unit tests for kurocode.renderer.

All tests inject a ``rich.Console(file=io.StringIO(...))`` so they capture
output without touching stdout/stderr — pure, no mocking, no tmp_path.
"""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

from kurocode.core.model_registry import ModelInfo
from kurocode.core.renderer import OutputFormat, Renderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_renderer(fmt: OutputFormat) -> tuple[Renderer, io.StringIO]:
    """Return a (Renderer, buffer) pair that captures all console output."""
    buf = io.StringIO()
    console = Console(
        file=buf,
        markup=(fmt == OutputFormat.RICH),
        highlight=False,
        width=120,
        no_color=True,  # strip ANSI so assertions on plain text work
    )
    renderer = Renderer(fmt=fmt, console=console)
    return renderer, buf


def _output(buf: io.StringIO) -> str:
    return buf.getvalue()


def _free_model(
    idx: int = 0,
    context_length: int = 32768,
) -> ModelInfo:
    return ModelInfo(
        id=f"vendor/model-{idx}:free",
        name=f"Model {idx}",
        context_length=context_length,
        pricing={"prompt": "0", "completion": "0"},
    )


# ---------------------------------------------------------------------------
# OutputFormat enum
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_all_values_accessible(self) -> None:
        assert OutputFormat.RICH.value == "rich"
        assert OutputFormat.PLAIN.value == "plain"
        assert OutputFormat.JSON.value == "json"

    def test_three_members(self) -> None:
        assert len(list(OutputFormat)) == 3


# ---------------------------------------------------------------------------
# Renderer construction
# ---------------------------------------------------------------------------


class TestRendererConstruction:
    def test_accepts_injected_console(self) -> None:
        buf = io.StringIO()
        console = Console(file=buf)
        r = Renderer(fmt=OutputFormat.PLAIN, console=console)
        r.success("ok")
        assert buf.getvalue()  # something was written

    def test_default_format_is_rich(self) -> None:
        buf = io.StringIO()
        console = Console(file=buf, no_color=True)
        r = Renderer(console=console)
        assert r._fmt == OutputFormat.RICH


# ---------------------------------------------------------------------------
# stream_token
# ---------------------------------------------------------------------------


class TestStreamToken:
    def test_plain_writes_token_without_newline(self) -> None:
        r, buf = _make_renderer(OutputFormat.PLAIN)
        r.stream_token("Hello")
        r.stream_token(" world")
        out = _output(buf)
        # No newlines between consecutive tokens
        assert "Hello world" in out

    def test_plain_multiple_tokens_concatenate(self) -> None:
        r, buf = _make_renderer(OutputFormat.PLAIN)
        tokens = ["The", " quick", " brown", " fox"]
        for t in tokens:
            r.stream_token(t)
        out = _output(buf)
        assert "The quick brown fox" in out

    def test_json_emits_one_object_per_token(self) -> None:
        r, buf = _make_renderer(OutputFormat.JSON)
        r.stream_token("abc")
        lines = [ln for ln in _output(buf).splitlines() if ln.strip()]
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj == {"type": "token", "data": "abc"}

    def test_json_preserves_token_content_exactly(self) -> None:
        r, buf = _make_renderer(OutputFormat.JSON)
        r.stream_token('special "chars" & <html>')
        line = _output(buf).strip()
        obj = json.loads(line)
        assert obj["data"] == 'special "chars" & <html>'

    def test_end_stream_adds_newline(self) -> None:
        r, buf = _make_renderer(OutputFormat.PLAIN)
        r.stream_token("tok")
        r.end_stream()
        out = _output(buf)
        assert out.endswith("\n")

    def test_end_stream_noop_for_json(self) -> None:
        r, buf = _make_renderer(OutputFormat.JSON)
        r.stream_token("tok")
        before_len = len(_output(buf))
        r.end_stream()
        assert len(_output(buf)) == before_len  # no extra output


# ---------------------------------------------------------------------------
# model_table
# ---------------------------------------------------------------------------


class TestModelTable:
    def test_plain_includes_all_model_ids(self) -> None:
        r, buf = _make_renderer(OutputFormat.PLAIN)
        models = [_free_model(0), _free_model(1)]
        r.model_table(models)
        out = _output(buf)
        assert "vendor/model-0:free" in out
        assert "vendor/model-1:free" in out

    def test_plain_includes_header(self) -> None:
        r, buf = _make_renderer(OutputFormat.PLAIN)
        r.model_table([_free_model()])
        out = _output(buf)
        assert "ID" in out
        assert "NAME" in out

    def test_plain_empty_list_renders_header_only(self) -> None:
        r, buf = _make_renderer(OutputFormat.PLAIN)
        r.model_table([])
        out = _output(buf)
        assert "ID" in out

    def test_json_emits_valid_json_array(self) -> None:
        r, buf = _make_renderer(OutputFormat.JSON)
        models = [_free_model(0), _free_model(1)]
        r.model_table(models)
        out = _output(buf).strip()
        payload = json.loads(out)
        assert isinstance(payload, list)
        assert len(payload) == 2

    def test_json_contains_required_keys(self) -> None:
        r, buf = _make_renderer(OutputFormat.JSON)
        r.model_table([_free_model()])
        payload = json.loads(_output(buf).strip())
        assert "id" in payload[0]
        assert "name" in payload[0]
        assert "context_length" in payload[0]

    def test_json_context_length_is_int(self) -> None:
        r, buf = _make_renderer(OutputFormat.JSON)
        r.model_table([_free_model(context_length=131072)])
        payload = json.loads(_output(buf).strip())
        assert payload[0]["context_length"] == 131072

    def test_rich_includes_model_id(self) -> None:
        r, buf = _make_renderer(OutputFormat.RICH)
        r.model_table([_free_model(0)])
        out = _output(buf)
        assert "vendor/model-0:free" in out


# ---------------------------------------------------------------------------
# error
# ---------------------------------------------------------------------------


class TestError:
    def test_plain_includes_error_label(self) -> None:
        r, buf = _make_renderer(OutputFormat.PLAIN)
        r.error("Something failed")
        assert "Error: Something failed" in _output(buf)

    def test_plain_includes_hint_when_provided(self) -> None:
        r, buf = _make_renderer(OutputFormat.PLAIN)
        r.error("Oops", hint="Try again later")
        out = _output(buf)
        assert "Hint: Try again later" in out

    def test_plain_no_hint_text_when_omitted(self) -> None:
        r, buf = _make_renderer(OutputFormat.PLAIN)
        r.error("Oops")
        out = _output(buf)
        assert "Hint" not in out

    def test_json_type_is_error(self) -> None:
        r, buf = _make_renderer(OutputFormat.JSON)
        r.error("boom")
        obj = json.loads(_output(buf).strip())
        assert obj["type"] == "error"
        assert obj["message"] == "boom"

    def test_json_hint_included_when_given(self) -> None:
        r, buf = _make_renderer(OutputFormat.JSON)
        r.error("boom", hint="fix it")
        obj = json.loads(_output(buf).strip())
        assert obj["hint"] == "fix it"

    def test_json_hint_absent_when_not_given(self) -> None:
        r, buf = _make_renderer(OutputFormat.JSON)
        r.error("boom")
        obj = json.loads(_output(buf).strip())
        assert "hint" not in obj

    def test_rich_includes_message_text(self) -> None:
        r, buf = _make_renderer(OutputFormat.RICH)
        r.error("Something went wrong", hint="check logs")
        out = _output(buf)
        assert "Something went wrong" in out
        assert "check logs" in out


# ---------------------------------------------------------------------------
# success
# ---------------------------------------------------------------------------


class TestSuccess:
    def test_plain_includes_ok_prefix(self) -> None:
        r, buf = _make_renderer(OutputFormat.PLAIN)
        r.success("All done")
        assert "OK: All done" in _output(buf)

    def test_json_type_is_success(self) -> None:
        r, buf = _make_renderer(OutputFormat.JSON)
        r.success("yep")
        obj = json.loads(_output(buf).strip())
        assert obj["type"] == "success"
        assert obj["message"] == "yep"

    def test_rich_includes_message(self) -> None:
        r, buf = _make_renderer(OutputFormat.RICH)
        r.success("Done!")
        assert "Done!" in _output(buf)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


class TestInfo:
    def test_plain_prints_message(self) -> None:
        r, buf = _make_renderer(OutputFormat.PLAIN)
        r.info("Loading models…")
        assert "Loading models" in _output(buf)

    def test_json_type_is_info(self) -> None:
        r, buf = _make_renderer(OutputFormat.JSON)
        r.info("Loading")
        obj = json.loads(_output(buf).strip())
        assert obj["type"] == "info"

    def test_rich_prints_message(self) -> None:
        r, buf = _make_renderer(OutputFormat.RICH)
        r.info("hello")
        assert "hello" in _output(buf)
