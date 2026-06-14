"""
Output renderer for KuroCode.

Wraps ``rich.Console`` and supports three output formats:

* ``OutputFormat.RICH``  — colour / markup via rich (default in a TTY)
* ``OutputFormat.PLAIN`` — plain text, no ANSI codes
* ``OutputFormat.JSON``  — one JSON object per logical event (for scripting)

Because the ``Console`` instance is injected via the constructor, tests can
pass a ``Console(file=io.StringIO(...))`` to capture output without touching
real stdout/stderr — **no mocking needed**.

Usage::

    from kurocode.renderer import Renderer, OutputFormat

    r = Renderer()                     # auto-detect: RICH when TTY present
    r.stream_token("Hello")
    r.end_stream()
    r.success("Done!")
    r.error("Something went wrong", hint="Check your API key.")
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from rich.console import Console
from rich.table import Table

from kurocode.core.model_registry import ModelInfo


class OutputFormat(Enum):
    """Supported output rendering modes."""

    RICH = "rich"
    PLAIN = "plain"
    JSON = "json"


class Renderer:
    """
    Unified output renderer.

    Parameters
    ----------
    fmt:
        Output format to use.
    console:
        Pre-built ``rich.Console`` (useful for tests that need to capture
        output via ``io.StringIO``).  When *None* a suitable console is
        created automatically.
    """

    def __init__(
        self,
        fmt: OutputFormat = OutputFormat.RICH,
        console: Console | None = None,
    ) -> None:
        self._fmt = fmt
        if console is not None:
            self._console = console
        elif fmt == OutputFormat.RICH:
            self._console = Console()
        else:
            # PLAIN and JSON: strip all markup and highlighting
            self._console = Console(markup=False, highlight=False)

    # ------------------------------------------------------------------
    # Streaming tokens
    # ------------------------------------------------------------------

    def stream_token(self, token: str) -> None:
        """Write a single streaming token with no trailing newline."""
        if self._fmt == OutputFormat.JSON:
            self._console.print(
                json.dumps({"type": "token", "data": token}),
                markup=False,
                highlight=False,
            )
        else:
            self._console.print(token, end="", markup=False, highlight=False)

    def end_stream(self) -> None:
        """Emit a newline to finalise a streaming sequence."""
        if self._fmt != OutputFormat.JSON:
            self._console.print()

    # ------------------------------------------------------------------
    # Model table
    # ------------------------------------------------------------------

    def model_table(self, models: list[ModelInfo]) -> None:
        """
        Render *models* as a formatted table.

        * **RICH**: a ``rich.Table`` with columns for ID, name, and context.
        * **PLAIN**: tab-separated rows (header + one row per model).
        * **JSON**: a JSON array of ``{id, name, context_length}`` objects.
        """
        if self._fmt == OutputFormat.JSON:
            payload = [
                {"id": m.id, "name": m.name, "context_length": m.context_length}
                for m in models
            ]
            self._console.print(
                json.dumps(payload, indent=2), markup=False, highlight=False
            )
            return

        if self._fmt == OutputFormat.RICH:
            table = Table(title="Free Models", show_lines=True)
            table.add_column("Model ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="green")
            table.add_column("Context", justify="right", style="yellow")
            for m in models:
                table.add_row(m.id, m.name, f"{m.context_length:,}")
            self._console.print(table)
        else:  # PLAIN
            self._console.print("ID\tNAME\tCONTEXT", markup=False)
            for m in models:
                self._console.print(
                    f"{m.id}\t{m.name}\t{m.context_length}", markup=False
                )

    # ------------------------------------------------------------------
    # Status messages
    # ------------------------------------------------------------------

    def error(self, msg: str, hint: str | None = None) -> None:
        """Render an error message, with an optional *hint*."""
        if self._fmt == OutputFormat.JSON:
            payload: dict[str, Any] = {"type": "error", "message": msg}
            if hint:
                payload["hint"] = hint
            self._console.print(
                json.dumps(payload), markup=False, highlight=False
            )
            return

        if self._fmt == OutputFormat.RICH:
            self._console.print(f"[bold red]✗ Error:[/bold red] {msg}")
            if hint:
                self._console.print(f"[dim]  Hint: {hint}[/dim]")
        else:
            self._console.print(f"Error: {msg}", markup=False)
            if hint:
                self._console.print(f"  Hint: {hint}", markup=False)

    def success(self, msg: str) -> None:
        """Render a success message."""
        if self._fmt == OutputFormat.JSON:
            self._console.print(
                json.dumps({"type": "success", "message": msg}),
                markup=False,
                highlight=False,
            )
            return

        if self._fmt == OutputFormat.RICH:
            self._console.print(f"[bold green]✓[/bold green] {msg}")
        else:
            self._console.print(f"OK: {msg}", markup=False)

    def info(self, msg: str) -> None:
        """Render an informational message."""
        if self._fmt == OutputFormat.JSON:
            self._console.print(
                json.dumps({"type": "info", "message": msg}),
                markup=False,
                highlight=False,
            )
            return

        if self._fmt == OutputFormat.RICH:
            self._console.print(f"[blue]ℹ[/blue] {msg}")
        else:
            self._console.print(msg, markup=False)
