"""
Main CLI entry point for KuroCode.
"""

from __future__ import annotations

import sys
from typing import Any

import click

from kurocode.core.renderer import OutputFormat, Renderer
from kurocode.exceptions import KurocodeError
from kurocode.infra.config import load_config
from kurocode.types import CliContext

# We will import commands lazily or directly below.
from kurocode.commands.ask import ask_cmd
from kurocode.commands.chat import chat_cmd
from kurocode.commands.models import models_cmd
from kurocode.commands.history import history_cmd
from kurocode.commands.config_cmd import config_cmd_group


class KuroCodeGroup(click.Group):
    """Custom Click Group that catches KurocodeError globally."""

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except KurocodeError as exc:
            # Safely grab the renderer if ctx.obj was initialised
            renderer = None
            if ctx.obj is not None and hasattr(ctx.obj, "renderer"):
                renderer = ctx.obj.renderer
            
            if renderer is None:
                # Fallback if config loading failed before ctx.obj was set
                fmt_str = ctx.params.get("output_format", "rich")
                fmt = OutputFormat(fmt_str)
                renderer = Renderer(fmt=fmt)

            renderer.error(str(exc), hint=getattr(exc, "hint", None))
            sys.exit(1)


@click.group(cls=KuroCodeGroup)
@click.option(
    "--profile",
    type=str,
    default=None,
    help="Configuration profile to use from config.toml.",
)
@click.option(
    "--output-format",
    type=click.Choice(["rich", "plain", "json"]),
    default="rich",
    help="Output format to use.",
)
@click.option(
    "--no-stream",
    is_flag=True,
    help="Disable streaming output (wait for full response).",
)
@click.pass_context
def cli(
    ctx: click.Context,
    profile: str | None,
    output_format: str,
    no_stream: bool,
) -> None:
    """KuroCode: An agentic coding assistant CLI."""
    fmt = OutputFormat(output_format)
    renderer = Renderer(fmt=fmt)
    config = load_config(profile=profile)
    
    ctx.obj = CliContext(
        renderer=renderer,
        config=config,
        no_stream=no_stream,
    )


cli.add_command(ask_cmd, "ask")
cli.add_command(chat_cmd, "chat")
cli.add_command(models_cmd, "models")
cli.add_command(history_cmd, "history")
cli.add_command(config_cmd_group, "config")


if __name__ == "__main__":
    cli()
