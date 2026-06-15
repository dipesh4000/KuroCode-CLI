"""
history command for KuroCode.
"""

import click
import asyncio
from datetime import datetime

from kurocode.types import CliContext
from kurocode.infra.store import ConversationStore


@click.group(name="history")
def history_cmd() -> None:
    """View and export conversation history."""
    pass


@history_cmd.command(name="list")
@click.option("--limit", default=20, type=int, help="Number of conversations to show.")
@click.pass_obj
def list_history(ctx: CliContext, limit: int) -> None:
    """List recent conversations."""
    asyncio.run(run_list(ctx, limit))


async def run_list(ctx: CliContext, limit: int) -> None:
    async with ConversationStore(ctx.config.db_path) as store:
        conversations = await store.list_conversations(limit=limit)
        
    if not conversations:
        ctx.renderer.info("No conversations found.")
        return
        
    for c in conversations:
        dt = datetime.fromtimestamp(c.created_at).strftime("%Y-%m-%d %H:%M:%S")
        ctx.renderer.info(f"[{c.id}] {dt} - {c.model} - {c.title}")


@history_cmd.command(name="view")
@click.argument("conv_id")
@click.pass_obj
def view_history(ctx: CliContext, conv_id: str) -> None:
    """View a specific conversation by ID."""
    asyncio.run(run_run(ctx, conv_id))


async def run_run(ctx: CliContext, conv_id: str) -> None:
    async with ConversationStore(ctx.config.db_path) as store:
        messages = await store.get_messages(conv_id)
        
    if not messages:
        ctx.renderer.error(f"Conversation '{conv_id}' not found or empty.")
        return
        
    for m in messages:
        dt = datetime.fromtimestamp(m.created_at).strftime("%Y-%m-%d %H:%M:%S")
        ctx.renderer.info(f"[{dt}] **{m.role.upper()}**:\n{m.content}\n")


@history_cmd.command(name="export")
@click.argument("conv_id")
@click.option("--format", "fmt", type=click.Choice(["markdown"]), default="markdown", help="Export format")
@click.pass_obj
def export_history(ctx: CliContext, conv_id: str, fmt: str) -> None:
    """Export a specific conversation."""
    asyncio.run(run_export(ctx, conv_id, fmt))


async def run_export(ctx: CliContext, conv_id: str, fmt: str) -> None:
    async with ConversationStore(ctx.config.db_path) as store:
        messages = await store.get_messages(conv_id)
        
    if not messages:
        ctx.renderer.error(f"Conversation '{conv_id}' not found or empty.")
        return
        
    for m in messages:
        click.echo(f"**{m.role.capitalize()}**:\n{m.content}\n")
