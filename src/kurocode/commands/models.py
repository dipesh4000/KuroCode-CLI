"""
models command for KuroCode.
"""

import click
import asyncio

from kurocode.types import CliContext
from kurocode.core.model_registry import ModelRegistry


@click.group(name="models")
def models_cmd() -> None:
    """Discover and list models."""
    pass


@models_cmd.command(name="list")
@click.option("--refresh", is_flag=True, help="Force refresh from OpenRouter.")
@click.pass_obj
def list_models(ctx: CliContext, refresh: bool) -> None:
    """List available free models."""
    asyncio.run(run_list(ctx, refresh))


async def run_list(ctx: CliContext, refresh: bool) -> None:
    registry = ModelRegistry()
    models = await registry.fetch(force_refresh=refresh)
    ctx.renderer.model_table(models)


@models_cmd.command(name="search")
@click.argument("query")
@click.pass_obj
def search_models(ctx: CliContext, query: str) -> None:
    """Search for a model by name or ID."""
    asyncio.run(run_search(ctx, query))


async def run_search(ctx: CliContext, query: str) -> None:
    registry = ModelRegistry()
    models = await registry.fetch()
    q = query.lower()
    matches = [m for m in models if q in m.id.lower() or q in m.name.lower()]
    if not matches:
        ctx.renderer.error(f"No models found matching '{query}'.")
    else:
        ctx.renderer.model_table(matches)


@models_cmd.command(name="info")
@click.argument("model_id")
@click.pass_obj
def info_model(ctx: CliContext, model_id: str) -> None:
    """Show details for a specific model."""
    asyncio.run(run_info(ctx, model_id))


async def run_info(ctx: CliContext, model_id: str) -> None:
    registry = ModelRegistry()
    models = await registry.fetch()
    for m in models:
        if m.id.lower() == model_id.lower():
            ctx.renderer.info(f"Model ID: {m.id}")
            ctx.renderer.info(f"Name: {m.name}")
            ctx.renderer.info(f"Context Length: {m.context_length}")
            ctx.renderer.info(f"Description: {m.description}")
            return
    ctx.renderer.error(f"Model '{model_id}' not found.")
