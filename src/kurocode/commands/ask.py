"""
ask command for KuroCode.
"""

import sys
import json
import asyncio
import click

from kurocode.types import CliContext
from kurocode.core.session import Session
from kurocode.core.renderer import OutputFormat
from kurocode.infra.openrouter_client import OpenRouterClient

@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--model",
    default="openai/gpt-4o-mini",
    help="Model to use for the response.",
)
@click.pass_obj
def ask_cmd(ctx: CliContext, prompt: str | None, model: str) -> None:
    """Ask a single question and get a response."""
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            ctx.renderer.error("No prompt provided. Please pass a prompt or pipe to stdin.")
            sys.exit(1)
            
    if not prompt:
        ctx.renderer.error("Empty prompt provided.")
        sys.exit(1)

    asyncio.run(run_ask(ctx, prompt, model))


async def run_ask(ctx: CliContext, prompt: str, model: str) -> None:
    session = Session(model_id=model)
    session.add_user_message(prompt)
    messages = session.to_openrouter_messages()

    async with OpenRouterClient(ctx.config) as client:
        if ctx.no_stream:
            resp = await client.chat(messages=messages, model=model)
            content = resp.choices[0].message.content
            
            # Pipe-friendly JSON output for `jq .content`
            if ctx.renderer._fmt == OutputFormat.JSON:
                ctx.renderer._console.print(
                    json.dumps({"content": content}),
                    markup=False,
                    highlight=False
                )
            else:
                ctx.renderer.stream_token(content)
                ctx.renderer.end_stream()
        else:
            try:
                async for chunk in client.stream_chat(messages=messages, model=model):
                    delta = chunk.choices[0].delta.content
                    if delta:
                        ctx.renderer.stream_token(delta)
                ctx.renderer.end_stream()
            except KeyboardInterrupt:
                ctx.renderer.end_stream()
                ctx.renderer.error("\n[Interrupted]")
