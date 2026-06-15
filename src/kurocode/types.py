"""
Shared types and context for KuroCode CLI.
"""

from dataclasses import dataclass
from kurocode.core.renderer import Renderer
from kurocode.infra.config import Settings


@dataclass
class CliContext:
    """Context object passed to all click commands."""

    renderer: Renderer
    config: Settings
    no_stream: bool
