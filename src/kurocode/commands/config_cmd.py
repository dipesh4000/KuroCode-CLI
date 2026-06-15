"""
config command for KuroCode.
"""

import click
import os
import tomllib
from typing import Any
from pathlib import Path

from kurocode.types import CliContext
from kurocode.infra.config import load_config

_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "kurocode" / "config.toml"


@click.group(name="config")
def config_cmd_group() -> None:
    """Manage KuroCode configuration."""
    pass


def _dump_toml(d: dict[str, Any]) -> str:
    """Naive TOML dumper for KuroCode config structures."""
    lines = []
    
    def format_val(val: Any) -> str:
        if isinstance(val, bool):
            return str(val).lower()
        if isinstance(val, (int, float)):
            return str(val)
        # default to string with basic escaping
        return '"' + str(val).replace('"', '\\"') + '"'

    # Root keys
    for k, v in d.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {format_val(v)}")
            
    # Sections
    for k, v in d.items():
        if isinstance(v, dict):
            if k == "profiles":
                for pk, pv in v.items():
                    lines.append(f"\n[profiles.{pk}]")
                    if isinstance(pv, dict):
                        for pkk, pvv in pv.items():
                            lines.append(f"{pkk} = {format_val(pvv)}")
            else:
                lines.append(f"\n[{k}]")
                for sk, sv in v.items():
                    lines.append(f"{sk} = {format_val(sv)}")
                    
    return "\n".join(lines).strip() + "\n"


@config_cmd_group.command(name="set")
@click.argument("key")
@click.argument("value")
@click.option("--profile", default=None, help="Profile to set the key in (defaults to 'default').")
@click.pass_obj
def set_config(ctx: CliContext, key: str, value: str, profile: str | None) -> None:
    """Set a configuration value."""
    config_env = os.environ.get("KUROCODE_CONFIG")
    toml_path = Path(config_env) if config_env else _DEFAULT_CONFIG_PATH
    
    if toml_path.exists():
        with toml_path.open("rb") as f:
            doc = tomllib.load(f)
    else:
        doc = {}
        toml_path.parent.mkdir(parents=True, exist_ok=True)
        
    if profile:
        doc.setdefault("profiles", {})
        doc["profiles"].setdefault(profile, {})
        target = doc["profiles"][profile]
    else:
        doc.setdefault("default", {})
        target = doc["default"]
        
    # Attempt to cast value to int or float if possible
    final_value: Any = value
    if value.lower() == "true":
        final_value = True
    elif value.lower() == "false":
        final_value = False
    else:
        try:
            if "." in value:
                final_value = float(value)
            else:
                final_value = int(value)
        except ValueError:
            pass
            
    target[key] = final_value
    
    with toml_path.open("w", encoding="utf-8") as f:
        f.write(_dump_toml(doc))
        
    p_name = f"profiles.{profile}" if profile else "default"
    ctx.renderer.success(f"Set '{key}' = '{value}' in [{p_name}].")


@config_cmd_group.command(name="list")
@click.option("--profile", default=None, help="Profile to list configuration for.")
@click.pass_obj
def list_config(ctx: CliContext, profile: str | None) -> None:
    """List current resolved configuration."""
    if profile:
        cfg = load_config(profile=profile)
    else:
        cfg = ctx.config
    
    data = cfg.model_dump(mode="json")
    for k, v in data.items():
        if k == "api_key":
            v = "***" if v else ""
        ctx.renderer.info(f"{k} = {v}")
