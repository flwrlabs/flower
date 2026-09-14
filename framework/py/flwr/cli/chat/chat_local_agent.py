# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Local AgentApp helpers for Flower Chat."""

import hashlib
import shlex
from dataclasses import dataclass
from pathlib import Path

import click

from flwr.cli.build import build_fab_from_disk
from flwr.cli.config_utils import load_and_validate
from flwr.cli.constant import CHAT_LOAD_COMMAND
from flwr.common.config import get_metadata_from_config
from flwr.common.constant import FAB_CONFIG_FILE

from ..utils import AppPathDepthError


@dataclass(frozen=True)
class LocalAgent:
    """A local AgentApp built for use in the current chat session."""

    path: Path
    app_spec: str
    fab_hash: str
    fab_content: bytes
    warnings: tuple[str, ...]


def parse_local_agent_path(prompt: str) -> Path:
    """Extract the local AgentApp path from a load command."""
    try:
        lexer = shlex.shlex(prompt, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        lexer.escape = ""
        parts = list(lexer)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    if len(parts) != 2:
        raise click.ClickException(f"Usage: {CHAT_LOAD_COMMAND} <path>")
    return Path(parts[1])


def build_local_agent(path: Path) -> LocalAgent:
    """Build a local AgentApp for use in the current chat session."""
    try:
        path = path.expanduser().resolve()
        if not path.is_dir():
            raise click.ClickException(
                f"The path '{path}' is not a valid path to a Flower App."
            )
        config, warnings = load_and_validate(path / FAB_CONFIG_FILE, check_module=False)
        components = config["tool"]["flwr"]["app"].get("components", {})
        if "agentapp" not in components:
            raise click.ClickException(
                f"The Flower App at '{path}' does not define an AgentApp component."
            )
        fab_content = build_fab_from_disk(path)
    except AppPathDepthError as exc:
        raise exc.to_click_exception() from None
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from None

    fab_hash = hashlib.sha256(fab_content).hexdigest()
    fab_id, _ = get_metadata_from_config(config)
    return LocalAgent(
        path=path,
        app_spec=f"@{fab_id}",
        fab_hash=fab_hash,
        fab_content=fab_content,
        warnings=tuple(warnings),
    )


def format_local_agent_success(local_agent: LocalAgent) -> str:
    """Describe a successfully loaded local AgentApp."""
    return f"Loaded {local_agent.app_spec} from {local_agent.path}.\n\n"


def format_local_agent_failure(error: click.ClickException) -> str:
    """Describe a failed local AgentApp load operation."""
    return (
        f"Load failed: {error.format_message()}\n"
        "The previously selected agent remains selected.\n\n"
    )
