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
"""AgentApp loading utils."""


from __future__ import annotations

from collections.abc import Callable
from logging import DEBUG
from pathlib import Path

from flwr.common.config import (
    get_metadata_from_config,
    get_project_config,
    get_project_dir,
)
from flwr.common.logger import log
from flwr.common.object_ref import load_app, validate
from flwr.supercore.utils import get_flwr_home

from .agent_app import AgentApp
from .exceptions import LoadAgentAppError


def get_load_agent_app_fn(
    default_app_ref: str,
    app_path: str | None,
    multi_app: bool,
) -> Callable[[str, str, str], AgentApp]:
    """Get the load_agent_app_fn function."""
    if not multi_app:
        log(
            DEBUG,
            "Flower SuperLink will load and validate AgentApp `%s`",
            default_app_ref,
        )
        valid, error_msg = validate(default_app_ref, project_dir=app_path)
        if not valid and error_msg:
            raise LoadAgentAppError(error_msg) from None

    def _load(fab_id: str, fab_version: str, fab_hash: str) -> AgentApp:
        runtime_app_dir = Path(app_path if app_path else "").absolute()
        if not multi_app:
            agent_app_ref = default_app_ref
        elif app_path is not None:
            config = get_project_config(runtime_app_dir)
            this_fab_id, this_fab_version = get_metadata_from_config(config)
            if this_fab_version != fab_version or this_fab_id != fab_id:
                raise LoadAgentAppError(
                    f"FAB ID or version mismatch: Expected FAB ID '{this_fab_id}' and "
                    f"FAB version '{this_fab_version}', but received FAB ID '{fab_id}' "
                    f"and FAB version '{fab_version}'.",
                ) from None
            agent_app_ref = _get_agent_app_ref(config)
        else:
            try:
                runtime_app_dir = get_project_dir(fab_id, fab_version, fab_hash)
                config = get_project_config(runtime_app_dir)
            except Exception as err:
                raise LoadAgentAppError(
                    "Failed to load AgentApp. Possible reasons for error include "
                    "mismatched `fab_id`, `fab_version`, or `fab_hash` in "
                    f"{str(get_flwr_home().resolve())}."
                ) from err
            agent_app_ref = _get_agent_app_ref(config)

        log(DEBUG, "Loading AgentApp `%s`", agent_app_ref)
        agent_app = load_app(agent_app_ref, LoadAgentAppError, runtime_app_dir)
        if not isinstance(agent_app, AgentApp):
            raise LoadAgentAppError(
                f"Attribute {agent_app_ref} is not of type {AgentApp}",
            ) from None
        return agent_app

    return _load


def _get_agent_app_ref(config: dict[str, object]) -> str:
    """Return the AgentApp object reference from project config."""
    tool = config.get("tool")
    if not isinstance(tool, dict):
        raise LoadAgentAppError("Missing [tool.flwr.app.components].agentapp.")
    flwr = tool.get("flwr")
    if not isinstance(flwr, dict):
        raise LoadAgentAppError("Missing [tool.flwr.app.components].agentapp.")
    app = flwr.get("app")
    if not isinstance(app, dict):
        raise LoadAgentAppError("Missing [tool.flwr.app.components].agentapp.")
    components = app.get("components")
    if not isinstance(components, dict):
        raise LoadAgentAppError("Missing [tool.flwr.app.components].agentapp.")
    agent_app_ref = components.get("agentapp")
    if not isinstance(agent_app_ref, str) or not agent_app_ref:
        raise LoadAgentAppError("Missing [tool.flwr.app.components].agentapp.")
    return agent_app_ref
