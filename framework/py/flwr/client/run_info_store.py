# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Deprecated Run Info Store."""


from dataclasses import dataclass
from pathlib import Path

from flwr.app.user_config import UserConfig
from flwr.common import Context, RecordDict
from flwr.common.config import (
    get_fused_config,
    get_fused_config_from_dir,
    get_fused_config_from_fab,
)
from flwr.common.typing import Fab, Run


@dataclass()
class RunInfo:
    """Contains the Context and initial run_config of a Run."""

    context: Context
    initial_run_config: UserConfig


class DeprecatedRunInfoStore:
    """State of a node where client nodes execute runs."""

    def __init__(
        self,
        node_id: int,
        node_config: UserConfig,
    ) -> None:
        self.node_id = node_id
        self.node_config = node_config
        self.run_infos: dict[int, RunInfo] = {}

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def register_context(
        self,
        run_id: int,
        run: Run | None = None,
        app_dir: str | None = None,
        fab: Fab | None = None,
    ) -> None:
        """Register new run context for this node."""
        if run_id not in self.run_infos:
            initial_run_config = {}
            if app_dir:
                # Load from app directory
                app_path = Path(app_dir)
                if app_path.is_dir():
                    override_config = run.override_config if run else {}
                    initial_run_config = get_fused_config_from_dir(
                        app_path, override_config
                    )
                else:
                    raise ValueError("The specified `app_dir` must be a directory.")
            else:
                if run:
                    if fab:
                        # Load pyproject.toml from FAB file and fuse
                        initial_run_config = get_fused_config_from_fab(fab.content, run)
                    else:
                        # Load pyproject.toml from installed FAB and fuse
                        initial_run_config = get_fused_config(run)
                else:
                    initial_run_config = {}
            self.run_infos[run_id] = RunInfo(
                initial_run_config=initial_run_config,
                context=Context(
                    run_id=run_id,
                    node_id=self.node_id,
                    node_config=self.node_config,
                    state=RecordDict(),
                    run_config=initial_run_config.copy(),
                ),
            )

    def retrieve_context(self, run_id: int) -> Context:
        """Get run context given a run_id."""
        if run_id in self.run_infos:
            return self.run_infos[run_id].context

        raise RuntimeError(
            f"Context for run_id={run_id} doesn't exist."
            " A run context must be registered before it can be retrieved or updated "
            " by a client."
        )

    def update_context(self, run_id: int, context: Context) -> None:
        """Update run context."""
        if context.run_config != self.run_infos[run_id].initial_run_config:
            raise ValueError(
                "The `run_config` field of the `Context` object cannot be "
                f"modified (run_id: {run_id})."
            )
        self.run_infos[run_id].context = context
